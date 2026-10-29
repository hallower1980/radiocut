"""End-to-end: transcribe -> heuristics (+ optional local LLM) -> keep/remove segments."""
import json
from .util import load_audio, fmt_ts
from .transcribe import transcribe
from . import heuristics as H
from . import llm as L
from . import speakers as SP

DEFAULTS = {"whisper_model": "base.en", "window": 30.0, "threshold": 0.5,
            "min_break": 20.0, "min_keep": 60.0, "use_llm": False,
            "llm_base": L.PRESETS["LM Studio"], "llm_model": "", "llm_weight": 0.65,
            "host_focus": True, "host_minutes": 5.0}


def analyze(path, cfg=None, progress=None, log=print):
    cfg = {**DEFAULTS, **(cfg or {})}
    prog = progress or (lambda p, m: None)
    prog(0.0, "Decoding audio")
    audio = load_audio(path)
    tr = transcribe(path, cfg["whisper_model"], audio=audio,
                    progress=lambda p, m: prog(0.05 + 0.5 * p, m))
    wins = H.make_windows(tr, cfg["window"])
    prog(0.56, "Matching recurring ads/jingles")
    trep = H.transcript_repeats(wins)
    arep = H.audio_repeats(audio, 16000, wins)
    hf = None
    if cfg["host_focus"]:
        prog(0.57, "Identifying the host voice")
        hf = SP.host_focus(tr, audio, cfg, progress=lambda p, m: prog(0.57 + 0.03 * p, m), log=log)
    del audio
    heur = H.heuristic_scores(wins, trep, arep)
    probs, labels, reasons = [], [], []
    for i, (w, (hs, hr)) in enumerate(zip(wins, heur)):
        p, lab, why = hs, ("BREAK?" if hs >= cfg["threshold"] else "CONTENT"), "; ".join(hr)
        if cfg["use_llm"] and cfg["llm_model"]:
            try:
                lab, conf, r = L.classify(cfg["llm_base"], cfg["llm_model"],
                                          wins[i - 1]["text"] if i else "", w["text"],
                                          wins[i + 1]["text"] if i + 1 < len(wins) else "")
                lp = conf if lab != "CONTENT" else (1 - conf) * 0.3
                # heuristics alone can still flag (e.g. a replayed news brief); the LLM adds/boosts
                p = max(hs, cfg["llm_weight"] * lp + (1 - cfg["llm_weight"]) * hs)
                why = f"LLM: {lab} ({r})" + (f"; {why}" if why else "")
            except Exception as e:  # server down: fall back to heuristics
                log(f"LLM error at {fmt_ts(w['start'])}: {e}")
        probs.append(p); labels.append(lab); reasons.append(why)
        prog(0.6 + 0.38 * (i + 1) / len(wins), f"Classifying {i+1}/{len(wins)}")
        w.update(p=round(p, 3), label=lab, reason=why, trep=round(trep[i], 2), arep=round(arep[i], 2))
    segs = host_segments(hf, wins, tr, cfg) if hf else to_segments(wins, tr, cfg)
    prog(1.0, "Done")
    return {"source": path, "duration": tr["duration"], "windows": wins, "segments": segs,
            "transcript": tr["segments"], "config": cfg}


def to_segments(wins, tr, cfg):
    th = cfg["threshold"]
    # 3-window median smoothing
    p = [w["p"] for w in wins]
    sm = [sorted(p[i - 1:i + 2])[1] if 0 < i < len(p) - 1 else min(p[max(0, i - 1):i + 2])
          for i in range(len(p))]
    flag = [x >= th for x in sm]
    runs = []
    for i, f in enumerate(flag):
        if runs and runs[-1][2] == f:
            runs[-1][1] = i
        else:
            runs.append([i, i, f])
    def dur(r):
        return wins[r[1]]["end"] - wins[r[0]]["start"]
    changed = True
    while changed and len(runs) > 1:  # absorb too-short runs into neighbours
        changed = False
        for k, r in enumerate(runs):
            lim = cfg["min_break"] if r[2] else cfg["min_keep"]
            if dur(r) < lim and (k > 0 or k < len(runs) - 1):
                r[2] = not r[2]
                merged = []
                for x in runs:
                    if merged and merged[-1][2] == x[2]:
                        merged[-1][1] = x[1]
                    else:
                        merged.append(list(x))
                runs = merged; changed = True
                break
    # snap boundaries to nearest gap between transcript segments (+-8 s)
    gaps = [(a["end"] + b["start"]) / 2 for a, b in zip(tr["segments"], tr["segments"][1:])]
    def snap(t):
        c = [g for g in gaps if abs(g - t) <= 8]
        return min(c, key=lambda g: abs(g - t)) if c else t
    segs = []
    for r in runs:
        a = wins[r[0]]["start"]; b = wins[r[1]]["end"]
        rs = [wins[i]["reason"] for i in range(r[0], r[1] + 1) if wins[i]["reason"]]
        labs = [wins[i]["label"] for i in range(r[0], r[1] + 1)]
        lab = max(set(labs), key=labs.count) if r[2] else "CONTENT"
        if r[2] and lab == "CONTENT":
            lab = "BREAK"
        segs.append({"start": a, "end": b, "keep": not r[2], "label": lab,
                     "reason": " | ".join(dict.fromkeys(rs))[:300] if r[2] else "",
                     "preview": " ".join(wins[i]["text"] for i in range(r[0], min(r[1] + 1, r[0] + 2)))[:200]})
    for x, y in zip(segs, segs[1:]):
        t = snap(x["end"]); x["end"] = y["start"] = round(t, 2)
    return segs


def host_segments(hf, wins, tr, cfg):
    """Host-focus mode: keep/cut per transcript segment (speaker turns), boundaries in the
    pauses between Whisper segments so the host is never cut mid-sentence."""
    # optional: an LLM-confirmed ad/promo read by the host is still cut
    if cfg["use_llm"] and cfg["llm_model"]:
        for w in wins:
            if w["label"] in ("AD", "PROMO", "MUSIC") and w["p"] >= max(cfg["threshold"], 0.7):
                for s in hf:
                    mid = (s["start"] + s["end"]) / 2
                    if w["start"] <= mid < w["end"] and s["keep"]:
                        s["keep"], s["role"], s["llm"] = False, "HOST AD", w["reason"]
    runs = []
    for s in hf:
        if runs and runs[-1]["keep"] == s["keep"]:
            runs[-1]["items"].append(s)
        else:
            runs.append({"keep": s["keep"], "items": [s]})
    out = []
    for k, r in enumerate(runs):
        it = r["items"]
        a = 0.0 if k == 0 else round((runs[k - 1]["items"][-1]["end"] + it[0]["start"]) / 2, 2)
        b = tr["duration"] if k == len(runs) - 1 else round((it[-1]["end"] + runs[k + 1]["items"][0]["start"]) / 2, 2)
        roles = [x["role"] for x in it]
        if r["keep"]:
            lab = "HOST+GUEST" if any(x == "GUEST" for x in roles) else "HOST"
            why = ""
        else:
            lab = "HOST AD" if "HOST AD" in roles else "OTHER VOICE"
            spk = sorted({x.get("speaker", "") for x in it if x.get("speaker")})
            sim = max(x["hostsim"] for x in it)
            exm = max([x.get("exchanges", 0) for x in it] or [0])
            why = (f"not the host (max host similarity {sim:.2f}); voices {', '.join(spk)[:60]}; "
                   f"{exm} turn exchanges with host nearby")
            ll = [x["llm"] for x in it if x.get("llm")]
            if ll: why = ll[0]
        out.append({"start": a, "end": b, "keep": r["keep"], "label": lab, "reason": why,
                    "preview": " ".join(x["text"] for x in it[:3])[:200]})
    return out


def save_project(res, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
