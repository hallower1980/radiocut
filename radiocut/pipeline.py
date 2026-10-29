"""End-to-end: transcribe -> heuristics (+ optional local LLM) -> keep/remove segments."""
import json
from .util import load_audio, fmt_ts
from .transcribe import transcribe
from . import heuristics as H
from . import llm as L

DEFAULTS = {"whisper_model": "base.en", "window": 30.0, "threshold": 0.5,
            "min_break": 20.0, "min_keep": 60.0, "use_llm": False,
            "llm_base": L.PRESETS["LM Studio"], "llm_model": "", "llm_weight": 0.65}


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
    segs = to_segments(wins, tr, cfg)
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


def save_project(res, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
