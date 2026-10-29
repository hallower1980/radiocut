"""No-LLM detection of breaks: keyword cues, transcript repetition, recurring-audio
fingerprints and speech gaps (music / jingles / dead air)."""
import re
from collections import defaultdict
import numpy as np

AD_CUES = [
    r"brought to you by", r"sponsored by", r"promo code", r"use (the )?code", r"call (now|today)",
    r"visit .{0,30}\.(com|org|net)", r"\b\w+\.(com|org|net)\b", r"dot com", r"1[- ]?800", r"\b8(00|88|77|66|55)[- ]",
    r"\d{3}[- ]\d{3}[- ]\d{4}", r"free (consultation|shipping|quote|trial)", r"limited time", r"% off",
    r"terms and conditions", r"restrictions apply", r"order (now|today)", r"financing", r"donate", r"donation",
    r"tax[- ]deductible", r"save \$?\d+", r"offer", r"discount", r"insurance", r"apply now",
]
PROMO_CUES = [
    r"stay (tuned|with us)", r"we'?ll be (right )?back", r"after (the|this) break", r"coming up",
    r"don'?t (go|move) away", r"you'?re listening to", r"this is .{0,40}(radio|network)",
    r"(radio|network)\b.{0,30}\b(station|affiliate)", r"download (the|our) (free )?app", r"podcast",
    r"subscribe", r"tune in", r"weekdays at", r"every (weekday|morning|afternoon|evening|sunday|saturday)",
    r"(am|fm) \d", r"\bhd\d?\b", r"listener[- ]supported", r"on air", r"join us",
    r"top of the hour", r"news (is )?next", r"traffic and weather",
]
_ad = [re.compile(p, re.I) for p in AD_CUES]
_promo = [re.compile(p, re.I) for p in PROMO_CUES]


def make_windows(transcript, size=30.0):
    """Fixed windows over the whole timeline (so gaps without speech are covered)."""
    dur = transcript["duration"]
    n = int(np.ceil(dur / size))
    wins = [{"start": i * size, "end": min(dur, (i + 1) * size), "text": "", "speech": 0.0}
            for i in range(n)]
    for s in transcript["segments"]:
        i = min(n - 1, int(s["start"] // size))
        wins[i]["text"] += " " + s["text"]
        # speech coverage, split across windows
        a, b = s["start"], s["end"]
        while a < b and i < n:
            e = min(b, wins[i]["end"])
            wins[i]["speech"] += max(0.0, e - a)
            a = e
            i += 1
    for w in wins:
        w["text"] = w["text"].strip()
        w["speech"] = w["speech"] / max(1e-6, w["end"] - w["start"])
    return wins


def cue_scores(text):
    ad = sum(len(p.findall(text)) for p in _ad)
    promo = sum(len(p.findall(text)) for p in _promo)
    return ad, promo


def _shingles(text, k=6):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def transcript_repeats(wins, min_gap=300.0):
    """Fraction of each window's 6-word shingles that also occur in a window at least
    min_gap seconds away: recurring ads/promos/IDs repeat; live conversation does not."""
    sh = [_shingles(w["text"]) for w in wins]
    where = defaultdict(set)
    for i, s in enumerate(sh):
        for g in s:
            where[g].add(i)
    out = []
    for i, s in enumerate(sh):
        if not s:
            out.append(0.0)
            continue
        hit = sum(1 for g in s if any(abs(wins[j]["start"] - wins[i]["start"]) >= min_gap for j in where[g]))
        out.append(hit / len(s))
    return out


def audio_repeats(audio, sr, wins, min_gap=300.0, hop=0.5):
    """Lightweight audio fingerprint: 16-band log-energy-difference bit patterns per
    0.5 s frame. Frames whose 32-bit codes recur far away (same jingle / spot replayed)
    are marked; returns per-window repeated fraction."""
    if audio is None or len(audio) < sr * 10:
        return [0.0] * len(wins)
    n = int(sr * 1.0)
    h = int(sr * hop)
    frames = (len(audio) - n) // h
    win = np.hanning(n).astype(np.float32)
    edges = np.geomspace(150, 4000, 18)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bands = [(freqs >= edges[i]) & (freqs < edges[i + 1]) for i in range(17)]
    codes = np.zeros(frames, np.int64)
    prev = None
    for f in range(frames):
        x = audio[f * h:f * h + n] * win
        if np.abs(x).max() < 1e-3:
            codes[f] = -1
            prev = None
            continue
        sp = np.abs(np.fft.rfft(x)) ** 2
        e = np.log(np.array([sp[b].sum() for b in bands]) + 1e-9)
        d = np.diff(e)  # 16 values
        bits = (d > 0).astype(np.int64)
        if prev is not None:
            bits = np.concatenate([bits, (e[:16] > prev[:16]).astype(np.int64)])
        else:
            bits = np.concatenate([bits, np.zeros(16, np.int64)])
        codes[f] = int((bits * (1 << np.arange(32))).sum())
        prev = e
    idx = defaultdict(list)
    for f, c in enumerate(codes):
        if c >= 0:
            idx[c].append(f)
    gapf = int(min_gap / hop)
    rep = np.zeros(frames, bool)
    # require runs: a match at offset d for frame f and also f+1,f+2 (same offset)
    for f, c in enumerate(codes):
        if c < 0 or len(idx[c]) > 50:
            continue
        for g in idx[c]:
            d = g - f
            if abs(d) < gapf:
                continue
            ok = all(0 <= f + k < frames and 0 <= g + k < frames and codes[f + k] == codes[g + k] and codes[f + k] >= 0
                     for k in (1, 2, 3))
            if ok:
                rep[f:f + 4] = True
                break
    out = []
    for w in wins:
        a, b = int(w["start"] / hop), int(w["end"] / hop)
        seg = rep[a:min(b, frames)]
        out.append(float(seg.mean()) if len(seg) else 0.0)
    return out


def heuristic_scores(wins, trep, arep):
    """Returns per-window break probability 0..1 and reasons."""
    res = []
    for w, tr, ar in zip(wins, trep, arep):
        ad, promo = cue_scores(w["text"])
        words = max(1, len(w["text"].split()))
        reasons = []
        s = 0.0
        if ad:
            s += min(0.45, 0.15 * ad); reasons.append(f"{ad} ad cue(s)")
        if promo:
            s += min(0.35, 0.12 * promo); reasons.append(f"{promo} promo cue(s)")
        if tr > 0.25:
            s += min(0.5, tr); reasons.append(f"transcript repeats elsewhere ({tr:.0%})")
        if ar > 0.15:
            s += min(0.5, ar * 1.5); reasons.append(f"audio repeats elsewhere ({ar:.0%})")
        if w["speech"] < 0.35:
            s += 0.3; reasons.append("little speech (music/jingle/silence)")
        res.append((min(1.0, s), reasons))
    return res
