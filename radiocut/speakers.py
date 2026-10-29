"""Offline speaker embeddings (WeSpeaker CAM++ ONNX, ~29 MB, run with onnxruntime) and
host-focus logic: enroll host voice(s) from the opening minutes, then keep the host and
the people the host talks with directly."""
import os, urllib.request
import numpy as np
from .util import app_dir

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx")
MODEL_FILE = "wespeaker_en_voxceleb_CAMpp.onnx"
SR = 16000
WIN, HOP = 2.0, 0.5  # embedding window / hop in seconds


def model_path(progress=None):
    d = os.path.join(app_dir(), "models"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, MODEL_FILE)
    if not os.path.exists(p):
        if progress: progress(0.0, "Downloading speaker model (29 MB, first run only)")
        urllib.request.urlretrieve(MODEL_URL, p + ".part"); os.replace(p + ".part", p)
    return p


def _mel_bank(n_fft=512, n_mels=80, lo=20.0, hi=8000.0):
    mel = lambda f: 1127.0 * np.log(1 + f / 700.0)
    m = np.linspace(mel(lo), mel(hi), n_mels + 2)
    fr = np.fft.rfftfreq(n_fft, 1 / SR); fm = mel(fr)
    B = np.zeros((n_mels, len(fr)), np.float32)
    for i in range(n_mels):
        l, c, r = m[i], m[i + 1], m[i + 2]
        B[i] = np.clip(np.minimum((fm - l) / (c - l), (r - fm) / (r - c)), 0, None)
    return B


def fbank(audio):
    """Kaldi-style 80-dim log-mel fbank (25 ms / 10 ms, povey window, preemphasis)."""
    x = audio.astype(np.float32) * 32768.0
    fl, fs = 400, 160
    n = 1 + (len(x) - fl) // fs
    if n <= 0:
        return np.zeros((0, 80), np.float32)
    B = _mel_bank().T
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fl) / (fl - 1))) ** 0.85
    out = np.empty((n, 80), np.float32)
    for s in range(0, n, 20000):
        idx = np.arange(s, min(n, s + 20000))[:, None] * fs + np.arange(fl)[None, :]
        fr = x[idx]
        fr = fr - fr.mean(1, keepdims=True)
        fr = np.concatenate([fr[:, :1] * 0.03, fr[:, 1:] - 0.97 * fr[:, :-1]], 1) * win
        p = np.abs(np.fft.rfft(fr, 512)) ** 2
        out[s:s + len(fr)] = np.log(np.maximum(p @ B, 1.1920929e-07))
    return out


def embed_track(audio, progress=None):
    """Return (times, embeddings[N,512] L2-normalised, energy[N]) every HOP seconds."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path(progress), providers=["CPUExecutionProvider"])
    F = fbank(audio)
    w, h = int(WIN * 100), int(HOP * 100)
    starts = list(range(0, max(1, len(F) - w + 1), h))
    embs = np.zeros((len(starts), 512), np.float32)
    rms = np.array([float(np.sqrt(np.mean(audio[int(s * 160):int(s * 160) + int(WIN * SR)] ** 2) + 1e-12)) for s in starts])
    bs = 64
    for b in range(0, len(starts), bs):
        chunk = np.stack([F[s:s + w] for s in starts[b:b + bs]])
        chunk = chunk - chunk.mean(1, keepdims=True)  # CMN
        e = sess.run(None, {"feats": chunk})[0]
        embs[b:b + len(e)] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        if progress: progress(b / len(starts), f"Speaker analysis {b * HOP / 60:.0f}/{len(starts) * HOP / 60:.0f} min")
    times = np.array(starts) * 0.01 + WIN / 2
    return times, embs, rms


def cluster(embs, th=0.55):
    """Greedy agglomerative clustering (average linkage via centroids)."""
    cents, members = [], []
    for i, e in enumerate(embs):
        if cents:
            sims = np.array(cents) @ e
            j = int(sims.argmax())
            if sims[j] >= th:
                members[j].append(i)
                c = embs[members[j]].mean(0); cents[j] = c / np.linalg.norm(c); continue
        cents.append(e.copy()); members.append([i])
    # merge similar clusters
    merged = True
    while merged:
        merged = False
        for a in range(len(cents)):
            for b in range(a + 1, len(cents)):
                if float(cents[a] @ cents[b]) >= th:
                    members[a] += members.pop(b); cents.pop(b)
                    c = embs[members[a]].mean(0); cents[a] = c / np.linalg.norm(c)
                    merged = True; break
            if merged: break
    return cents, members


def seg_embeddings(segments, times, embs):
    out = []
    for s in segments:
        m = (times >= s["start"]) & (times <= s["end"])
        if not m.any():
            m = np.abs(times - (s["start"] + s["end"]) / 2) <= HOP
        e = embs[m].mean(0) if m.any() else np.zeros(512, np.float32)
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.array(out)


def host_focus(transcript, audio, cfg, progress=None, log=print):
    """Label every transcript segment HOST / GUEST / OTHER and return keep decisions."""
    segs = [dict(s) for s in transcript["segments"]]
    if len(segs) < 3:
        log("Host focus: too little speech to identify a host; skipping")
        return None
    times, embs, rms = embed_track(audio, progress)
    E = seg_embeddings(segs, times, embs)
    # --- enroll: speech segments in the first N minutes (>= 2 s), cluster, take dominant
    N = cfg.get("host_minutes", 5.0) * 60
    idx = [i for i, s in enumerate(segs) if s["start"] < N and s["end"] - s["start"] >= 2.0 and len(s["text"].split()) >= 4]
    if not idx:
        idx = [i for i, s in enumerate(segs) if s["start"] < max(N, 600)][:50]
    cents, mem = cluster(E[idx], cfg.get("cluster_th", 0.5))
    dur = [sum(segs[idx[k]]["end"] - segs[idx[k]]["start"] for k in m) for m in mem]
    order = np.argsort(dur)[::-1]
    total = sum(dur)
    hosts = [cents[order[0]]]
    for o in order[1:3]:  # co-host if it holds a big share of the opening
        if dur[o] >= 0.3 * total:
            hosts.append(cents[o])
    log(f"Enrolled {len(hosts)} host voice(s) from first {N/60:.0f} min "
        f"({dur[order[0]]:.0f}s of {total:.0f}s speech is the main host)")
    H = np.array(hosts)
    hth = cfg.get("host_th", 0.5)
    hsim = (E @ H.T).max(1)
    is_host = hsim >= hth
    # --- other speakers
    oth = [i for i in range(len(segs)) if not is_host[i]]
    spk = {}
    if oth:
        c2, m2 = cluster(E[oth], cfg.get("cluster_th", 0.5))
        for k, m in enumerate(m2):
            for j in m:
                spk[oth[j]] = k
    # --- turn-taking: a non-host voice is a conversation partner only if it trades turns
    # with the host repeatedly around that passage (an interview/call). A single hand-off
    # in and out (news brief, recorded feature, ad) is two exchanges and gets cut.
    gap = cfg.get("turn_gap", 25.0)
    exch = []  # (time, speaker)
    for i in range(1, len(segs)):
        a, b = i - 1, i
        if segs[b]["start"] - segs[a]["end"] <= gap and is_host[a] != is_host[b]:
            exch.append((segs[b]["start"], spk.get(b if is_host[a] else a)))
    W = cfg.get("guest_window", 240.0)
    min_ex = cfg.get("min_exchanges", 4)
    run_bounds = {}
    i = 0
    while i < len(segs):
        if is_host[i]:
            i += 1; continue
        j = i
        while j + 1 < len(segs) and not is_host[j + 1]:
            j += 1
        for k in range(i, j + 1):
            run_bounds[k] = (segs[i]["start"], segs[j]["end"])
        i = j + 1
    for i, s in enumerate(segs):
        s["hostsim"] = round(float(hsim[i]), 3)
        if is_host[i]:
            s["role"], s["keep"] = "HOST", True
            continue
        k = spk.get(i); rs, re_ = run_bounds[i]
        n = sum(1 for t, kk in exch if kk == k and rs - W <= t <= re_ + W)
        s["speaker"], s["exchanges"] = f"S{k}", n
        s["role"] = "GUEST" if n >= min_ex else "OTHER"
        s["keep"] = n >= min_ex
    # second pass: a voice that is a confirmed conversation partner within +-10 min keeps
    # its other answers too (e.g. the first long answer of an interview)
    gt = {}
    for s in segs:
        if s["role"] == "GUEST":
            gt.setdefault(s["speaker"], []).append(s["start"])
    for s in segs:
        if s["role"] == "OTHER" and any(abs(t - s["start"]) <= 600 for t in gt.get(s["speaker"], [])):
            s["role"], s["keep"] = "GUEST", True
    # host sign-off words often blend with a jingle: keep a tiny tail glued to host speech
    for i in range(1, len(segs)):
        s, p = segs[i], segs[i - 1]
        if not s["keep"] and p["role"] == "HOST" and s["end"] - s["start"] < 2.0 and s["start"] - p["end"] < 0.5 and len(s["text"].split()) <= 3:
            s["keep"] = True; s["role"] = "HOST?"
    # smooth: flip short isolated runs (avoid choppy cuts / host mid-sentence slips)
    def runs():
        r = []
        for i, s in enumerate(segs):
            if r and r[-1][2] == s["keep"]:
                r[-1][1] = i
            else:
                r.append([i, i, s["keep"]])
        return r
    for _ in range(3):
        for a, b, k in runs():
            d = segs[b]["end"] - segs[a]["start"]
            if (not k and d < cfg.get("min_cut", 6.0)) or (k and d < cfg.get("min_keep_run", 8.0) and not segs[a]["role"].startswith("HOST")):
                for i in range(a, b + 1):
                    segs[i]["keep"] = not k; segs[i]["smoothed"] = True
    return segs
