import numpy as np
from radiocut import speakers as SP
from radiocut.pipeline import host_segments, DEFAULTS


def voice(k, rng):
    v = np.zeros(512, np.float32); v[k * 10:(k + 1) * 10] = 1
    v += rng.normal(0, 0.05, 512).astype(np.float32)
    return v / np.linalg.norm(v)


def scenario():
    # (start, end, voice) - 0 host, 1 news reader, 2 interview guest, 3 recorded feature
    plan = [(t, t + 10, 0) for t in range(0, 300, 10)]
    plan += [(300, 310, 0)] + [(t, t + 10, 1) for t in range(310, 420, 10)] + [(420, 430, 0)]
    for t in range(430, 610, 20):
        plan += [(t, t + 10, 0), (t + 10, t + 20, 2)]
    plan += [(610, 620, 0)] + [(t, t + 10, 3) for t in range(620, 720, 10)] + [(720, 800, 0)]
    return plan


def test_host_focus(monkeypatch):
    rng = np.random.default_rng(0)
    plan = scenario()
    times = np.arange(0, 800, 0.5) + 0.25
    who = {}
    for a, b, v in plan:
        for t in times[(times >= a) & (times < b)]:
            who[t] = v
    embs = np.array([voice(who[t], rng) for t in times])
    monkeypatch.setattr(SP, "embed_track", lambda audio, progress=None: (times, embs, None))
    tr = {"duration": 800.0, "segments": [{"start": a + 0.3, "end": b - 0.3, "text": "one two three four five"} for a, b, _ in plan]}
    segs = SP.host_focus(tr, None, {"host_minutes": 5}, log=lambda *a: None)
    keep = {v: [s["keep"] for s, (_, _, vv) in zip(segs, plan) if vv == v] for v in range(4)}
    assert all(keep[0]) and all(keep[2]) and not any(keep[1]) and not any(keep[3])
    out = host_segments(segs, [], tr, DEFAULTS)
    cuts = [(round(s["start"]), round(s["end"])) for s in out if not s["keep"]]
    assert cuts == [(310, 420), (620, 720)]


def test_fbank_shape():
    x = np.random.default_rng(1).normal(0, 0.1, 16000).astype(np.float32)
    f = SP.fbank(x)
    assert f.shape == (98, 80) and np.isfinite(f).all()
