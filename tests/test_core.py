from radiocut import heuristics as H
from radiocut.pipeline import to_segments, DEFAULTS
from radiocut.export import write_cutlists


def fake():
    segs = []
    for i in range(40):  # 20 min of talk with an ad block at 8-10 min
        t = i * 30.0
        if 480 <= t < 600:
            txt = "This show is brought to you by Acme. Call 1-800-555-1234 or visit acme.com, use promo code RADIO."
        else:
            import random
            rnd = random.Random(i)
            vocab = "faith work rest family prayer joy morning question listener story church parish heart hope".split()
            txt = " ".join(rnd.choice(vocab) + str(rnd.randint(0, 999)) for _ in range(40))
        segs.append({"start": t + 1, "end": t + 28, "text": txt})
    return {"duration": 1200.0, "segments": segs}


def test_ad_block_removed(tmp_path):
    tr = fake()
    w = H.make_windows(tr)
    sc = H.heuristic_scores(w, H.transcript_repeats(w, 60), [0.0] * len(w))
    for x, (p, r) in zip(w, sc):
        x.update(p=p, label="CONTENT", reason="; ".join(r))
    segs = to_segments(w, tr, DEFAULTS)
    rm = [s for s in segs if not s["keep"]]
    assert len(rm) == 1 and 470 <= rm[0]["start"] <= 490 and 590 <= rm[0]["end"] <= 610
    rows, slots, t = write_cutlists({"source": "x.mp4", "duration": 1200.0, "segments": segs}, str(tmp_path / "o"))
    assert len(slots) == 1 and abs(t - 1080) < 25
    assert (tmp_path / "o.youtube-chapters.txt").read_text().startswith("0:00")
