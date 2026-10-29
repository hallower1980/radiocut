import argparse, json, os
from .pipeline import analyze, save_project, DEFAULTS
from .export import render, write_cutlists
from .util import fmt_ts


def main(argv=None):
    import sys
    if (argv if argv is not None else sys.argv[1:]) == ["--selftest-speaker"]:
        import numpy as np
        from .speakers import embed_track
        t, e, _ = embed_track(np.random.default_rng(0).normal(0, 0.1, 16000 * 4).astype("float32"))
        print("speaker embeddings OK", e.shape); return 0
    ap = argparse.ArgumentParser(prog="radiocut", description="Strip ads/promos from talk-radio recordings (offline).")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output mp4 (default: <input>.radiocut.mp4)")
    ap.add_argument("--whisper", default=DEFAULTS["whisper_model"])
    ap.add_argument("--llm", choices=["none", "lmstudio", "ollama", "custom"], default="none")
    ap.add_argument("--llm-url"); ap.add_argument("--llm-model", default="")
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    ap.add_argument("--still", help="image to show when the input is audio-only")
    ap.add_argument("--split", action="store_true", help="also export each kept segment as its own clip")
    ap.add_argument("--no-host-focus", action="store_true",
                    help="v0.1.1 behaviour: only cut ads/breaks, keep every voice")
    ap.add_argument("--host-minutes", type=float, default=DEFAULTS["host_minutes"],
                    help="enroll the host voice(s) from the first N minutes (default 5)")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    base = {"lmstudio": "http://localhost:1234/v1", "ollama": "http://localhost:11434/v1"}.get(a.llm, a.llm_url)
    cfg = {"whisper_model": a.whisper, "threshold": a.threshold, "use_llm": a.llm != "none",
           "llm_base": base or DEFAULTS["llm_base"], "llm_model": a.llm_model,
           "host_focus": not a.no_host_focus, "host_minutes": a.host_minutes}
    res = analyze(a.input, cfg, progress=lambda p, m: print(f"[{p:5.1%}] {m}", end="\r", flush=True))
    print()
    out = a.output or os.path.splitext(a.input)[0] + ".radiocut.mp4"
    stem = os.path.splitext(out)[0]
    save_project(res, stem + ".project.json")
    rows, slots, t = write_cutlists(res, stem)
    for r in rows:
        if r["action"] == "REMOVE":
            print(f"REMOVE {r['src_start_tc']}-{r['src_end_tc']} {r['label']}: {r['reason'][:100]}")
    print(f"Kept {fmt_ts(t)} of {fmt_ts(res['duration'])}")
    if not a.analyze_only:
        render(a.input, res["segments"], out, still=a.still)
        if a.split:
            for i, s in enumerate([s for s in res["segments"] if s["keep"]], 1):
                render(a.input, [s], f"{stem}.part{i:02d}.mp4", still=a.still)
        print("Wrote", out)


if __name__ == "__main__":
    main()
