"""Render the kept segments with short audio fades and write cut lists / chapters."""
import csv, json, os, subprocess, sys
from .util import ffmpeg_exe, fmt_ts, has_video

FADE = 0.25


def _run(cmd, log=print):
    kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])


def render(src, segs, out, still=None, log=print, ranges=None):
    keep = ranges or [(s["start"], s["end"]) for s in segs if s["keep"]]
    if not keep:
        raise ValueError("Nothing selected to keep")
    video = has_video(src)
    parts, fc = [], []
    for i, (a, b) in enumerate(keep):
        d = b - a
        fd = min(FADE, d / 4)
        fc.append(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d={fd:.3f},"
                  f"afade=t=out:st={d - fd:.3f}:d={fd:.3f}[a{i}]")
        if video:
            fc.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}]")
            parts.append(f"[v{i}][a{i}]")
        else:
            parts.append(f"[a{i}]")
    n = len(keep)
    cmd = [ffmpeg_exe(), "-y", "-nostdin", "-i", src]
    if video:
        fc.append("".join(parts) + f"concat=n={n}:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    else:
        fc.append("".join(parts) + f"concat=n={n}:v=0:a=1[a]")
        if still:
            cmd += ["-loop", "1", "-framerate", "2", "-i", still]
            fc.append("[1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v]")
        else:
            fc.append("[a]asplit[a][aw];[aw]showwaves=s=1280x360:mode=cline:rate=15:colors=0x4FC3F7[w];"
                      "color=c=0x101820:s=1280x720:r=15[bg];[bg][w]overlay=0:180:shortest=1,format=yuv420p[v]")
        maps = ["-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-shortest"]
    script = out + ".filter.txt"
    with open(script, "w") as f:
        f.write(";\n".join(fc))
    cmd += ["-filter_complex_script", script] + maps + ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    log("Rendering " + os.path.basename(out))
    try:
        _run(cmd)
    finally:
        os.remove(script)
    return out


def write_cutlists(res, base, fps=30):
    segs = res["segments"]
    t = 0.0
    rows, chapters, slots = [], [], []
    for s in segs:
        d = s["end"] - s["start"]
        row = {"action": "KEEP" if s["keep"] else "REMOVE", "label": s["label"],
               "src_start": round(s["start"], 2), "src_end": round(s["end"], 2), "duration": round(d, 2),
               "src_start_tc": fmt_ts(s["start"]), "src_end_tc": fmt_ts(s["end"]),
               "out_position": round(t, 2), "out_position_tc": fmt_ts(t), "reason": s.get("reason", ""),
               "preview": s.get("preview", "")}
        rows.append(row)
        if s["keep"]:
            chapters.append((t, s.get("title") or s.get("preview", "")[:60].strip() or "Segment"))
            t += d
        else:
            if not slots or slots[-1]["out_position"] != round(t, 2):
                slots.append({"out_position": round(t, 2), "out_position_tc": fmt_ts(t),
                              "removed_seconds": round(d, 2), "removed_label": s["label"]})
            else:
                slots[-1]["removed_seconds"] += round(d, 2)
    with open(base + ".cutlist.json", "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(res["source"]), "source_duration": res["duration"],
                   "output_duration": round(t, 2), "segments": rows, "ad_slots": slots}, f, indent=1)
    with open(base + ".cutlist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # YouTube chapters: first must be 0:00, each >= 10 s
    with open(base + ".youtube-chapters.txt", "w", encoding="utf-8") as f:
        last = -99
        for i, (ts, title) in enumerate(chapters):
            if ts - last >= 10:
                f.write(f"{fmt_ts(ts)} {title}\n"); last = ts
        f.write("\n# Ad insertion points (output timeline):\n")
        for s in slots:
            f.write(f"# {s['out_position_tc']}  (removed {s['removed_seconds']:.0f}s {s['removed_label']})\n")
    # CMX3600 EDL of kept source ranges (record = output timeline)
    def tc(x):
        fr = int(round(x * fps)); return f"{fr//(3600*fps):02d}:{fr//(60*fps)%60:02d}:{fr//fps%60:02d}:{fr%fps:02d}"
    with open(base + ".edl", "w", encoding="utf-8") as f:
        f.write("TITLE: RadioCut\nFCM: NON-DROP FRAME\n\n")
        rec, n = 0.0, 1
        for s in segs:
            if not s["keep"]:
                continue
            d = s["end"] - s["start"]
            f.write(f"{n:03d}  AX       AA/V  C        {tc(s['start'])} {tc(s['end'])} {tc(rec)} {tc(rec+d)}\n")
            f.write(f"* FROM CLIP NAME: {os.path.basename(res['source'])}\n\n")
            rec += d; n += 1
    return rows, slots, t
