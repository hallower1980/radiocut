import os, sys


def app_dir():
    base = os.environ.get("RADIOCUT_HOME") or os.path.join(os.path.expanduser("~"), ".radiocut")
    os.makedirs(base, exist_ok=True)
    return base


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def fmt_ts(t, ms=False):
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    if ms:
        base += f".{int((t - int(t)) * 1000):03d}"
    return base


def load_audio(path, sr=16000):
    """Decode any audio/video file to mono float32 numpy using the bundled ffmpeg."""
    import subprocess, numpy as np
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    raw = subprocess.run(cmd, capture_output=True, check=True, **kw).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def has_video(path):
    import subprocess, re
    kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
    r = subprocess.run([ffmpeg_exe(), "-nostdin", "-i", path], capture_output=True, text=True,
                       errors="replace", **kw)
    return bool(re.search(r"Stream #\S+.*Video: (?!mjpeg|png)", r.stderr))
