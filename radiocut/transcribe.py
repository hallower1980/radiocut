"""Local speech-to-text with faster-whisper (CTranslate2). Fully offline after the
model is downloaded once (or placed in the models folder)."""
import json, os
from .util import app_dir, load_audio

MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "small", "medium"]


def transcribe(path, model_name="base.en", cache=True, progress=None, threads=0, audio=None):
    cache_file = path + f".{model_name}.transcript.json"
    if cache and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="auto", compute_type="int8",
                         download_root=os.path.join(app_dir(), "models"),
                         cpu_threads=threads)
    if audio is None:
        audio = load_audio(path)
    segs, info = model.transcribe(audio, vad_filter=True, beam_size=1,
                                  condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        if progress and info.duration:
            progress(min(1.0, s.end / info.duration), f"Transcribing {s.end/60:.1f}/{info.duration/60:.1f} min")
    data = {"duration": info.duration, "language": info.language, "segments": out}
    if cache:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data
