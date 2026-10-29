# RadioCut

**Turn live talk-radio recordings into concise, ad-free videos for YouTube — fully offline, with local AI.**

RadioCut listens to a recording (video or audio), finds the commercial breaks, sponsor reads,
network/station promos, station IDs, bumpers and jingles, and exports a clean video of just the
episode's own conversation. Because it is built for studios that sell their own spots, it also
exports a **cut list with ad-insertion points**, an **EDL** for your editor and **YouTube chapters**.

Nothing leaves your computer: speech-to-text runs locally (faster-whisper), and the optional AI
step talks to a model running on your own machine through **LM Studio** or **Ollama**.

## Download & install

Get the latest build from the [Releases page](https://github.com/hallower1980/radiocut/releases/latest):

| System | File |
|---|---|
| Windows 10/11 (64-bit) | `RadioCut-Setup-Windows-x64.exe` (installer) or `RadioCut-Windows-x64-portable.zip` |
| Mac with Apple Silicon (M1–M4) | `RadioCut-macOS-AppleSilicon.dmg` |
| Mac with Intel | `RadioCut-macOS-Intel.dmg` |

ffmpeg is bundled — nothing else to install.

### The "unknown developer" warnings (the app is not code-signed yet)

* **Windows (SmartScreen):** if you see *"Windows protected your PC"*, click **More info → Run anyway**.
  The installer installs for your user only (no admin needed).
* **macOS (Gatekeeper):** open the `.dmg`, drag **RadioCut** to **Applications**. The first time,
  **right-click (or Control-click) RadioCut → Open → Open**. On macOS 15+, if that is not offered:
  try to open it once, then go to **System Settings → Privacy & Security**, scroll down and click
  **Open Anyway**. (Advanced alternative: `xattr -dr com.apple.quarantine /Applications/RadioCut.app`.)

The first time you process a file, the chosen speech model (75 MB for `base.en`, 480 MB for
`small.en`) is downloaded once into `~/.radiocut/models`. After that RadioCut works with no internet.
With **Host focus** on (the default) a 29 MB speaker-recognition model (WeSpeaker CAM++,
run with ONNX Runtime) is also downloaded once into the same folder.
For a fully air-gapped machine, copy that `models` folder from a computer that has run it once.

## Using it

1. Open RadioCut and **drop a recording** on the blue box (or click it to browse).
2. Pick a **Speech model**: `base.en` is fast on any laptop; `small.en` is more accurate.
3. Pick **Local AI** (optional, recommended): *None* (heuristics only), *LM Studio* or *Ollama*.
4. Click **1. Process**. A 2-hour show takes roughly 10–40 minutes on a typical laptop
   (longer with an AI model on a CPU-only machine).
5. Review the list: green rows are kept, red rows will be removed, with the reason.
   **Double-click** any row to flip it.
6. Click **2. Export**. You get:
   * `name.mp4` – the concise video (original video kept; for audio-only input a waveform
     or your own still image is used). Cuts get short fades so they sound clean.
   * `name.cutlist.csv` / `.cutlist.json` – every segment, KEEP/REMOVE, source and output times,
     reason, plus the **ad slots** (where breaks were, on the new timeline).
   * `name.edl` – CMX3600 edit list for Premiere/Resolve/Final Cut (via conversion).
   * `name.youtube-chapters.txt` – paste into the YouTube description; ad insertion points are listed below.
   * Optional: one clip per kept segment (`name.part01.mp4`, …).

Command line (same engine): `RadioCut input.mp4 --whisper small.en --llm lmstudio --llm-model <id> --split`

## Setting up LM Studio (beginner friendly)

1. Install [LM Studio](https://lmstudio.ai) and open it.
2. Search for and download a small instruct model, e.g. **Qwen2.5 3B Instruct** or
   **Llama 3.2 3B Instruct** (7B–8B models are more accurate if your laptop has 16 GB+ RAM).
3. Go to the **Developer** tab, load the model, and switch **Start server** on (port 1234).
4. In RadioCut choose **Local AI → LM Studio**, click **Refresh models**, select the model.

**Ollama** instead: install from [ollama.com](https://ollama.com), run `ollama pull qwen2.5:3b`,
and choose **Local AI → Ollama** (port 11434). Any other OpenAI-compatible local server works via *Custom*.

## How detection works

### Host focus (default, new in v0.1.2)

RadioCut learns the host's voice from the **first 5 minutes** (change it in the GUI or with
`--host-minutes`), then keeps only:

* the **host**, and
* **guests and callers the host talks with directly** – voices that trade turns back and forth with
  the host (several exchanges within ~25 s gaps, inside a ±4-minute neighbourhood).

Everything else is cut: news briefs, other shows' hosts, pre-recorded features ("Saint of the Day",
recorded reflections, segments hosted by someone else), ads, promos, station IDs and music.
A voice that is only handed to and taken back from once (e.g. "over to you, Bailey ... thanks,
Bailey") counts as a separate segment, not a conversation, and is cut.
Cuts fall in the pauses between sentences (Whisper segment boundaries) with short fades,
so the host is not cut mid-sentence. Each cut in the review list says why (host similarity,
voice cluster, number of turn exchanges). Turn it off (`--no-host-focus`) for the v0.1.1 behaviour,
which removes only ads and breaks and keeps every voice.

If a local LLM is enabled in host-focus mode, host-read ads it is confident about (≥0.7) are cut too.

### Ad / break detection (used when host focus is off, and for the host-read-ad check)

The recording is cut into 30-second windows and each gets a "break" score from:

* **Cue phrases** – "brought to you by", promo codes, phone numbers, web addresses, "call now",
  "stay with us", "coming up", "you're listening to…", app/podcast plugs, etc.
* **Repetition** – the same words recurring far apart in the show (ads and promos air more than once).
* **Audio fingerprint** – the same jingle/spot audio replayed elsewhere in the recording.
* **Speech density** – stretches with little speech (music beds, jingles, dead air).
* **Local LLM (optional)** – reads each window with its neighbours and labels it CONTENT / AD / PROMO / MUSIC;
  blended with the heuristics.

Scores are smoothed, very short runs are merged (breaks < 20 s and kept parts < 60 s are absorbed),
and cut points are snapped to pauses in speech. Everything is a suggestion – review before exporting.

## Limitations

* Host focus assumes the host is the main voice in the opening minutes. If the recording starts
  with a long pre-roll, news or another show, raise/lower `--host-minutes` or trim the start first.
* A pre-recorded segment voiced by the host himself is kept (same voice). A co-host is detected
  only if they speak at least ~30% of the opening minutes.
* Telephone-quality callers are recognised by turn-taking only; there is no separate phone-line detector yet.

* 30-second resolution: a cut can include a few seconds of a neighbouring segment.
* Host-read ads blended into conversation, and promos for the show itself, are the hardest cases;
  an AI model helps. Recurring-audio matching only works when a spot airs more than once in the file.
* English cue phrases; other languages work for transcription but need the AI step for good results.
* Phones/tablets are future work (desktop laptops only for now). Apps are unsigned (see above).

## Running from source

```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_radiocut.py            # GUI
python run_radiocut.py file.mp4   # CLI
```

MIT licensed.
