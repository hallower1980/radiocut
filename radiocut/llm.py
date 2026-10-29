"""Optional local-LLM classification through any OpenAI-compatible local server:
LM Studio (http://localhost:1234/v1) or Ollama (http://localhost:11434/v1)."""
import json, re, urllib.request

PRESETS = {
    "LM Studio": "http://localhost:1234/v1",
    "Ollama": "http://localhost:11434/v1",
}

SYSTEM = (
    "You label 30-second chunks of a live talk-radio broadcast transcript. "
    "Labels: CONTENT = the show's own episode conversation (host talk, guests, callers, "
    "interviews, the show's topic, prayers/readings that are part of the program); "
    "AD = paid commercial/sponsor read/fundraising pitch; PROMO = network or station "
    "self-promotion, promos for other shows/events/apps, station ID, bumper, "
    "'stay with us/coming up' teases, network news briefs/newscasts and traffic/weather inserts that are not the show's own discussion; "
    "MUSIC = music/jingle/silence with no talk. Host-read sponsor messages are AD. "
    "Answer with JSON only: {\"label\": \"CONTENT|AD|PROMO|MUSIC\", \"confidence\": 0-1, \"reason\": \"<8 words\"}"
)


def list_models(base):
    with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=5) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def classify(base, model, prev_text, text, next_text, timeout=180):
    user = (f"[context only - do NOT label] ...{prev_text[-200:]}\n\n"
            f"=== CHUNK TO LABEL ===\n{text or '[no speech]'}\n=== END CHUNK ===\n\n"
            f"[context only - do NOT label] {next_text[:200]}...\n\n"
            "Label ONLY the text between the === markers.")
    body = {"model": model, "temperature": 0, "max_tokens": 60,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*?\}", txt, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    label = str(d.get("label", "")).upper()
    if label not in ("CONTENT", "AD", "PROMO", "MUSIC"):
        label = next((l for l in ("AD", "PROMO", "MUSIC", "CONTENT") if l in txt.upper()), "CONTENT")
    try:
        conf = float(d.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    return label, max(0.0, min(1.0, conf)), str(d.get("reason", ""))[:80]
