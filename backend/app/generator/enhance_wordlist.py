import json
from wordfreq import zipf_frequency
from . import config

IN_PATH = config.WORDLIST_PATH
OUT_PATH = config.ENRICHED_WORDLIST_PATH


def enhance_wordlist():
    raw = json.loads(IN_PATH.read_text())
    words = raw["words"]

    enhanced = []
    for w in words:
        freq = zipf_frequency(w.lower(), "en")
        if freq <= 0:
            freq = 1.0

        enhanced.append({
            "word": w.lower(),
            "zipf": round(freq, 4)
        })

    out = {
        "name": raw["name"],
        "size": len(enhanced),
        "words": enhanced
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved enriched wordlist → {OUT_PATH}")


if __name__ == "__main__":
    enhance_wordlist()
