"""Produce an enriched wordlist and persist it to the offline MinIO bucket."""

import json

from wordfreq import zipf_frequency

from . import config
from app.utils.s3_data import read_json, write_json

IN_KEY = config.WORDLIST_KEY
OUT_KEY = config.ENRICHED_WORDLIST_KEY
BUCKET = config.OFFLINE_BUCKET


def enhance_wordlist() -> None:
    raw = read_json(BUCKET, IN_KEY, env="offline")
    words = raw["words"]

    enhanced = []
    for w in words:
        freq = zipf_frequency(w.lower(), "en")
        if freq <= 0:
            freq = 1.0

        enhanced.append({"word": w.lower(), "zipf": round(freq, 4)})

    out = {"name": raw["name"], "size": len(enhanced), "words": enhanced}

    write_json(BUCKET, OUT_KEY, out, env="offline")
    print(f"Saved enriched wordlist → s3://{BUCKET}/{OUT_KEY}")


if __name__ == "__main__":
    enhance_wordlist()
