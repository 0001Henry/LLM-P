"""Download the official pretrained Chinese BERT, never random initialization."""
import argparse
import urllib.request
import time
from pathlib import Path

REVISION = "main"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    target = Path("models/bert-base-chinese")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "vocab.txt", "tokenizer_config.json", "model.safetensors"):
        dest = target / name
        if dest.exists():
            print(f"Already exists: {dest}", flush=True)
            continue
        url = f"{args.endpoint}/google-bert/bert-base-chinese/resolve/{REVISION}/{name}"
        print(f"Downloading {url}", flush=True)
        temp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as f:
                    while chunk := response.read(1024*1024):
                        f.write(chunk)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        temp.replace(dest)


if __name__ == "__main__":
    main()
