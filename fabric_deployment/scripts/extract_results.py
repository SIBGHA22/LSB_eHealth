import json
from pathlib import Path


files = {
    "PUF": "evidence/final/puf-benchmark.json",
    "Storage": "evidence/final/storage-benchmark.json",
    "EndToEnd": "evidence/final/end-to-end-benchmark.json"
}


for name, file in files.items():

    print("\n======", name, "======")

    path = Path(file)

    if not path.exists():
        print("Missing:", file)
        continue

    data=json.loads(path.read_text())

    for k,v in data.items():
        print(f"{k}: {v}")

