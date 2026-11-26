import json
import csv


results={}


files=[
"evidence/final/puf-benchmark.json",
"evidence/final/storage-benchmark.json",
"evidence/final/end-to-end-benchmark.json"
]


for f in files:

    with open(f) as x:
        results.update(json.load(x))


with open(
"evidence/final/LSB_metrics.csv",
"w",
newline=""
) as f:

    writer=csv.writer(f)

    writer.writerow([
        "Metric",
        "Value"
    ])

    for k,v in results.items():
        writer.writerow([
            k,
            v
        ])

print(
"Generated evidence/final/LSB_metrics.csv"
)
