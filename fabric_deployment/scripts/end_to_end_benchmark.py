import time
import json
import base64
import requests
import os


RESULT = {}

PUF = "http://127.0.0.1:18083"
STORAGE = "http://127.0.0.1:18084"
CONSENSUS = "http://127.0.0.1:18082"


# -------------------------
# 1. PUF Enrollment
# -------------------------

device = "patient-device-e2e"
challenge = "challenge-e2e"


start = time.time()

enroll = requests.post(
    PUF + "/v1/enroll",
    json={
        "device_id": device,
        "challenge": challenge
    }
)

enroll_latency = (time.time()-start)*1000


# -------------------------
# 2. PUF Authentication
# -------------------------

start = time.time()

auth = requests.post(
    PUF + "/v1/authenticate",
    json={
        "device_id": device,
        "challenge": challenge
    }
)

puf_latency = (time.time()-start)*1000

RESULT["puf_enroll_success"] = enroll.status_code == 200
RESULT["puf_success"] = (
    auth.status_code == 200 and
    auth.json().get("authenticated", False)
)

RESULT["puf_latency_ms"] = puf_latency


# -------------------------
# 3. Storage
# -------------------------

payload = base64.b64encode(
    b"patient healthcare record integration benchmark"
).decode()


start = time.time()

storage = requests.post(
    STORAGE + "/v1/storage/store",
    json={
        "id":"e2e-test-001",
        "payload":payload,
        "aad":"patient-record"
    }
)

storage_latency = (time.time()-start)*1000

RESULT["storage_success"] = storage.status_code == 200
RESULT["storage_latency_ms"] = storage_latency


# -------------------------
# 4. Consensus
# -------------------------

start = time.time()

consensus = requests.post(
    CONSENSUS + "/v1/consensus",
    json={
        "epoch":1,
        "shard_id":0,
        "seed_hex":"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
        "payload_hash":"1553a6d910a45a1a43cac5f0f9698e5e211da857c08ff0d3804ef55bd1e776e9",
        "previous_block_hash":"0000000000000000000000000000000000000000000000000000000000000000"
    }
)

consensus_latency = (time.time()-start)*1000


print("CONSENSUS STATUS:", consensus.status_code)
print("CONSENSUS BODY:", consensus.text)
RESULT["consensus_success"] = consensus.status_code == 200
RESULT["consensus_latency_ms"] = consensus_latency


RESULT["total_latency_ms"] = (
    RESULT["puf_latency_ms"] +
    RESULT["storage_latency_ms"] +
    RESULT["consensus_latency_ms"]
)


RESULT["overall_success"] = (
    RESULT["puf_success"]
    and RESULT["storage_success"]
    and RESULT["consensus_success"]
)


print(json.dumps(RESULT, indent=2))

import pathlib
out = pathlib.Path("evidence/final/end-to-end-benchmark.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(RESULT, indent=2))
print("written:", out)
