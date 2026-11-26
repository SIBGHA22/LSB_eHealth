import hashlib
import json
import statistics

from puf_core import SoftwarePUF


MASTER_SECRET = hashlib.sha256(
    b"LSB-EHEALTH-RUNTIME-PUF-EMULATION-V1"
).digest()

CHALLENGE = "lsb-runtime-challenge-001"

SALT = hashlib.sha256(
    b"LSB-EHEALTH-RUNTIME-HKDF-SALT-V1"
).digest()


def hamming_distance(
    left_hex: str,
    right_hex: str,
) -> int:
    left = bytes.fromhex(left_hex)
    right = bytes.fromhex(right_hex)

    return sum(
        (a ^ b).bit_count()
        for a, b in zip(left, right)
    )


def main():
    puf = SoftwarePUF(
        MASTER_SECRET
    )

    total_devices = 500
    repeated_reads = 20

    response_hashes = set()
    key_hashes = set()

    puf_latencies = []
    hash_latencies = []
    hkdf_latencies = []
    total_latencies = []

    enrolled = {}

    for i in range(total_devices):
        device_id = f"lsb-device-{i:04d}"

        result = puf.derive_key(
            device_id,
            CHALLENGE,
            SALT,
        )

        response_hashes.add(
            result.response_hash
        )

        key_hashes.add(
            result.key_hash
        )

        enrolled[device_id] = (
            result.response_hash
        )

        puf_latencies.append(
            result.puf_read_ms
        )

        hash_latencies.append(
            result.hash_ms
        )

        hkdf_latencies.append(
            result.hkdf_ms
        )

        total_latencies.append(
            result.total_ms
        )

    auth_attempts = 0
    auth_success = 0

    for device_id, expected in enrolled.items():
        for _ in range(repeated_reads):
            auth_attempts += 1

            if puf.authenticate(
                device_id,
                CHALLENGE,
                expected,
            ):
                auth_success += 1

    sample_responses = list(
        response_hashes
    )[:100]

    distances = []

    for i in range(
        len(sample_responses) - 1
    ):
        distance = hamming_distance(
            sample_responses[i],
            sample_responses[i + 1],
        )

        distances.append(
            distance / 256.0
        )

    metrics = {
        "implementation": (
            "software_puf_emulation"
        ),
        "physical_puf_claimed": False,
        "devices": total_devices,
        "repeated_reads_per_device": repeated_reads,
        "authentication_attempts": auth_attempts,
        "authentication_successes": auth_success,
        "authentication_success_rate": (
            auth_success / auth_attempts
        ),
        "response_collisions": (
            total_devices -
            len(response_hashes)
        ),
        "derived_key_collisions": (
            total_devices -
            len(key_hashes)
        ),
        "mean_inter_device_hamming_fraction": (
            statistics.mean(distances)
        ),
        "mean_puf_read_ms": statistics.mean(
            puf_latencies
        ),
        "mean_hash_ms": statistics.mean(
            hash_latencies
        ),
        "mean_hkdf_ms": statistics.mean(
            hkdf_latencies
        ),
        "mean_total_key_derivation_ms": (
            statistics.mean(total_latencies)
        ),
        "max_total_key_derivation_ms": max(
            total_latencies
        ),
    }

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
