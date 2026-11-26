from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any


HKDF_HASH_LEN = 32
DERIVED_KEY_LEN = 32


def hkdf_sha256(
    ikm: bytes,
    salt: bytes,
    info: bytes,
    length: int = DERIVED_KEY_LEN,
) -> bytes:
    if length <= 0:
        raise ValueError("HKDF length must be positive")

    if length > 255 * HKDF_HASH_LEN:
        raise ValueError("HKDF output length too large")

    if not salt:
        salt = b"\x00" * HKDF_HASH_LEN

    prk = hmac.new(
        salt,
        ikm,
        hashlib.sha256,
    ).digest()

    output = bytearray()
    previous = b""
    counter = 1

    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + info + bytes([counter]),
            hashlib.sha256,
        ).digest()

        output.extend(previous)
        counter += 1

    return bytes(output[:length])


@dataclass
class PUFDerivation:
    device_id_hash: str
    response_hash: str
    key_hash: str

    puf_read_ms: float
    hash_ms: float
    hkdf_ms: float
    total_ms: float

    # Internal only. Never place this in API/JSON output.
    key: bytes

    def public_dict(self) -> dict[str, Any]:
        return {
            "device_id_hash": self.device_id_hash,
            "response_hash": self.response_hash,
            "key_hash": self.key_hash,
            "puf_read_ms": self.puf_read_ms,
            "hash_ms": self.hash_ms,
            "hkdf_ms": self.hkdf_ms,
            "total_ms": self.total_ms,
        }


class SoftwarePUF:
    """
    Software-only PUF emulator for the LSB eHealth research prototype.

    This is not claimed to be a physical silicon PUF.

    The master secret emulates device-specific manufacturing entropy.
    """

    DOMAIN_DEVICE = b"LSB-EHEALTH|PUF|DEVICE|"
    DOMAIN_CHALLENGE = b"LSB-EHEALTH|PUF|CHALLENGE|"
    HKDF_INFO = b"LSB-EHEALTH|PUF|DEVICE-KEY|V1"

    def __init__(self, master_secret: bytes):
        if not isinstance(master_secret, bytes):
            raise TypeError("master_secret must be bytes")

        if len(master_secret) < 32:
            raise ValueError(
                "master_secret must contain at least 32 bytes"
            )

        self._master_secret = bytes(master_secret)

    @staticmethod
    def _validate_text(
        value: str,
        name: str,
    ) -> bytes:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")

        value = value.strip()

        if not value:
            raise ValueError(f"{name} cannot be empty")

        return value.encode("utf-8")

    def _device_secret(
        self,
        device_id: str,
    ) -> bytes:
        device_bytes = self._validate_text(
            device_id,
            "device_id",
        )

        return hmac.new(
            self._master_secret,
            self.DOMAIN_DEVICE + device_bytes,
            hashlib.sha256,
        ).digest()

    def read(
        self,
        device_id: str,
        challenge: str,
    ) -> bytes:
        device_secret = self._device_secret(device_id)

        challenge_bytes = self._validate_text(
            challenge,
            "challenge",
        )

        return hmac.new(
            device_secret,
            self.DOMAIN_CHALLENGE + challenge_bytes,
            hashlib.sha256,
        ).digest()

    def response_hash(
        self,
        device_id: str,
        challenge: str,
    ) -> str:
        response = self.read(
            device_id,
            challenge,
        )

        return hashlib.sha256(response).hexdigest()

    def authenticate(
        self,
        device_id: str,
        challenge: str,
        expected_response_hash: str,
    ) -> bool:
        actual = self.response_hash(
            device_id,
            challenge,
        )

        return hmac.compare_digest(
            actual.lower(),
            expected_response_hash.strip().lower(),
        )

    def derive_key(
        self,
        device_id: str,
        challenge: str,
        salt: bytes,
    ) -> PUFDerivation:
        if not isinstance(salt, bytes):
            raise TypeError("salt must be bytes")

        if len(salt) < 16:
            raise ValueError(
                "salt must contain at least 16 bytes"
            )

        total_start = time.perf_counter_ns()

        puf_start = time.perf_counter_ns()

        response = self.read(
            device_id,
            challenge,
        )

        puf_end = time.perf_counter_ns()

        hash_start = time.perf_counter_ns()

        response_digest = hashlib.sha256(
            response
        ).digest()

        response_hash = response_digest.hex()

        hash_end = time.perf_counter_ns()

        hkdf_start = time.perf_counter_ns()

        key = hkdf_sha256(
            ikm=response_digest,
            salt=salt,
            info=self.HKDF_INFO,
            length=DERIVED_KEY_LEN,
        )

        hkdf_end = time.perf_counter_ns()

        device_id_hash = hashlib.sha256(
            device_id.strip().encode("utf-8")
        ).hexdigest()

        key_hash = hashlib.sha256(
            key
        ).hexdigest()

        total_end = time.perf_counter_ns()

        return PUFDerivation(
            device_id_hash=device_id_hash,
            response_hash=response_hash,
            key_hash=key_hash,
            puf_read_ms=(puf_end - puf_start) / 1_000_000,
            hash_ms=(hash_end - hash_start) / 1_000_000,
            hkdf_ms=(hkdf_end - hkdf_start) / 1_000_000,
            total_ms=(total_end - total_start) / 1_000_000,
            key=key,
        )
