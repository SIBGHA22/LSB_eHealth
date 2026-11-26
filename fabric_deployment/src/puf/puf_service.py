from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from puf_core import SoftwarePUF


class PUFService:
    def __init__(
        self,
        master_secret: bytes,
        enrollment_file: str | None = None,
    ):
        self.puf = SoftwarePUF(master_secret)
        self.enrollment_file = (
            Path(enrollment_file)
            if enrollment_file
            else None
        )

        self.lock = threading.RLock()

        self.enrollments: dict[str, dict[str, str]] = {}

        self.metrics = {
            "enrollment_requests": 0,
            "authentication_requests": 0,
            "authentication_successes": 0,
            "authentication_failures": 0,
            "derivation_requests": 0,
            "last_puf_read_ms": 0.0,
            "last_hash_ms": 0.0,
            "last_hkdf_ms": 0.0,
            "last_total_derivation_ms": 0.0,
        }

        self._load_enrollments()

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(
            value.strip().encode("utf-8")
        ).hexdigest()

    def _enrollment_key(
        self,
        device_id: str,
        challenge: str,
    ) -> str:
        return (
            self._hash_text(device_id)
            + ":"
            + self._hash_text(challenge)
        )

    def _load_enrollments(self) -> None:
        if (
            self.enrollment_file is None
            or not self.enrollment_file.exists()
        ):
            return

        data = json.loads(
            self.enrollment_file.read_text()
        )

        if not isinstance(data, dict):
            raise ValueError(
                "invalid enrollment database"
            )

        self.enrollments = data

    def _save_enrollments(self) -> None:
        if self.enrollment_file is None:
            return

        self.enrollment_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = self.enrollment_file.with_suffix(
            ".tmp"
        )

        tmp.write_text(
            json.dumps(
                self.enrollments,
                indent=2,
                sort_keys=True,
            )
        )

        tmp.replace(self.enrollment_file)

    def enroll(
        self,
        device_id: str,
        challenge: str,
    ) -> dict[str, Any]:

        response_hash = self.puf.response_hash(
            device_id,
            challenge,
        )

        device_id_hash = self._hash_text(
            device_id
        )

        challenge_hash = self._hash_text(
            challenge
        )

        key = self._enrollment_key(
            device_id,
            challenge,
        )

        record = {
            "device_id_hash": device_id_hash,
            "challenge_hash": challenge_hash,
            "response_hash": response_hash,
        }

        with self.lock:
            existing = self.enrollments.get(key)

            if (
                existing is not None
                and existing != record
            ):
                raise ValueError(
                    "existing enrollment does not match current PUF response"
                )

            self.enrollments[key] = record
            self.metrics[
                "enrollment_requests"
            ] += 1

            self._save_enrollments()

        return {
            "status": "enrolled",
            **record,
        }

    def authenticate(
        self,
        device_id: str,
        challenge: str,
    ) -> dict[str, Any]:

        key = self._enrollment_key(
            device_id,
            challenge,
        )

        with self.lock:
            self.metrics[
                "authentication_requests"
            ] += 1

            record = self.enrollments.get(key)

        if record is None:
            with self.lock:
                self.metrics[
                    "authentication_failures"
                ] += 1

            return {
                "authenticated": False,
                "reason": "device/challenge not enrolled",
            }

        success = self.puf.authenticate(
            device_id,
            challenge,
            record["response_hash"],
        )

        with self.lock:
            if success:
                self.metrics[
                    "authentication_successes"
                ] += 1
            else:
                self.metrics[
                    "authentication_failures"
                ] += 1

        return {
            "authenticated": success,
            "device_id_hash": record[
                "device_id_hash"
            ],
        }

    def derive_metadata(
        self,
        device_id: str,
        challenge: str,
        salt: bytes,
    ) -> dict[str, Any]:

        auth = self.authenticate(
            device_id,
            challenge,
        )

        if not auth["authenticated"]:
            raise PermissionError(
                "PUF authentication failed"
            )

        result = self.puf.derive_key(
            device_id,
            challenge,
            salt,
        )

        public = result.public_dict()

        with self.lock:
            self.metrics[
                "derivation_requests"
            ] += 1

            self.metrics[
                "last_puf_read_ms"
            ] = result.puf_read_ms

            self.metrics[
                "last_hash_ms"
            ] = result.hash_ms

            self.metrics[
                "last_hkdf_ms"
            ] = result.hkdf_ms

            self.metrics[
                "last_total_derivation_ms"
            ] = result.total_ms

        return public

    def get_metrics(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self.metrics,
                "enrolled_device_challenges":
                    len(self.enrollments),
            }


def require_string(
    body: dict[str, Any],
    name: str,
) -> str:

    value = body.get(name)

    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{name} cannot be empty"
        )

    return value


def make_handler(service: PUFService):
    class Handler(BaseHTTPRequestHandler):

        server_version = "LSB-PUF/1.0"

        def log_message(
            self,
            format: str,
            *args: Any,
        ) -> None:
            print(
                "%s - %s"
                % (
                    self.address_string(),
                    format % args,
                )
            )

        def send_json(
            self,
            status: int,
            value: Any,
        ) -> None:
            encoded = json.dumps(
                value
            ).encode("utf-8")

            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(encoded)),
            )
            self.end_headers()

            self.wfile.write(encoded)

        def read_json(self) -> dict[str, Any]:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if length <= 0:
                raise ValueError(
                    "JSON body is required"
                )

            body = json.loads(
                self.rfile.read(
                    length
                )
            )

            if not isinstance(body, dict):
                raise ValueError(
                    "JSON body must be an object"
                )

            return body

        def do_GET(self):
            if self.path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service":
                            "lsb-software-puf",
                        "physical_puf_claimed":
                            False,
                    },
                )
                return

            if self.path == "/metrics":
                self.send_json(
                    HTTPStatus.OK,
                    service.get_metrics(),
                )
                return

            self.send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not found"},
            )

        def do_POST(self):
            try:
                body = self.read_json()

                if self.path == "/v1/enroll":
                    result = service.enroll(
                        require_string(
                            body,
                            "device_id",
                        ),
                        require_string(
                            body,
                            "challenge",
                        ),
                    )

                    self.send_json(
                        HTTPStatus.OK,
                        result,
                    )
                    return

                if self.path == "/v1/authenticate":
                    result = service.authenticate(
                        require_string(
                            body,
                            "device_id",
                        ),
                        require_string(
                            body,
                            "challenge",
                        ),
                    )

                    status = (
                        HTTPStatus.OK
                        if result[
                            "authenticated"
                        ]
                        else HTTPStatus.UNAUTHORIZED
                    )

                    self.send_json(
                        status,
                        result,
                    )
                    return

                if self.path == "/v1/derive-metadata":
                    device_id = require_string(
                        body,
                        "device_id",
                    )

                    challenge = require_string(
                        body,
                        "challenge",
                    )

                    salt_hex = require_string(
                        body,
                        "salt_hex",
                    )

                    try:
                        salt = bytes.fromhex(
                            salt_hex
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "salt_hex must be valid hexadecimal"
                        ) from exc

                    result = (
                        service.derive_metadata(
                            device_id,
                            challenge,
                            salt,
                        )
                    )

                    self.send_json(
                        HTTPStatus.OK,
                        result,
                    )
                    return

                self.send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "not found"},
                )

            except PermissionError as exc:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": str(exc)},
                )

            except (
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )

    return Handler


def load_master_secret() -> bytes:
    value = os.environ.get(
        "LSB_PUF_MASTER_SECRET_HEX",
        "",
    ).strip()

    if len(value) != 64:
        raise RuntimeError(
            "LSB_PUF_MASTER_SECRET_HEX must contain exactly 64 hex characters"
        )

    try:
        secret = bytes.fromhex(value)
    except ValueError as exc:
        raise RuntimeError(
            "invalid PUF master secret"
        ) from exc

    return secret


def main():
    address = os.environ.get(
        "LSB_PUF_HOST",
        "127.0.0.1",
    )

    port = int(
        os.environ.get(
            "LSB_PUF_PORT",
            "18083",
        )
    )

    enrollment_file = os.environ.get(
        "LSB_PUF_ENROLLMENT_FILE",
        "",
    ).strip() or None

    service = PUFService(
        master_secret=load_master_secret(),
        enrollment_file=enrollment_file,
    )

    server = ThreadingHTTPServer(
        (address, port),
        make_handler(service),
    )

    print(
        f"LSB Software-PUF runtime listening on "
        f"{address}:{port}",
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
