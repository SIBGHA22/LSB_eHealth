import hashlib
import os
import unittest

from puf_core import (
    DERIVED_KEY_LEN,
    SoftwarePUF,
    hkdf_sha256,
)


class TestSoftwarePUF(unittest.TestCase):

    def setUp(self):
        self.master = hashlib.sha256(
            b"lsb-ehealth-test-master-secret"
        ).digest()

        self.puf = SoftwarePUF(
            self.master
        )

        self.device = "device-001"
        self.challenge = "challenge-001"
        self.salt = hashlib.sha256(
            b"lsb-ehealth-test-salt"
        ).digest()

    def test_hkdf_length(self):
        result = hkdf_sha256(
            b"input-key-material",
            b"0123456789abcdef",
            b"test-info",
        )

        self.assertEqual(
            len(result),
            DERIVED_KEY_LEN,
        )

    def test_same_input_is_reliable(self):
        first = self.puf.read(
            self.device,
            self.challenge,
        )

        second = self.puf.read(
            self.device,
            self.challenge,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_different_devices_are_unique(self):
        first = self.puf.read(
            "device-001",
            self.challenge,
        )

        second = self.puf.read(
            "device-002",
            self.challenge,
        )

        self.assertNotEqual(
            first,
            second,
        )

    def test_different_challenges_change_response(self):
        first = self.puf.read(
            self.device,
            "challenge-A",
        )

        second = self.puf.read(
            self.device,
            "challenge-B",
        )

        self.assertNotEqual(
            first,
            second,
        )

    def test_authentication_success(self):
        enrolled = self.puf.response_hash(
            self.device,
            self.challenge,
        )

        self.assertTrue(
            self.puf.authenticate(
                self.device,
                self.challenge,
                enrolled,
            )
        )

    def test_wrong_challenge_rejected(self):
        enrolled = self.puf.response_hash(
            self.device,
            self.challenge,
        )

        self.assertFalse(
            self.puf.authenticate(
                self.device,
                "wrong-challenge",
                enrolled,
            )
        )

    def test_key_derivation_is_deterministic(self):
        first = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        second = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        self.assertEqual(
            first.key,
            second.key,
        )

        self.assertEqual(
            first.key_hash,
            second.key_hash,
        )

    def test_different_salt_changes_key(self):
        first = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        second = self.puf.derive_key(
            self.device,
            self.challenge,
            os.urandom(32),
        )

        self.assertNotEqual(
            first.key,
            second.key,
        )

    def test_derived_key_is_256_bit(self):
        result = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        self.assertEqual(
            len(result.key),
            32,
        )

    def test_public_output_excludes_secret_key(self):
        result = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        public = result.public_dict()

        self.assertNotIn(
            "key",
            public,
        )

        self.assertNotIn(
            "raw_response",
            public,
        )

        self.assertEqual(
            len(public["key_hash"]),
            64,
        )

    def test_latency_is_measured_nonzero(self):
        result = self.puf.derive_key(
            self.device,
            self.challenge,
            self.salt,
        )

        self.assertGreater(
            result.puf_read_ms,
            0.0,
        )

        self.assertGreater(
            result.hash_ms,
            0.0,
        )

        self.assertGreater(
            result.hkdf_ms,
            0.0,
        )

        self.assertGreater(
            result.total_ms,
            0.0,
        )

    def test_no_collisions_for_1000_devices(self):
        responses = set()
        keys = set()

        for i in range(1000):
            device = f"device-{i:04d}"

            result = self.puf.derive_key(
                device,
                self.challenge,
                self.salt,
            )

            self.assertNotIn(
                result.response_hash,
                responses,
            )

            self.assertNotIn(
                result.key_hash,
                keys,
            )

            responses.add(
                result.response_hash
            )

            keys.add(
                result.key_hash
            )

        self.assertEqual(
            len(responses),
            1000,
        )

        self.assertEqual(
            len(keys),
            1000,
        )


if __name__ == "__main__":
    unittest.main()
