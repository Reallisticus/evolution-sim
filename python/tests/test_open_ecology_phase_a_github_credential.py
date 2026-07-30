from __future__ import annotations

import json
from pathlib import Path
import pickle
import unittest
from unittest import mock

from evolution_sim.mind import open_ecology_phase_a_qualification as qualification
from evolution_sim.mind import open_ecology_phase_a_readiness as readiness


class _Response:
    status = 200
    reason = "OK"

    def read(self, limit: int) -> bytes:
        del limit
        return b'{"check_runs":[]}'


class _Connection:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.headers: dict[str, str] | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
    ) -> None:
        self.method = method
        self.path = path
        self.headers = headers

    def getresponse(self) -> _Response:
        return _Response()

    def close(self) -> None:
        return


class OpenEcologyPhaseAGithubCredentialTests(unittest.TestCase):
    def test_one_shot_https_credential_is_not_serializable_or_retained(self) -> None:
        token = bytearray(b"secret-github-token-value")
        connection = _Connection()
        with (
            mock.patch.object(
                qualification.http.client,
                "HTTPSConnection",
                return_value=connection,
            ),
            qualification.ephemeral_github_credential(token) as credential,
        ):
            with self.assertRaisesRegex(TypeError, "cannot be serialized"):
                pickle.dumps(credential)
            receipt = qualification.collect_github_check_runs(
                Path.cwd(),
                source_commit="a" * 40,
            )
        self.assertEqual(token, bytearray(len(token)))
        self.assertEqual(
            receipt["schema_version"],
            qualification.AUTHENTICATED_HTTPS_RECEIPT_SCHEMA_VERSION,
        )
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("secret-github-token-value", serialized)
        self.assertFalse(receipt["request"]["credential_in_environment"])
        self.assertFalse(receipt["request"]["credential_in_argv"])
        self.assertEqual(connection.method, "GET")
        self.assertTrue(connection.headers["Authorization"].startswith("Bearer "))
        validated = readiness._validate_authenticated_github_https_receipt(
            receipt,
            field="test HTTPS receipt",
        )
        self.assertEqual(validated, receipt)

    def test_missing_or_invalid_credential_fails_closed_without_gh(self) -> None:
        with (
            mock.patch.object(qualification.shutil, "which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            qualification.collect_github_check_runs(
                Path.cwd(),
                source_commit="a" * 40,
            )
        for token in (bytearray(), bytearray(b"short"), bytearray(b"x\n" * 12)):
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                with qualification.ephemeral_github_credential(token):
                    pass

    def test_https_receipt_rejects_credential_leakage(self) -> None:
        token = bytearray(b"another-secret-token-value")
        connection = _Connection()
        with (
            mock.patch.object(
                qualification.http.client,
                "HTTPSConnection",
                return_value=connection,
            ),
            qualification.ephemeral_github_credential(token),
        ):
            receipt = qualification.collect_github_check_runs(
                Path.cwd(),
                source_commit="a" * 40,
            )
        receipt["response"]["body"] = "github_pat_hostile"
        receipt["response"]["body_sha256"] = (
            __import__("hashlib").sha256(b"github_pat_hostile").hexdigest()
        )
        with self.assertRaisesRegex(Exception, "credential-redaction"):
            readiness._validate_authenticated_github_https_receipt(
                receipt,
                field="hostile HTTPS receipt",
            )

    def test_generic_token_echo_is_rejected_before_receipt_construction(
        self,
    ) -> None:
        for location in ("body", "reason"):
            with self.subTest(location=location):
                raw_token = b"generic-token-without-a-known-prefix"
                token = bytearray(raw_token)

                class EchoResponse:
                    status = 200
                    reason = raw_token.decode("ascii") if location == "reason" else "OK"

                    def read(self, limit: int) -> bytes:
                        del limit
                        return raw_token if location == "body" else b"{}"

                connection = _Connection()
                with (
                    mock.patch.object(
                        connection,
                        "getresponse",
                        return_value=EchoResponse(),
                    ),
                    mock.patch.object(
                        qualification.http.client,
                        "HTTPSConnection",
                        return_value=connection,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "credential-redaction",
                    ),
                ):
                    with qualification.ephemeral_github_credential(token):
                        qualification.collect_github_check_runs(
                            Path.cwd(),
                            source_commit="a" * 40,
                        )
                self.assertEqual(token, bytearray(len(raw_token)))


if __name__ == "__main__":
    unittest.main()
