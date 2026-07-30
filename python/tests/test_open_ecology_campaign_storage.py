from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from collections import namedtuple

from evolution_sim.io import open_ecology_campaign_storage as storage
from evolution_sim.io import open_ecology_archive_authority as archive_authority
from scripts import archive_open_ecology_campaign as archive_tool


SOURCE_SHA = "a" * 40
SOURCE_MANIFEST = "b" * 64
CAMPAIGN_ID = "campaign-1"
BUNDLE_ID = "bundle-1"
DiskUsage = namedtuple("DiskUsage", ("total", "used", "free"))
SAFE_DISK_USAGE = DiskUsage(
    total=1024 * 1024**3,
    used=512 * 1024**3,
    free=512 * 1024**3,
)


class OpenEcologyCampaignStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.disk_usage_patch = patch.object(
            storage,
            "_descriptor_disk_usage",
            return_value=(SAFE_DISK_USAGE.total, SAFE_DISK_USAGE.free),
        )
        self.disk_usage_patch.start()
        self.addCleanup(self.disk_usage_patch.stop)

    def _limits(self, **overrides: int) -> storage.CampaignStorageLimits:
        values = {
            "max_campaign_bytes": 1024 * 1024,
            "min_campaign_free_bytes": (storage.DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES),
            "min_remote_free_bytes": 300 * 1024**3,
            "max_entries": 100,
        }
        values.update(overrides)
        return storage.CampaignStorageLimits(**values)

    def _trees(self, directory: str) -> tuple[Path, Path]:
        base = Path(directory).resolve()
        active = base / "active"
        bundle = base / BUNDLE_ID
        active.mkdir()
        bundle.mkdir()
        (active / "writer.jsonl").write_text('{"tick":1}\n', encoding="utf-8")
        (bundle / "checkpoint.bin").write_bytes(b"checkpoint")
        (bundle / "evidence").mkdir()
        (bundle / "evidence" / "events.jsonl").write_text(
            '{"event":1}\n',
            encoding="utf-8",
        )
        return active, bundle

    def _seal(self, active: Path, bundle: Path) -> storage.ClosedBundleSnapshot:
        return storage.seal_closed_bundle(
            active,
            bundle,
            campaign_id=CAMPAIGN_ID,
            bundle_id=BUNDLE_ID,
            source_git_sha=SOURCE_SHA,
            source_manifest_sha256=SOURCE_MANIFEST,
            limits=self._limits(),
        )

    def test_active_campaign_is_only_scanned_and_quota_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, _ = self._trees(directory)
            original = (active / "writer.jsonl").read_bytes()
            scan = storage.check_campaign_storage(
                active,
                campaign_id=CAMPAIGN_ID,
                source_git_sha=SOURCE_SHA,
                limits=self._limits(),
            )
            self.assertEqual(scan.total_file_bytes, len(original))
            self.assertEqual((active / "writer.jsonl").read_bytes(), original)
            self.assertFalse((active / storage.MARKER_NAME).exists())

            with self.assertRaisesRegex(
                storage.CampaignStorageError, "hard byte quota"
            ):
                storage.check_campaign_storage(
                    active,
                    campaign_id=CAMPAIGN_ID,
                    source_git_sha=SOURCE_SHA,
                    limits=self._limits(max_campaign_bytes=len(original) - 1),
                )

    def test_preregistered_resource_floors_cannot_be_weakened(self) -> None:
        cases = (
            {
                "max_campaign_bytes": (storage.ACTIVE_CAMPAIGN_BUDGET_BYTES + 1),
            },
            {
                "min_campaign_free_bytes": (
                    storage.DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES - 1
                ),
            },
            {"min_remote_free_bytes": storage.DEFAULT_REMOTE_FREE_BYTES - 1},
            {"max_entries": storage.DEFAULT_MAX_ENTRIES + 1},
        )
        for override in cases:
            with self.subTest(override=override):
                values = {
                    "max_campaign_bytes": storage.ACTIVE_CAMPAIGN_BUDGET_BYTES,
                    "min_campaign_free_bytes": (
                        storage.DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES
                    ),
                    "min_remote_free_bytes": storage.DEFAULT_REMOTE_FREE_BYTES,
                    "max_entries": storage.DEFAULT_MAX_ENTRIES,
                }
                values.update(override)
                with self.assertRaises(storage.CampaignStorageError):
                    storage.CampaignStorageLimits(**values).validate()

    def test_filesystem_floor_is_greater_of_100_gib_or_twenty_percent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, _ = self._trees(directory)
            total = 1024 * 1024**3
            with patch.object(
                storage,
                "_descriptor_disk_usage",
                return_value=(total, 199 * 1024**3),
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "free-space floor",
                ):
                    storage.check_campaign_storage(
                        active,
                        campaign_id=CAMPAIGN_ID,
                        source_git_sha=SOURCE_SHA,
                        limits=self._limits(),
                    )

            with patch.object(
                storage,
                "_descriptor_disk_usage",
                return_value=(total, 205 * 1024**3),
            ):
                storage.check_campaign_storage(
                    active,
                    campaign_id=CAMPAIGN_ID,
                    source_git_sha=SOURCE_SHA,
                    limits=self._limits(),
                )

    def test_every_ancestor_symlink_and_raw_resolved_identity_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            real = base / "real"
            active = real / "active"
            active.mkdir(parents=True)
            alias = base / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "not a real directory",
            ):
                storage.check_campaign_storage(
                    alias / "active",
                    campaign_id=CAMPAIGN_ID,
                    source_git_sha=SOURCE_SHA,
                    limits=self._limits(),
                )

    def test_directory_tree_creation_rejects_user_controlled_symlink_ancestor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            outside = base / "outside"
            outside.mkdir()
            hostile = base / "attempts"
            hostile.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "not a real directory",
            ):
                storage.ensure_real_directory_tree(
                    hostile / "evidence",
                    field="test preservation tree",
                )

            self.assertEqual(list(outside.iterdir()), [])
            created = storage.ensure_real_directory_tree(
                base / "real" / "nested",
                field="test preservation tree",
            )
            self.assertEqual(created, base / "real" / "nested")
            self.assertTrue(created.is_dir())

    def test_root_owned_top_alias_never_resolves_a_lower_symlink(self) -> None:
        if not Path("/var/tmp").is_dir():
            self.skipTest("/var/tmp compatibility namespace is unavailable")
        with tempfile.TemporaryDirectory(dir="/var/tmp") as directory:
            alias_base = Path(directory)
            expected_base = alias_base.resolve()
            created = storage.ensure_real_directory_tree(
                alias_base / "real" / "nested",
                field="root compatibility tree",
            )
            self.assertEqual(created, expected_base / "real" / "nested")

            outside = expected_base / "outside"
            outside.mkdir()
            hostile = alias_base / "lower-alias"
            hostile.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "not a real directory",
            ):
                storage.ensure_real_directory_tree(
                    hostile / "must-not-exist",
                    field="root compatibility tree",
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_root_compatibility_target_is_descriptor_walked_not_resolved(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            outside = base / "outside"
            outside.mkdir()
            lower_alias = base / "compatibility-target"
            lower_alias.symlink_to(outside, target_is_directory=True)
            synthetic_top = Path("/synthetic-root-compatibility")
            real_lstat = storage.os.lstat

            def root_owned_synthetic_alias(path: object) -> object:
                if Path(path) == synthetic_top:
                    return SimpleNamespace(
                        st_mode=stat.S_IFLNK | 0o777,
                        st_uid=0,
                    )
                return real_lstat(path)

            with (
                patch.object(
                    storage.os,
                    "lstat",
                    side_effect=root_owned_synthetic_alias,
                ),
                patch.object(
                    storage.os,
                    "readlink",
                    return_value=str(lower_alias),
                ),
                self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "not a real directory",
                ),
            ):
                storage.ensure_real_directory_tree(
                    synthetic_top / "must-not-exist",
                    field="synthetic root compatibility tree",
                )

            self.assertEqual(list(outside.iterdir()), [])

    def test_scan_rejects_symlink_hardlink_and_special_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            for case in ("symlink", "hardlink", "fifo"):
                with self.subTest(case=case):
                    root = base / case
                    root.mkdir()
                    source = root / "source.bin"
                    source.write_bytes(b"payload")
                    if case == "symlink":
                        (root / "alias").symlink_to("source.bin")
                    elif case == "hardlink":
                        os.link(source, root / "alias.bin")
                    else:
                        os.mkfifo(root / "pipe")
                    with self.assertRaises(storage.CampaignStorageError):
                        storage.check_campaign_storage(
                            root,
                            campaign_id=CAMPAIGN_ID,
                            source_git_sha=SOURCE_SHA,
                            limits=self._limits(),
                        )

    def test_scan_rejects_child_device_mount_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, _ = self._trees(directory)
            crossing = active / "mounted"
            crossing.mkdir()
            real_stat_at = storage._stat_at_without_following

            def device_drift(parent_descriptor: int, name: str) -> object:
                metadata = real_stat_at(parent_descriptor, name)
                if name == crossing.name:
                    return SimpleNamespace(
                        st_dev=metadata.st_dev + 1,
                        st_mode=metadata.st_mode,
                    )
                return metadata

            with patch.object(
                storage,
                "_stat_at_without_following",
                side_effect=device_drift,
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "mount boundary",
                ):
                    storage.check_campaign_storage(
                        active,
                        campaign_id=CAMPAIGN_ID,
                        source_git_sha=SOURCE_SHA,
                        limits=self._limits(),
                    )

    def test_scan_fails_closed_when_child_directory_is_swapped_for_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            active = base / "active"
            active.mkdir()
            victim = active / "victim"
            victim.mkdir()
            outside = base / "outside"
            outside.mkdir()
            (outside / "untrusted.bin").write_bytes(b"outside")
            real_open = os.open
            swapped = False

            def swap_before_open(
                candidate: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal swapped
                if (
                    not swapped
                    and candidate == victim.name
                    and dir_fd is not None
                    and flags & getattr(os, "O_DIRECTORY", 0)
                ):
                    os.rmdir(victim)
                    victim.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return real_open(candidate, flags, mode, dir_fd=dir_fd)

            with patch.object(storage.os, "open", side_effect=swap_before_open):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "cannot safely open storage directory",
                ):
                    storage.check_campaign_storage(
                        active,
                        campaign_id=CAMPAIGN_ID,
                        source_git_sha=SOURCE_SHA,
                        limits=self._limits(),
                    )
            self.assertTrue(swapped)

    def test_scan_rejects_same_size_mutation_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active = Path(directory).resolve() / "active"
            active.mkdir()
            target = active / "payload.bin"
            target.write_bytes(b"original")
            original = target.stat()
            real_read = os.read
            mutated = False

            def mutate_before_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                if not mutated:
                    writer = os.open(target, os.O_WRONLY)
                    try:
                        os.write(writer, b"mutated!")
                    finally:
                        os.close(writer)
                    os.utime(
                        target,
                        ns=(original.st_atime_ns, original.st_mtime_ns),
                    )
                    mutated = True
                return real_read(descriptor, size)

            with patch.object(storage.os, "read", side_effect=mutate_before_read):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "changed while hashing",
                ):
                    storage.check_campaign_storage(
                        active,
                        campaign_id=CAMPAIGN_ID,
                        source_git_sha=SOURCE_SHA,
                        limits=self._limits(),
                    )
            self.assertTrue(mutated)

    def test_active_root_or_descendant_cannot_be_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            active = base / "active"
            bundle = active / BUNDLE_ID
            bundle.mkdir(parents=True)
            (bundle / "evidence").write_bytes(b"x")
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "separate tree",
            ):
                self._seal(active, bundle)

    def test_seal_counts_active_and_closed_bundle_against_one_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, bundle = self._trees(directory)
            combined_bytes = sum(
                path.stat().st_size
                for root in (active, bundle)
                for path in root.rglob("*")
                if path.is_file()
            )
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "active root plus closed bundle",
            ):
                storage.seal_closed_bundle(
                    active,
                    bundle,
                    campaign_id=CAMPAIGN_ID,
                    bundle_id=BUNDLE_ID,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                    limits=self._limits(max_campaign_bytes=combined_bytes - 1),
                )
            self.assertFalse((bundle / storage.MARKER_NAME).exists())
            self.assertTrue(stat.S_IMODE(bundle.stat().st_mode) & 0o200)

    def test_seal_creates_immutable_marker_and_exact_producer_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, bundle = self._trees(directory)
            snapshot = self._seal(active, bundle)
            self.assertEqual(snapshot.marker["campaign_id"], CAMPAIGN_ID)
            self.assertEqual(snapshot.marker["bundle_id"], BUNDLE_ID)
            self.assertEqual(snapshot.producer_manifest["symlink_count"], 0)
            self.assertEqual(
                snapshot.producer_manifest["root_name"],
                BUNDLE_ID,
            )
            self.assertFalse(stat.S_IMODE(bundle.stat().st_mode) & 0o222)
            for path in bundle.rglob("*"):
                self.assertFalse(stat.S_IMODE(path.stat().st_mode) & 0o222)
            repeated = storage.validate_closed_bundle(
                active,
                bundle,
                campaign_id=CAMPAIGN_ID,
                bundle_id=BUNDLE_ID,
                source_git_sha=SOURCE_SHA,
                source_manifest_sha256=SOURCE_MANIFEST,
                limits=self._limits(),
            )
            self.assertEqual(repeated.marker_sha256, snapshot.marker_sha256)

    def test_marker_or_source_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, bundle = self._trees(directory)
            self._seal(active, bundle)
            marker = bundle / storage.MARKER_NAME
            os.chmod(bundle, 0o755)
            os.chmod(marker, 0o644)
            decoded = json.loads(marker.read_bytes())
            decoded["file_count"] = int(decoded["file_count"]) + 1
            marker.write_bytes(storage.canonical_json_bytes(decoded))
            os.chmod(marker, 0o444)
            os.chmod(bundle, 0o555)
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "tampered",
            ):
                storage.validate_closed_bundle(
                    active,
                    bundle,
                    campaign_id=CAMPAIGN_ID,
                    bundle_id=BUNDLE_ID,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                    limits=self._limits(),
                )

        with tempfile.TemporaryDirectory() as directory:
            active, bundle = self._trees(directory)
            self._seal(active, bundle)
            evidence = bundle / "checkpoint.bin"
            os.chmod(evidence, 0o644)
            evidence.write_bytes(b"changed-byte")
            os.chmod(evidence, 0o444)
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "tampered",
            ):
                storage.validate_closed_bundle(
                    active,
                    bundle,
                    campaign_id=CAMPAIGN_ID,
                    bundle_id=BUNDLE_ID,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                    limits=self._limits(),
                )

    def test_storage_lock_blocks_concurrent_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            active, _ = self._trees(directory)
            lock_path = storage.default_storage_lock_path(active)
            with storage.CampaignStorageLock(
                lock_path,
                campaign_id=CAMPAIGN_ID,
                source_git_sha=SOURCE_SHA,
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "locked by another process",
                ):
                    with storage.CampaignStorageLock(
                        lock_path,
                        campaign_id=CAMPAIGN_ID,
                        source_git_sha=SOURCE_SHA,
                    ):
                        self.fail("second lock unexpectedly acquired")

    def test_receipt_is_exclusive_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory).resolve() / "receipt.json"
            envelope = storage.write_verified_receipt(
                receipt,
                {"status": "verified", "objects": 3},
            )
            self.assertEqual(
                storage.load_verified_receipt(receipt),
                envelope,
            )
            os.chmod(receipt, 0o644)
            decoded = json.loads(receipt.read_bytes())
            decoded["payload"]["objects"] = 2
            receipt.write_bytes(storage.canonical_json_bytes(decoded))
            with self.assertRaisesRegex(
                storage.CampaignStorageError,
                "digest mismatch",
            ):
                storage.load_verified_receipt(receipt)

    def test_atomic_publication_preserves_a_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            destination = parent / "receipt.json"
            real_rename = storage._rename_name_no_replace

            def publish_competitor_then_rename(
                parent_descriptor: int,
                source_name: str,
                destination_name: str,
            ) -> None:
                competitor = os.open(
                    destination_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o444,
                    dir_fd=parent_descriptor,
                )
                try:
                    storage._write_all(competitor, b"competitor\n")
                    os.fsync(competitor)
                finally:
                    os.close(competitor)
                real_rename(
                    parent_descriptor,
                    source_name,
                    destination_name,
                )

            with patch.object(
                storage,
                "_rename_name_no_replace",
                side_effect=publish_competitor_then_rename,
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "refusing to overwrite",
                ):
                    storage._atomic_create(
                        destination,
                        b"authoritative\n",
                        mode=0o444,
                    )

            self.assertEqual(destination.read_bytes(), b"competitor\n")
            stages = list(parent.glob(f".{destination.name}.pending-*"))
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0].read_bytes(), b"authoritative\n")

    def test_atomic_publication_detects_stage_replacement_without_deleting_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            destination = parent / "receipt.json"
            real_rename = storage._rename_name_no_replace

            def replace_stage_then_publish(
                parent_descriptor: int,
                source_name: str,
                destination_name: str,
            ) -> None:
                os.unlink(source_name, dir_fd=parent_descriptor)
                competitor = os.open(
                    source_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o444,
                    dir_fd=parent_descriptor,
                )
                try:
                    storage._write_all(competitor, b"stage-competitor\n")
                    os.fchmod(competitor, 0o444)
                    os.fsync(competitor)
                finally:
                    os.close(competitor)
                real_rename(
                    parent_descriptor,
                    source_name,
                    destination_name,
                )

            with patch.object(
                storage,
                "_rename_name_no_replace",
                side_effect=replace_stage_then_publish,
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "identity changed",
                ):
                    storage._atomic_create(
                        destination,
                        b"authoritative\n",
                        mode=0o444,
                    )

            self.assertEqual(destination.read_bytes(), b"stage-competitor\n")

    def test_atomic_publication_detects_post_open_destination_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            destination = parent / "receipt.json"
            retained = parent / "retained-authoritative.json"
            real_readback = storage._read_exact_descriptor
            readback_count = 0

            def replace_destination_during_readback(
                descriptor: int,
                *,
                expected_bytes: int,
            ) -> bytes:
                nonlocal readback_count
                readback_count += 1
                observed = real_readback(
                    descriptor,
                    expected_bytes=expected_bytes,
                )
                if readback_count == 2:
                    os.replace(destination, retained)
                    destination.write_bytes(b"post-open-competitor\n")
                    destination.chmod(0o444)
                return observed

            with patch.object(
                storage,
                "_read_exact_descriptor",
                side_effect=replace_destination_during_readback,
            ):
                with self.assertRaisesRegex(
                    storage.CampaignStorageError,
                    "changed during publication|changed after readback",
                ):
                    storage._atomic_create(
                        destination,
                        b"authoritative\n",
                        mode=0o444,
                    )

            self.assertEqual(readback_count, 2)
            self.assertEqual(destination.read_bytes(), b"post-open-competitor\n")
            self.assertEqual(retained.read_bytes(), b"authoritative\n")


class OpenEcologyArchiveToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.disk_usage_patch = patch.object(
            storage,
            "_descriptor_disk_usage",
            return_value=(SAFE_DISK_USAGE.total, SAFE_DISK_USAGE.free),
        )
        self.disk_usage_patch.start()
        self.addCleanup(self.disk_usage_patch.stop)
        self.ssh_endpoint_patch = patch.object(
            archive_tool,
            "_require_effective_ssh_endpoint",
        )
        self.real_ssh_endpoint_verifier = archive_tool._require_effective_ssh_endpoint
        self.real_remote_authority_verifier = archive_tool._verify_remote_authority
        self.remote_authority_patch = patch.object(
            archive_tool,
            "_verify_remote_authority",
        )
        self.ssh_endpoint_patch.start()
        self.remote_authority_patch.start()
        self.addCleanup(self.ssh_endpoint_patch.stop)
        self.addCleanup(self.remote_authority_patch.stop)

    def _limits(self) -> storage.CampaignStorageLimits:
        return storage.CampaignStorageLimits(
            max_campaign_bytes=1024 * 1024,
            min_campaign_free_bytes=storage.DEFAULT_ACTIVE_FILESYSTEM_FREE_BYTES,
            min_remote_free_bytes=300 * 1024**3,
            max_entries=100,
        )

    def _options(self, directory: str) -> archive_tool.RemoteArchiveOptions:
        base = Path(directory).resolve()
        receipt = base / "receipt.json"
        tools = base / "tools"
        tools.mkdir()
        local_tools: dict[str, dict[str, str]] = {}
        for name in archive_authority.LOCAL_TOOL_NAMES:
            path = tools / name
            path.write_bytes(f"#!/bin/sh\n# {name}\n".encode())
            path.chmod(0o755)
            local_tools[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        rclone_config = base / "rclone.conf"
        rclone_config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
        rclone_config.chmod(0o600)
        remote_tools = {
            name: {
                "path": f"/usr/bin/{name}",
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for name in archive_authority.REMOTE_TOOL_NAMES
        }
        remote_helpers = {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in archive_authority.REMOTE_HELPER_PATHS
        }
        payload = {
            "endpoint": {
                "rclone_base": archive_tool.DEFAULT_RCLONE_BASE,
                "rclone_config": {
                    "path": str(rclone_config),
                    "sha256": hashlib.sha256(rclone_config.read_bytes()).hexdigest(),
                },
                "ssh_effective_config_sha256": "c" * 64,
                "ssh_connection": {
                    "address": "192.0.2.40",
                    "authenticated_host": "trainer-node.example.invalid",
                    "authentication": "publickey",
                    "host_key": (
                        "ssh-ed25519 SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    ),
                    "port": 22,
                },
                "ssh_target": "trainer-node",
            },
            "local_tools": local_tools,
            "remote_helpers": remote_helpers,
            "remote_tools": remote_tools,
            "schema_version": (archive_authority.ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION),
            "source": {
                "git_sha": SOURCE_SHA,
                "manifest_sha256": SOURCE_MANIFEST,
                "remote_repository_root": "/srv/evolution-sim/checkout",
            },
        }
        authority_path = base / "archive-authority.json"
        authority_bytes = storage.canonical_json_bytes(payload)
        authority_path.write_bytes(authority_bytes)
        return archive_tool.RemoteArchiveOptions(
            ssh_target="trainer-node",
            remote_repository_root="/srv/evolution-sim/checkout",
            remote_active_campaign_root="/srv/evolution-sim/runs/campaign-1",
            remote_closed_bundle_dir="/srv/evolution-sim/closed/bundle-1",
            remote_staging_dir="/srv/evolution-sim/staging/bundle-1",
            campaign_id=CAMPAIGN_ID,
            bundle_id=BUNDLE_ID,
            source_git_sha=SOURCE_SHA,
            source_manifest_sha256=SOURCE_MANIFEST,
            expected_marker_sha256="c" * 64,
            expected_entry_count=5,
            expected_file_count=3,
            expected_directory_count=2,
            expected_total_file_bytes=123,
            receipt_path=receipt,
            tool_authority_path=authority_path,
            tool_authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
            limits=self._limits(),
        )

    def _build(
        self,
        options: archive_tool.RemoteArchiveOptions | None = None,
    ) -> dict[str, object]:
        marker_sha256 = "c" * 64
        archive_name = f"bundle-1-{marker_sha256[:16]}.tar.zst"
        names = (
            archive_name,
            f"{archive_name}.manifest.json",
            f"{archive_name}.sha256",
        )
        objects = [
            {
                "name": name,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "size": index + 11,
                "source_path": f"/srv/evolution-sim/staging/bundle-1/{name}",
            }
            for index, name in enumerate(names)
        ]
        payload: dict[str, object] = {
            "archive_name": archive_name,
            "bundle_id": BUNDLE_ID,
            "campaign_id": CAMPAIGN_ID,
            "closed_bundle": {
                "marker_sha256": marker_sha256,
                "entry_count": 5,
                "file_count": 3,
                "directory_count": 2,
                "total_file_bytes": 123,
            },
            "objects": objects,
            "remote_input_pruned": False,
            "remote_staging_pruned": False,
            "schema_version": archive_tool.REMOTE_BUILD_SCHEMA_VERSION,
            "source_git_sha": SOURCE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST,
            "status": "deterministic_tar_zst_triple_validated",
        }
        if options is not None:
            authority = archive_tool._load_and_validate_authority(options)
            payload["tool_authority"] = (
                archive_tool.RemoteBuildAuthority.from_archive_authority(
                    authority
                ).receipt_record()
            )
        return payload

    def _inventory(self, build: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "id": f"drive-{index}",
                "name": row["name"],
                "size": row["size"],
            }
            for index, row in enumerate(build["objects"], start=1)  # type: ignore[arg-type]
        ]

    def test_default_drive_floor_and_destination_are_sealed(self) -> None:
        self.assertEqual(
            storage.CampaignStorageLimits().min_remote_free_bytes,
            300 * 1024**3,
        )
        self.assertEqual(
            archive_tool.DEFAULT_RCLONE_BASE,
            "gdrive:evolution-sim-backups/archives/open-ecology",
        )

    def test_fake_path_cannot_replace_pinned_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            authority = archive_tool._load_and_validate_authority(options)
            hostile = Path(directory).resolve() / "hostile"
            hostile.mkdir()
            marker = Path(directory).resolve() / "path-tool-ran"
            for name in archive_authority.LOCAL_TOOL_NAMES:
                executable = hostile / name
                executable.write_text(
                    f"#!/bin/sh\n/usr/bin/touch {marker}\n",
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            git_pin = authority.local_tool("git")
            with patch.dict(os.environ, {"PATH": str(hostile)}):
                completed = archive_tool._run_local_tool(
                    authority,
                    "git",
                    (git_pin.path,),
                )
            self.assertEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())

    def test_authenticated_ssh_endpoint_replacement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            authority = archive_tool._load_and_validate_authority(options)
            effective_config = b"host trainer-node\nhostname 192.0.2.40\n"
            authority = replace(
                authority,
                ssh_effective_config_sha256=(
                    hashlib.sha256(effective_config).hexdigest()
                ),
            )
            changed_connection = b"\n".join(
                (
                    (
                        b"debug1: Server host key: ssh-ed25519 "
                        b"SHA256:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
                    ),
                    (
                        b"Authenticated to replacement.example.invalid "
                        b'([192.0.2.41]:22) using "publickey".'
                    ),
                    b"",
                )
            )
            with patch.object(
                archive_tool,
                "_run_pinned",
                side_effect=(
                    subprocess.CompletedProcess([], 0, effective_config, b""),
                    subprocess.CompletedProcess([], 0, b"", changed_connection),
                ),
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "host key/address differs",
                ):
                    self.real_ssh_endpoint_verifier(authority)

    def test_pinned_local_executable_replacement_fails_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            authority = archive_tool._load_and_validate_authority(options)
            rclone_path = Path(authority.local_tool("rclone").path)
            rclone_path.write_text("#!/bin/sh\nexit 19\n", encoding="utf-8")
            rclone_path.chmod(0o755)
            with self.assertRaisesRegex(
                archive_authority.ArchiveAuthorityError,
                "SHA256 mismatch",
            ):
                archive_tool._run_local_tool(
                    authority,
                    "rclone",
                    archive_tool._rclone_command(authority, "version"),
                )

    def test_remote_helper_replacement_fails_external_authority_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            authority = archive_tool._load_and_validate_authority(options)
            expected = {pin.path: pin.sha256 for _, pin in authority.remote_tools}
            repository = PurePosixPath(options.remote_repository_root)
            for relative_path, digest in authority.remote_helpers:
                expected[str(repository / relative_path)] = digest
            replaced_path = str(repository / archive_authority.REMOTE_HELPER_PATHS[-1])
            expected[replaced_path] = "0" * 64
            stdout = "".join(
                f"{digest}  {path}\n" for path, digest in sorted(expected.items())
            ).encode()
            with patch.object(
                archive_tool,
                "_run_local_tool",
                return_value=subprocess.CompletedProcess([], 0, stdout, b""),
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "executable or helper differs",
                ):
                    self.real_remote_authority_verifier(options, authority)

    def test_authority_file_replacement_fails_its_external_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            options.tool_authority_path.write_bytes(b"{}\n")
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "SHA256 mismatch",
            ):
                archive_tool._load_and_validate_authority(options)

    def test_source_head_or_manifest_drift_fails_closed(self) -> None:
        with patch.object(
            archive_tool,
            "_git_output",
            return_value="f" * 40,
        ):
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "HEAD differs",
            ):
                archive_tool._require_exact_source_binding(
                    archive_tool._REPOSITORY_ROOT,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                )
        with (
            patch.object(
                archive_tool,
                "_git_output",
                side_effect=(SOURCE_SHA, "HEAD", ""),
            ),
            patch.object(
                archive_tool,
                "source_file_hash_manifest",
                return_value={"aggregate_sha256": "0" * 64},
            ),
            patch.object(archive_tool, "_require_committed_file_bytes"),
        ):
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "source manifest drifted",
            ):
                archive_tool._require_exact_source_binding(
                    archive_tool._REPOSITORY_ROOT,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                )

        with patch.object(
            archive_tool,
            "_git_output",
            side_effect=(SOURCE_SHA, "codex/mutable-branch"),
        ):
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "must be detached",
            ):
                archive_tool._require_exact_source_binding(
                    archive_tool._REPOSITORY_ROOT,
                    source_git_sha=SOURCE_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST,
                )

    def test_committed_script_bytes_cannot_hide_behind_assume_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            script = repository / "scripts" / "producer.py"
            script.parent.mkdir()
            script.write_text("print('committed')\n", encoding="utf-8")
            commands = (
                ("init", "-q"),
                ("add", "scripts/producer.py"),
                (
                    "-c",
                    "user.name=Storage Test",
                    "-c",
                    "user.email=storage@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ),
            )
            for command in commands:
                subprocess.run(
                    ("git", "-C", str(repository), *command),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            source_git_sha = (
                subprocess.run(
                    ("git", "-C", str(repository), "rev-parse", "HEAD"),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                .stdout.decode()
                .strip()
            )
            archive_tool._require_committed_file_bytes(
                repository,
                source_git_sha=source_git_sha,
                relative_path=PurePosixPath("scripts/producer.py"),
            )
            subprocess.run(
                (
                    "git",
                    "-C",
                    str(repository),
                    "update-index",
                    "--assume-unchanged",
                    "scripts/producer.py",
                ),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            script.write_text("print('tampered')\n", encoding="utf-8")
            status = subprocess.run(
                (
                    "git",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(status.stdout, b"")
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "differs from committed bytes",
            ):
                archive_tool._require_committed_file_bytes(
                    repository,
                    source_git_sha=source_git_sha,
                    relative_path=PurePosixPath("scripts/producer.py"),
                )

    def test_remote_build_object_must_come_from_exact_staging_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            build = self._build(options)
            first = build["objects"][0]  # type: ignore[index]
            first["source_path"] = f"/tmp/{first['name']}"  # type: ignore[index]
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "evidence is invalid",
            ):
                archive_tool._validated_remote_build(build, options=options)

    def test_valid_resealed_bundle_cannot_replace_externally_pinned_closure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            options = self._options(directory)
            authority = archive_tool._load_and_validate_authority(options)
            execution_authority = (
                archive_tool.RemoteBuildAuthority.from_archive_authority(authority)
            )
            repository = base / "repository"
            active = base / "active"
            bundle = base / BUNDLE_ID
            staging = base / "staging"
            for path in (repository, active, bundle):
                path.mkdir()
            (active / "writer").write_bytes(b"active")
            (bundle / "evidence").write_bytes(b"valid-resealed-replacement")
            snapshot = storage.seal_closed_bundle(
                active,
                bundle,
                campaign_id=CAMPAIGN_ID,
                bundle_id=BUNDLE_ID,
                source_git_sha=SOURCE_SHA,
                source_manifest_sha256=SOURCE_MANIFEST,
                limits=self._limits(),
            )
            with patch.object(
                archive_tool,
                "_require_exact_source_binding",
                return_value=repository,
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "externally pinned closure",
                ):
                    archive_tool.build_remote_closed_bundle_archive(
                        repository_root=repository,
                        active_campaign_root=active,
                        closed_bundle_dir=bundle,
                        staging_dir=staging,
                        campaign_id=CAMPAIGN_ID,
                        bundle_id=BUNDLE_ID,
                        source_git_sha=SOURCE_SHA,
                        source_manifest_sha256=SOURCE_MANIFEST,
                        expected_marker_sha256="0" * 64,
                        expected_entry_count=len(snapshot.scan.entries),
                        expected_file_count=snapshot.scan.file_count,
                        expected_directory_count=snapshot.scan.directory_count,
                        expected_total_file_bytes=snapshot.scan.total_file_bytes,
                        limits=self._limits(),
                        execution_authority=execution_authority,
                    )

    def test_producer_manifest_must_equal_descriptor_safe_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            active = base / "active"
            bundle = base / BUNDLE_ID
            staging = base / "staging"
            active.mkdir()
            bundle.mkdir()
            staging.mkdir()
            (active / "writer").write_bytes(b"active")
            (bundle / "evidence").write_bytes(b"sealed")
            snapshot = storage.seal_closed_bundle(
                active,
                bundle,
                campaign_id=CAMPAIGN_ID,
                bundle_id=BUNDLE_ID,
                source_git_sha=SOURCE_SHA,
                source_manifest_sha256=SOURCE_MANIFEST,
                limits=self._limits(),
            )
            archive = staging / "bundle.tar.zst"
            manifest = staging / "bundle.tar.zst.manifest.json"
            sidecar = staging / "bundle.tar.zst.sha256"
            archive.write_bytes(b"not-real-zstd-for-unit-test")
            manifest.write_bytes(
                storage.canonical_json_bytes(snapshot.producer_manifest)
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
            with patch.object(
                archive_tool,
                "_validate_archive_against_snapshot",
                return_value=digest,
            ):
                rows = archive_tool._validate_producer_objects(
                    snapshot,
                    (archive, manifest, sidecar),
                )
            self.assertEqual(len(rows), 3)

            os.chmod(manifest, 0o644)
            bad_manifest = dict(snapshot.producer_manifest)
            bad_manifest["file_count"] = int(bad_manifest["file_count"]) + 1
            manifest.write_bytes(storage.canonical_json_bytes(bad_manifest))
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "descriptor-safe preflight",
            ):
                archive_tool._validate_producer_objects(
                    snapshot,
                    (archive, manifest, sidecar),
                )

    def test_inventory_rejects_surplus_duplicate_name_and_duplicate_id(self) -> None:
        expected = ("a", "b", "c")
        good_payload = [
            {
                "Name": name,
                "Path": name,
                "Size": index,
                "ID": f"id-{name}",
                "IsDir": False,
            }
            for index, name in enumerate(expected, start=1)
        ]
        with patch.object(
            archive_tool,
            "_run",
            return_value=subprocess.CompletedProcess(
                [],
                0,
                json.dumps(good_payload).encode(),
                b"",
            ),
        ):
            self.assertEqual(
                len(
                    archive_tool._inspect_remote_inventory(
                        "gdrive:x",
                        expected_names=expected,
                        require_complete=True,
                    )
                ),
                3,
            )

        cases = (
            [
                {
                    "Name": "a",
                    "Path": "a",
                    "Size": 1,
                    "ID": "id-a",
                    "IsDir": False,
                },
                {
                    "Name": "surplus",
                    "Path": "surplus",
                    "Size": 1,
                    "ID": "id-x",
                    "IsDir": False,
                },
            ],
            [
                {
                    "Name": "a",
                    "Path": "a",
                    "Size": 1,
                    "ID": "id-a",
                    "IsDir": False,
                },
                {
                    "Name": "a",
                    "Path": "a",
                    "Size": 1,
                    "ID": "id-b",
                    "IsDir": False,
                },
            ],
            [
                {
                    "Name": "a",
                    "Path": "a",
                    "Size": 1,
                    "ID": "id-a",
                    "IsDir": False,
                },
                {
                    "Name": "b",
                    "Path": "b",
                    "Size": 1,
                    "ID": "id-a",
                    "IsDir": False,
                },
            ],
            [
                {
                    "Name": "a",
                    "Path": "a",
                    "Size": 0,
                    "ID": "dir-a",
                    "IsDir": True,
                }
            ],
            [
                {
                    "Name": "a",
                    "Path": "nested/a",
                    "Size": 1,
                    "ID": "id-a",
                    "IsDir": False,
                }
            ],
        )
        for payload in cases:
            with self.subTest(payload=payload):
                completed = subprocess.CompletedProcess(
                    [],
                    0,
                    json.dumps(payload).encode(),
                    b"",
                )
                with patch.object(archive_tool, "_run", return_value=completed):
                    with self.assertRaises(
                        archive_tool.OpenEcologyArchiveError,
                    ):
                        archive_tool._inspect_remote_inventory(
                            "gdrive:x",
                            expected_names=expected,
                            require_complete=False,
                        )

    def test_absent_drive_prefix_exit_three_is_an_empty_retry_prefix(self) -> None:
        missing = subprocess.CompletedProcess(
            [],
            3,
            b"[\n",
            b"ERROR: directory not found\n",
        )
        with patch.object(archive_tool, "_run", return_value=missing):
            rows = archive_tool._inspect_remote_inventory(
                "gdrive:missing",
                expected_names=("a", "b", "c"),
                require_complete=False,
            )
        self.assertEqual(rows, [])

    def test_rclone_check_requires_three_canonical_rows_and_no_one_way(self) -> None:
        expected = {
            name: {"sha256": character * 64, "size": 1}
            for name, character in zip(("a", "b", "c"), "abc", strict=True)
        }
        good = subprocess.CompletedProcess(
            [],
            0,
            b"= a\n= b\n= c\n",
            b"",
        )
        with patch.object(archive_tool, "_run", return_value=good) as run:
            rows = archive_tool._run_independent_rclone_check(
                "gdrive:prefix",
                expected=expected,
            )
        command = run.call_args.args[0]
        self.assertIn("--checkfile", command)
        self.assertIn("SHA-256", command)
        self.assertIn("--combined", command)
        self.assertNotIn("--one-way", command)
        self.assertEqual(rows, ["= a", "= b", "= c"])

        bad = subprocess.CompletedProcess([], 0, b"= a\n= b\n- c\n", b"")
        with patch.object(archive_tool, "_run", return_value=bad):
            with self.assertRaisesRegex(
                archive_tool.OpenEcologyArchiveError,
                "exactly three canonical",
            ):
                archive_tool._run_independent_rclone_check(
                    "gdrive:prefix",
                    expected=expected,
                )

    def test_partial_exact_retry_fills_only_missing_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            build = self._build(options)
            objects = archive_tool._validated_remote_build(build, options=options)
            first = objects[0]
            initial = [
                {
                    "id": "drive-existing",
                    "name": first["name"],
                    "size": first["size"],
                }
            ]
            final = self._inventory(build)

            def readback(path: str, **_: object) -> dict[str, object]:
                name = path.rsplit("/", 1)[1]
                row = next(row for row in objects if row["name"] == name)
                return {"sha256": row["sha256"], "size": row["size"]}

            with (
                patch.object(
                    archive_tool,
                    "_require_drive_quota",
                    return_value=400 * 1024**3,
                ),
                patch.object(archive_tool, "_remote_build", return_value=build),
                patch.object(
                    archive_tool,
                    "_inspect_remote_inventory",
                    side_effect=(initial, final, final),
                ),
                patch.object(
                    archive_tool,
                    "_rclone_readback",
                    side_effect=readback,
                ),
                patch.object(
                    archive_tool,
                    "_stream_remote_object",
                ) as upload,
                patch.object(
                    archive_tool,
                    "_run_independent_rclone_check",
                    return_value=[
                        f"= {row['name']}"
                        for row in sorted(objects, key=lambda row: str(row["name"]))
                    ],
                ),
            ):
                result = archive_tool.execute_remote_archive(options)
            self.assertEqual(upload.call_count, 2)
            uploaded_paths = {
                invocation.kwargs["destination_path"].rsplit("/", 1)[1]
                for invocation in upload.call_args_list
            }
            self.assertEqual(
                uploaded_paths,
                {str(row["name"]) for row in objects[1:]},
            )
            self.assertTrue(options.receipt_path.is_file())
            self.assertEqual(
                result["destination_prefix"],
                f"{archive_tool.DEFAULT_RCLONE_BASE}/{CAMPAIGN_ID}/{BUNDLE_ID}",
            )
            receipt_payload = storage.load_verified_receipt(options.receipt_path)[
                "payload"
            ]
            self.assertEqual(
                receipt_payload["archive_tool_authority"]["authority_sha256"],  # type: ignore[index]
                options.tool_authority_sha256,
            )
            self.assertEqual(
                receipt_payload["drive_quota"],  # type: ignore[index]
                {
                    "free_after_bytes": 400 * 1024**3,
                    "free_before_bytes": 400 * 1024**3,
                    "minimum_free_bytes": 300 * 1024**3,
                },
            )

    def test_existing_mismatch_writes_no_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            build = self._build(options)
            objects = archive_tool._validated_remote_build(build, options=options)
            initial = [
                {
                    "id": "drive-existing",
                    "name": objects[0]["name"],
                    "size": objects[0]["size"],
                }
            ]
            with (
                patch.object(
                    archive_tool,
                    "_require_drive_quota",
                    return_value=400 * 1024**3,
                ),
                patch.object(archive_tool, "_remote_build", return_value=build),
                patch.object(
                    archive_tool,
                    "_inspect_remote_inventory",
                    return_value=initial,
                ),
                patch.object(
                    archive_tool,
                    "_rclone_readback",
                    return_value={"sha256": "0" * 64, "size": objects[0]["size"]},
                ),
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "mismatch",
                ):
                    archive_tool.execute_remote_archive(options)
            self.assertFalse(options.receipt_path.exists())

    def test_drive_inventory_identity_or_check_failure_writes_no_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            build = self._build(options)
            objects = archive_tool._validated_remote_build(build, options=options)
            before = self._inventory(build)
            after = [dict(row) for row in before]
            after[0]["id"] = "drive-replaced"

            def readback(path: str, **_: object) -> dict[str, object]:
                name = path.rsplit("/", 1)[1]
                row = next(row for row in objects if row["name"] == name)
                return {"sha256": row["sha256"], "size": row["size"]}

            with (
                patch.object(
                    archive_tool,
                    "_require_drive_quota",
                    return_value=400 * 1024**3,
                ),
                patch.object(archive_tool, "_remote_build", return_value=build),
                patch.object(
                    archive_tool,
                    "_inspect_remote_inventory",
                    side_effect=([], before, after),
                ),
                patch.object(
                    archive_tool,
                    "_rclone_readback",
                    side_effect=readback,
                ),
                patch.object(archive_tool, "_stream_remote_object"),
                patch.object(
                    archive_tool,
                    "_run_independent_rclone_check",
                    return_value=[
                        f"= {row['name']}"
                        for row in sorted(
                            objects,
                            key=lambda row: str(row["name"]),
                        )
                    ],
                ),
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "changed during verification",
                ):
                    archive_tool.execute_remote_archive(options)
            self.assertFalse(options.receipt_path.exists())

        with tempfile.TemporaryDirectory() as directory:
            options = self._options(directory)
            build = self._build(options)
            objects = archive_tool._validated_remote_build(build, options=options)
            final = self._inventory(build)

            def readback(path: str, **_: object) -> dict[str, object]:
                name = path.rsplit("/", 1)[1]
                row = next(row for row in objects if row["name"] == name)
                return {"sha256": row["sha256"], "size": row["size"]}

            with (
                patch.object(
                    archive_tool,
                    "_require_drive_quota",
                    return_value=400 * 1024**3,
                ),
                patch.object(archive_tool, "_remote_build", return_value=build),
                patch.object(
                    archive_tool,
                    "_inspect_remote_inventory",
                    side_effect=([], final),
                ),
                patch.object(
                    archive_tool,
                    "_rclone_readback",
                    side_effect=readback,
                ),
                patch.object(archive_tool, "_stream_remote_object"),
                patch.object(
                    archive_tool,
                    "_run_independent_rclone_check",
                    side_effect=archive_tool.OpenEcologyArchiveError(
                        "combined output mismatch"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    archive_tool.OpenEcologyArchiveError,
                    "combined output mismatch",
                ):
                    archive_tool.execute_remote_archive(options)
            self.assertFalse(options.receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
