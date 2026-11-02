from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from gmssl import sm2

from facecheck.crypto import SM2_PRIVATE_KEY, SM2_PUBLIC_KEY
from facecheck.database import TaskRepository
from facecheck.feature_store import FeatureStore
from facecheck.key_management import (
    KeyManagementError,
    PinnedVideoKeyDirectory,
    PublicVideoKeyRecord,
    Sm2MemoryKeyProvider,
    VideoKeyProviderRegistry,
    WRAPPED_KEY_ALGORITHM,
)
from facecheck.rotation_control import (
    RotationControlError,
    RotationControlRepository,
    VideoKeyRotationWorker,
)
from facecheck.service import VideoTaskService
from facecheck.storage import store_encrypted_video, store_enveloped_video


def _keypair(number: int) -> tuple[str, str]:
    private = f"{number:064x}"
    calculator = sm2.CryptSM2(public_key="", private_key=private)
    public = calculator._kg(int(private, 16), sm2.default_ecc_table["g"])
    return private, public


def _completed_v1_task(tmp_path: Path, database: Path, task_id: str = "rotation-task-1"):
    source = tmp_path / "source-video.bin"
    source.write_bytes(b"short synthetic video payload" * 5)
    encryption_key, mac_key = b"0123456789abcdef", b"legacy-test-mac-key"
    storage_root = tmp_path / "storage"
    stored = store_encrypted_video(
        source, storage_root, encryption_key, mac_key,
        chunk_size=31, key_version="api-video-key-v1", video_id=task_id,
    )
    manifest_bytes = (stored.directory / "manifest.json").read_bytes()
    repository = TaskRepository(database)
    repository.create_task(task_id, "sample.mp4", source.stat().st_size)
    repository.transition(task_id, "processing")
    repository.transition(task_id, "completed", storage_id=task_id)
    repository.update_metrics(task_id, {
        "key_id": "legacy-api-v1",
        "key_version": "api-video-key-v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    })
    return source, storage_root, encryption_key, mac_key, stored, repository


def test_pinned_key_directory_and_provider_registry_fail_closed(tmp_path: Path):
    first = PublicVideoKeyRecord("recipient-old", WRAPPED_KEY_ALGORITHM, SM2_PUBLIC_KEY, "decrypt_only")
    second_private, second_public = _keypair(2)
    second = PublicVideoKeyRecord("recipient-new", WRAPPED_KEY_ALGORITHM, second_public, "active")
    directory_file = tmp_path / "public-keys.json"
    directory_file.write_text(json.dumps({
        "version": 1, "records": [first.as_dict(), second.as_dict()],
    }), encoding="utf-8")
    directory = PinnedVideoKeyDirectory.from_file(
        directory_file,
        pinned_sha256=PinnedVideoKeyDirectory.digest_records([first, second]),
    )
    assert directory.active_for_wrap().key_id == "recipient-new"
    assert directory.resolve("recipient-old").status == "decrypt_only"
    with pytest.raises(KeyManagementError, match="pin mismatch"):
        PinnedVideoKeyDirectory([first, second], "0" * 64)

    old_provider = Sm2MemoryKeyProvider(
        key_id="recipient-old", public_key=SM2_PUBLIC_KEY, private_key=SM2_PRIVATE_KEY,
    )
    new_provider = Sm2MemoryKeyProvider(
        key_id="recipient-new", public_key=second_public, private_key=second_private,
    )
    registry = VideoKeyProviderRegistry([old_provider, new_provider], "recipient-new")
    assert registry.resolve("recipient-old") is old_provider
    with pytest.raises(KeyManagementError, match="registered references"):
        registry.revoke("recipient-old", readiness_check=lambda _: False)
    registry.revoke("recipient-old", readiness_check=lambda _: True)
    with pytest.raises(KeyManagementError, match="unavailable"):
        registry.resolve("recipient-old")


def test_database_rotation_switches_pointer_atomically_and_blocks_early_revoke(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    source, storage_root, old_encryption, old_mac, stored, task_repository = _completed_v1_task(
        tmp_path, database,
    )
    backup_root = tmp_path / "backup-root"
    backup_root.mkdir()
    shutil.copytree(stored.directory, backup_root / "rotation-task-1")
    feature_store = FeatureStore(
        tmp_path / "protected-features", old_encryption, old_mac,
    )
    cache_directory = feature_store.root / "rotation-task-1"
    cache_directory.mkdir(parents=True)
    (cache_directory / "frame-cache.bin").write_bytes(b"synthetic authenticated cache fixture")
    control = RotationControlRepository(database)
    job_id = control.create_plan_from_storage(
        storage_root, "legacy-api-v1", "recipient-new",
        feature_store=feature_store, backup_roots=[backup_root],
        backup_inventory_complete=True, writers_fenced=True, readers_drained=True,
    )
    item = control.claim_next(job_id)
    assert item["task_id"] == "rotation-task-1"
    assert item["attempts"] == 1
    old_source_artifact = next(
        artifact for artifact in control.get_job(job_id)["artifacts"]
        if artifact["kind"] == "replica"
    )
    with pytest.raises(RotationControlError, match="must reference the target key"):
        control.finish_artifact(
            job_id, old_source_artifact["artifact_id"], disposition="deleted",
            evidence="must not remove the active task object",
        )

    new_private, new_public = _keypair(3)
    new_provider = Sm2MemoryKeyProvider(
        key_id="recipient-new", public_key=new_public, private_key=new_private,
    )
    worker = VideoKeyRotationWorker(
        control, storage_root, target_provider=new_provider,
        source_provider_resolver=lambda key_id: (_ for _ in ()).throw(KeyError(key_id)),
        legacy_encryption_key=old_encryption,
        legacy_mac_key=old_mac,
        feature_store=feature_store,
    )
    with pytest.raises(RotationControlError, match="shutdown must be confirmed"):
        control.recover_interrupted(job_id, worker_stopped_confirmed=False)
    assert control.recover_interrupted(job_id, worker_stopped_confirmed=True) == 1
    result = worker.process_next(job_id)
    assert result["state"] == "switched"
    assert result["cache_cleanup"][0]["deleted_files"] == 1
    assert not cache_directory.exists()
    task = task_repository.get("rotation-task-1")
    assert task["storage_id"] == result["target_storage_id"]
    assert task["processing_metrics"]["key_id"] == "recipient-new"
    assert task["processing_metrics"]["manifest_sha256"] == result["target_manifest_sha256"]

    restored = tmp_path / "new-restored.bin"
    service = VideoTaskService(
        task_repository, storage_root,
        key_provider_resolver=lambda key_id: new_provider,
    )
    service.restore_task_storage(task, storage_root / task["storage_id"], restored)
    assert restored.read_bytes() == source.read_bytes()

    readiness = control.revocation_readiness(job_id)
    assert not readiness["ready"]
    assert readiness["unresolved_artifacts"] == 2  # rollback source and backup remain
    with pytest.raises(RotationControlError, match="incomplete inventory"):
        control.revoke_old_key(job_id, lambda _: "fake-receipt")

    source_manifest_sha = hashlib.sha256((stored.directory / "manifest.json").read_bytes()).hexdigest()
    assert source_manifest_sha == item["source_manifest_sha256"]
    shutil.rmtree(stored.directory)
    source_artifact = next(a for a in control.get_job(job_id)["artifacts"] if a["kind"] == "replica")
    control.finish_artifact(
        job_id, source_artifact["artifact_id"], disposition="deleted",
        evidence="synthetic test removed retained source object",
    )
    assert not control.revocation_readiness(job_id)["ready"]
    backup_artifact = next(a for a in control.get_job(job_id)["artifacts"] if a["kind"] == "backup")
    shutil.rmtree(backup_root / "rotation-task-1")
    control.finish_artifact(
        job_id, backup_artifact["artifact_id"], disposition="deleted",
        evidence="synthetic test removed backup object after retention check",
    )
    readiness = control.revocation_readiness(job_id)
    assert readiness["ready"]
    revoked = control.revoke_old_key(job_id, lambda key_id: "local-test-receipt:" + key_id)
    assert revoked["revoked"]
    assert control.get_job(job_id)["job"]["status"] == "revoked"


def test_rotation_compare_and_swap_rejects_changed_task_reference(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    _, _, _, _, stored, tasks = _completed_v1_task(tmp_path, database, "rotation-task-2")
    digest = hashlib.sha256((stored.directory / "manifest.json").read_bytes()).hexdigest()
    control = RotationControlRepository(database)
    job_id = control.create_plan(
        "legacy-api-v1", "recipient-next",
        [{"task_id": "rotation-task-2", "source_storage_id": "rotation-task-2",
          "source_manifest_sha256": digest}],
        inventory_sha256="a" * 64, inventory_complete=True, writers_fenced=True,
        readers_drained=True,
    )
    control.claim_next(job_id)
    tasks.update_metrics("rotation-task-2", {"manifest_sha256": "b" * 64})
    with pytest.raises(RotationControlError, match="reference changed"):
        control.complete_verified_and_switch(
            job_id, "rotation-task-2", target_storage_id="rot-test",
            target_manifest_sha256="c" * 64, target_key_version="v2",
        )
    assert control.get_job(job_id)["items"][0]["state"] == "stale"
    assert tasks.get("rotation-task-2")["storage_id"] == "rotation-task-2"


def test_inventory_fails_closed_on_hidden_staging_and_unannotated_legacy_key(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    _, storage_root, _, _, stored, tasks = _completed_v1_task(
        tmp_path, database, "rotation-task-legacy-metadata",
    )
    manifest_digest = hashlib.sha256((stored.directory / "manifest.json").read_bytes()).hexdigest()
    tasks.update_metrics("rotation-task-legacy-metadata", {"manifest_sha256": manifest_digest})
    (storage_root / ".rotation-stage-abandoned").mkdir()

    control = RotationControlRepository(database)
    job_id = control.create_plan_from_storage(
        storage_root, "legacy-api-v1", "recipient-fail-closed",
        backup_inventory_complete=True,
    )

    readiness = control.revocation_readiness(job_id)
    assert not readiness["ready"]
    assert "inventory completeness has not been attested" in readiness["blockers"]
    assert readiness["current_database_references"] == ["rotation-task-legacy-metadata"]


def test_rotation_plan_cannot_reuse_a_source_key_id(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    _, storage_root, _, _, _, _ = _completed_v1_task(tmp_path, database)
    control = RotationControlRepository(database)
    control.create_plan_from_storage(
        storage_root, "legacy-api-v1", "recipient-first",
        backup_inventory_complete=True,
    )

    with pytest.raises(RotationControlError, match="already exists"):
        control.create_plan_from_storage(
            storage_root, "legacy-api-v1", "recipient-second",
            backup_inventory_complete=True,
        )


def test_recovery_reuses_a_verified_target_published_before_worker_interruption(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    source, storage_root, old_encryption, old_mac, stored, _ = _completed_v1_task(
        tmp_path, database, "rotation-task-3",
    )
    control = RotationControlRepository(database)
    job_id = control.create_plan_from_storage(
        storage_root, "legacy-api-v1", "recipient-recover",
        backup_inventory_complete=True, writers_fenced=True, readers_drained=True,
    )
    assert control.claim_next(job_id) is not None
    control.recover_interrupted(job_id, worker_stopped_confirmed=True)

    new_private, new_public = _keypair(4)
    new_provider = Sm2MemoryKeyProvider(
        key_id="recipient-recover", public_key=new_public, private_key=new_private,
    )
    target_id = "rot-" + hashlib.sha256(
        (job_id + ":rotation-task-3").encode()
    ).hexdigest()[:32]
    prepared_root = tmp_path / "prepared-target"
    prepared = store_enveloped_video(
        source, prepared_root, new_provider, chunk_size=31,
        key_version="enveloped-recipient-recover", video_id="rotation-task-3",
    )
    target_hash = hashlib.sha256((prepared.directory / "manifest.json").read_bytes()).hexdigest()
    os.replace(prepared.directory, storage_root / target_id)

    worker = VideoKeyRotationWorker(
        control, storage_root, target_provider=new_provider,
        source_provider_resolver=lambda key_id: (_ for _ in ()).throw(KeyError(key_id)),
        legacy_encryption_key=old_encryption, legacy_mac_key=old_mac,
    )
    result = worker.process_next(job_id)
    assert result["target_manifest_sha256"] == target_hash
    assert result["target_storage_id"] == target_id
    assert control.get_job(job_id)["items"][0]["state"] == "switched"


def test_database_worker_rewraps_v2_without_changing_video_chunks(tmp_path: Path):
    database = tmp_path / "facecheck.sqlite3"
    task_id = "rotation-v2-task"
    source = tmp_path / "v2-source.bin"
    source.write_bytes(b"small v2 object for recipient-key rewrap" * 3)
    old_provider = Sm2MemoryKeyProvider(
        key_id="recipient-before", public_key=SM2_PUBLIC_KEY,
        private_key=SM2_PRIVATE_KEY,
    )
    new_private, new_public = _keypair(5)
    new_provider = Sm2MemoryKeyProvider(
        key_id="recipient-after", public_key=new_public, private_key=new_private,
    )
    storage_root = tmp_path / "v2-storage"
    stored = store_enveloped_video(
        source, storage_root, old_provider, chunk_size=29,
        key_version="v2-video-key", video_id=task_id,
    )
    first_chunk = (stored.directory / "chunks" / "00000000.bin").read_bytes()
    manifest_hash = hashlib.sha256((stored.directory / "manifest.json").read_bytes()).hexdigest()
    tasks = TaskRepository(database)
    tasks.create_task(task_id, "sample.mp4", source.stat().st_size)
    tasks.transition(task_id, "processing")
    tasks.transition(task_id, "completed", storage_id=task_id)
    tasks.update_metrics(task_id, {"key_id": old_provider.key_id,
                                   "key_version": "v2-video-key",
                                   "manifest_sha256": manifest_hash})
    control = RotationControlRepository(database)
    job_id = control.create_plan_from_storage(
        storage_root, old_provider.key_id, new_provider.key_id,
        backup_inventory_complete=True, writers_fenced=True, readers_drained=True,
    )
    worker = VideoKeyRotationWorker(
        control, storage_root, target_provider=new_provider,
        source_provider_resolver=lambda key_id: old_provider,
    )
    result = worker.process_next(job_id)
    task = tasks.get(task_id)
    target = storage_root / task["storage_id"]
    assert (target / "chunks" / "00000000.bin").read_bytes() == first_chunk
    assert json.loads((target / "manifest.json").read_text())["key_envelope"]["key_id"] == new_provider.key_id
    restored = tmp_path / "v2-worker-restored.bin"
    VideoTaskService(tasks, storage_root, key_provider_resolver=lambda _: new_provider).restore_task_storage(
        task, target, restored,
    )
    assert restored.read_bytes() == source.read_bytes()
    assert result["state"] == "switched"


def test_feature_cache_inventory_and_invalidation_are_scoped_to_one_video(tmp_path: Path):
    store = FeatureStore(tmp_path / "cache", b"0123456789abcdef", b"test-mac-key")
    directory = store.root / "video-a"
    directory.mkdir(parents=True)
    cache_file = directory / "cached.bin"
    cache_file.write_bytes(b"synthetic encrypted cache bytes")
    inventory = store.inventory_video("video-a")
    assert inventory[0]["sha256"] == hashlib.sha256(cache_file.read_bytes()).hexdigest()
    invalidated = store.invalidate("video-a")
    assert invalidated["deleted_files"] == 1
    assert not directory.exists()
