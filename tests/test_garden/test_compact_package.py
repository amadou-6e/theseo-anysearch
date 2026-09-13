import json

import pytest
import torch

from theseo_anysearch.garden import compact_package as package
from theseo_anysearch.garden.compact_structured import StructuredReadout


def fixture_package(tmp_path):
    encoder = package.empty_encoder(); head = StructuredReadout(9, 1)
    path = tmp_path / "seed401.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, path)
    record = {"file": path.name, "sha256": package.file_sha(path),
              "encoder_state_sha256": package.encoder_state_sha256(encoder),
              "head_state_sha256": package.encoder_state_sha256(head)}
    manifest = {"contract": package.CONTRACT, "artifacts": {str(s): dict(record) for s in (401, 402, 403)}}
    write_manifest(tmp_path, manifest)
    return encoder, head, manifest


def write_manifest(path, manifest):
    manifest = dict(manifest); manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = package.payload_sha256(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_roundtrip_and_trainable_head_without_encoder_grad(tmp_path):
    original, _, _ = fixture_package(tmp_path)
    encoder, head, manifest = package.load_compact_package(tmp_path)
    assert manifest["contract"]["dimension"] == 729
    encoder.train(); assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    occupancy = torch.zeros(2, 33, 33, 33); mask = torch.zeros_like(occupancy, dtype=torch.bool)
    code = encoder(occupancy, mask)
    assert code.shape == (2, 729) and not code.requires_grad and code.dtype == torch.float32
    torch.testing.assert_close(code, original(occupancy, mask))
    before = package.encoder_state_sha256(encoder)
    head(code, torch.zeros(2, 2, dtype=torch.long)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert package.encoder_state_sha256(encoder) == before


def test_mask_isolation():
    encoder = package.empty_encoder(); unknown = torch.ones(1, 33, 33, 33, dtype=torch.bool)
    torch.testing.assert_close(encoder(torch.zeros_like(unknown), unknown), encoder(torch.ones_like(unknown), unknown), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["shape", "mask", "binary", "empty"])
def test_reject_invalid_input(kind):
    encoder = package.empty_encoder(); occupancy = torch.zeros(1, 33, 33, 33)
    mask = torch.zeros_like(occupancy, dtype=torch.bool)
    if kind == "shape": occupancy = occupancy[:, :17]
    if kind == "mask": mask = mask.float()
    if kind == "binary": occupancy[0, 0, 0, 0] = .5
    if kind == "empty": occupancy, mask = occupancy[:0], mask[:0]
    with pytest.raises(ValueError): encoder(occupancy, mask)


@pytest.mark.parametrize("seed", [True, 401.0, 999])
def test_reject_bad_seed(tmp_path, seed):
    fixture_package(tmp_path)
    with pytest.raises(ValueError): package.load_compact_package(tmp_path, seed=seed)


def test_manifest_and_weight_tampering(tmp_path):
    _, _, manifest = fixture_package(tmp_path)
    (tmp_path / "seed401.pt").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="weight hash"): package.load_compact_package(tmp_path)
    _, _, manifest = fixture_package(tmp_path)
    manifest["artifacts"]["401"]["file"] = "../outside.pt"
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unsafe"): package.load_compact_package(tmp_path)
    _, _, manifest = fixture_package(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text()); saved["contract"]["dimension"] = 192
    (tmp_path / "manifest.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="manifest"): package.load_compact_package(tmp_path)


def test_statistics_validated():
    encoder = package.empty_encoder(); mean = torch.zeros(729); mean[1] = 1
    with pytest.raises(ValueError, match="shared"):
        package.FrozenCompactEncoder(encoder.backbone, encoder.aggregation, mean, torch.ones(729))
