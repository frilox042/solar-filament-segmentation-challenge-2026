"""weights.assemble: parts are concatenated in order, a wrong hash fails, and the real stored weights verify.
The manifest lists exactly nnHALO's stored files plus its selector bag, and is the mechanical image of
SOURCE_CHECKPOINTS.sha256 (only the stripped nnU-Net checkpoints have new hashes)."""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament_seg import weights  # noqa: E402


def _store(parts_dir: Path, rel: str, chunks: list):
    for i, c in enumerate(chunks):
        p = parts_dir / f"{rel}.part{i:03d}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(c)


def test_parts_are_concatenated_in_order_and_small_files_copied(tmp_path):
    parts, dest = tmp_path / "parts", tmp_path / "dest"
    _store(parts, "a/ck.pth", [b"AAA", b"BBB", b"C"])
    (parts / "a").mkdir(exist_ok=True)
    (parts / "a" / "plans.json").write_bytes(b"{}")
    exp = {"a/ck.pth": hashlib.sha256(b"AAABBBC").hexdigest(), "a/plans.json": hashlib.sha256(b"{}").hexdigest()}
    assert sorted(weights.assemble(dest, parts, exp)) == ["a/ck.pth", "a/plans.json"]
    assert (dest / "a" / "ck.pth").read_bytes() == b"AAABBBC"
    assert weights.assemble(dest, parts, exp)          # a second call finds them in place and succeeds


def test_wrong_hash_fails_and_leaves_nothing(tmp_path):
    parts, dest = tmp_path / "parts", tmp_path / "dest"
    _store(parts, "a/ck.pth", [b"AAA", b"BBB"])
    with pytest.raises(ValueError, match="does not match"):
        weights.assemble(dest, parts, {"a/ck.pth": hashlib.sha256(b"BBBAAA").hexdigest()})   # parts in the wrong order
    assert not (dest / "a" / "ck.pth").exists() and not list(dest.rglob("*.tmp"))


def test_unlisted_file_is_refused(tmp_path):
    parts = tmp_path / "parts"
    _store(parts, "a/ck.pth", [b"x"])
    with pytest.raises(ValueError, match="not in MANIFEST"):
        weights.assemble(tmp_path / "dest", parts, {})


def test_real_stored_weights_verify_against_the_manifest(tmp_path):
    done = weights.assemble(tmp_path / "w", parts_dirs=(weights.PARTS_DIR,))      # nnHALO: assembled from parts
    assert sum(p.endswith("checkpoint_final.pth") for p in done) == 2
    assert all(p.startswith("nnUNet_results/") for p in done)             # the two nnU-Nets, nothing else
    assert max(p.stat().st_size for p in (ROOT / "weights" / "fast").rglob("*") if p.is_file()) < 100 * 1024 * 1024


def test_manifest_lists_exactly_the_stored_files():
    man = weights.manifest()
    assert set(man) == set(weights.stored_files()) | {"selector/FAST_bag.pkl"}
    assert sum(p.endswith("checkpoint_final.pth") for p in man) == 2
    assert not (ROOT / "weights" / "full").exists() and not (ROOT / ".gitattributes").exists()     # nothing stored outside plain git


def test_manifest_is_the_image_of_source_checkpoints():
    """Same paths after the training -> layout mapping, same hashes except the 2 nnU-Net checkpoint_final.pth, which
    are the stripped inference-only copies (weights/README.md)."""
    src = {}
    for line in (ROOT / "weights" / "SOURCE_CHECKPOINTS.sha256").read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            p = p.strip()
            assert p.startswith("nnunet_env/nnUNet_results/"), p
            src[p.replace("nnunet_env/nnUNet_results/", "nnUNet_results/", 1)] = h
    models = {k: v for k, v in weights.manifest().items() if not k.startswith("selector/")}
    assert set(models) == set(src)
    stripped = {k for k in models if k.startswith("nnUNet_results/") and k.endswith("/checkpoint_final.pth")}
    assert len(stripped) == 2
    assert all(models[k] != src[k] for k in stripped)                    # every nnU-Net checkpoint is the stripped one
    assert {k: v for k, v in models.items() if k not in stripped} == {k: v for k, v in src.items() if k not in stripped}
