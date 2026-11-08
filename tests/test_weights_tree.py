"""scripts/check_weights_tree.py: a weights tree passes only when every manifest model entry is present with the
manifest's sha256 and nothing unlisted is in it. Negative controls: one flipped byte, a missing file, an unlisted file.
The tree that scripts/assemble_weights.py writes from weights/fast/ passes; a tree given FILAMENT_WEIGHTS_TREE is
checked too (skipped otherwise)."""
import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_weights_tree as W  # noqa: E402

TR = "nnUNetTrainer_250epochs__nnUNetPlans__2d"
FILES = [f"nnUNet_results/Dataset007_FilamentVessel/{TR}/plans.json",
         f"nnUNet_results/Dataset007_FilamentVessel/{TR}/fold_0/checkpoint_final.pth",
         f"nnUNet_results/Dataset036_LimbDarkBarbClass/{TR}/fold_0/checkpoint_final.pth",
         f"nnUNet_results/Dataset036_LimbDarkBarbClass/{TR}/plans.json"]


def _tree(tmp_path):
    root, lines = tmp_path / "tree", []
    for i, rel in enumerate(FILES):
        data = f"payload {i} {rel}".encode() * 40
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "strip_report.json").write_text("[]")
    man = tmp_path / "MANIFEST.sha256"
    man.write_text("# fixture\n" + "\n".join(lines) + "\n" + "0" * 64 + "  selector/FAST_bag.pkl\n")
    return root, man


def _run(root, man, *args):
    return W.main([str(root), "--manifest", str(man), *args])


def test_matching_tree_passes(tmp_path):
    root, man = _tree(tmp_path)
    assert _run(root, man) == 0
    res = W.check(root, W.read_manifest(man))
    assert res["passed"] and sorted(res["verified"]) == sorted(FILES)        # strip_report.json ignored, bag not needed


def test_flipped_byte_fails(tmp_path):
    root, man = _tree(tmp_path)
    p = root / FILES[1]
    b = bytearray(p.read_bytes())
    b[5] ^= 1
    p.write_bytes(bytes(b))
    res = W.check(root, W.read_manifest(man))
    assert not res["passed"] and len(res["bad"]) == 1 and FILES[1] in res["bad"][0]
    assert _run(root, man) == 1


def test_missing_slot_file_fails(tmp_path):
    root, man = _tree(tmp_path)
    (root / FILES[2]).unlink()
    res = W.check(root, W.read_manifest(man))
    assert not res["passed"] and res["missing"] == [FILES[2]]


def test_unlisted_file_fails(tmp_path):
    root, man = _tree(tmp_path)
    extra = f"nnUNet_results/Dataset007_FilamentVessel/{TR}/fold_0/checkpoint_latest.pth"
    (root / extra).write_bytes(b"x")
    res = W.check(root, W.read_manifest(man))
    assert not res["passed"] and res["unlisted"] == [extra]


def test_assembled_repository_weights_pass(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from filament_seg import weights
    weights.assemble(tmp_path / "w")
    res = W.check(tmp_path / "w", W.read_manifest(ROOT / "weights" / "MANIFEST.sha256"))
    assert res["passed"] and res["n_verified"] == res["n_entries"] == 8


@pytest.mark.skipif(not os.environ.get("FILAMENT_WEIGHTS_TREE"), reason="set FILAMENT_WEIGHTS_TREE to the release tree")
def test_real_release_candidate_matches_manifest():
    root = Path(os.environ["FILAMENT_WEIGHTS_TREE"])
    res = W.check(root, W.read_manifest(ROOT / "weights" / "MANIFEST.sha256"))
    assert res["passed"], {k: res[k] for k in ("missing", "bad", "unlisted")}
