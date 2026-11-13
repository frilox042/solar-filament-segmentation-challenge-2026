"""Slot configuration: `configs/<slot>.json` -> one immutable `SlotConfig` passed explicitly to every stage.

Nothing in the pipeline reads a module-level setting. Every path in the config is RELATIVE TO THE RUN ROOT
(`filament_paths.WORK`) and is resolved here to an absolute path, once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from filament_seg import filament_paths as fp

# Inputs of the CPU stage, relative to WORK (see docs/work_layout.md).
RAW_TRAIN = "nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/imagesTr"     # channel 0 = raw H-alpha, img_XXXX_0000.png
RAW_TEST = "nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/imagesTs"      # filamentTs_XXXX_0000.png
TEST_CASE_MAP = "nnunet_env/nnUNet_raw/Dataset007_FilamentVessel/case_mapping_full.json"   # ["test"]: case -> image
TRAIN_CASE_MAP = "nnunet_env/nnUNet_raw/Dataset005_FilamentFull/case_mapping_full.json"   # ["train"]: case -> image


@dataclass(frozen=True)
class Member:
    """One selector member (one mask per image, two selector columns each: mb and cov).

    kind "file":           a binary PNG per image, written by a network: out of fold `<oof_dir>/<case>.png`, on the
                           test images `<test_dir>/<name>.png` with name = the case id (naming "case") or the
                           competition image id (naming "image").
    kind "base_threshold": no file: the base probability map itself, `pr >= t`, computed in process from the same
                           map the candidates are thresholded from (`selector_features.member_masks`)."""
    name: str
    kind: str                         # "file" | "base_threshold"
    oof_dir: Path | None = None
    test_dir: Path | None = None
    naming: str | None = None         # "case" | "image" (kind "file")
    t: float | None = None            # kind "base_threshold"

    @property
    def is_file(self) -> bool:
        return self.kind == "file"

    def test_file(self, case: str, image_id: str) -> str:
        """The member's test PNG of one image (kind "file" only)."""
        return f"{self.test_dir}/{case if self.naming == 'case' else image_id}.png"


def _member(m: dict, W: Path) -> Member:
    if m["kind"] == "file":
        assert m["naming"] in ("case", "image"), m
        return Member(m["name"], "file", oof_dir=W / m["oof_dir"], test_dir=W / m["test_dir"], naming=m["naming"])
    if m["kind"] == "base_threshold":
        return Member(m["name"], "base_threshold", t=float(m["t"]))
    raise ValueError(f"unknown member kind {m['kind']!r}")


@dataclass(frozen=True)
class SlotConfig:
    """Everything the CPU stage of one slot needs, resolved to absolute paths."""
    slot: str
    expected_md5: str
    expected_two_cut: float
    base_mask: str                     # "t97": p >= t
    core_t: float                      # the core threshold t (0.97)
    oof_prob_dir: Path                 # 707 OOF uint16 probability maps of the base network
    members: tuple[Member, ...]        # the selector members, in column order
    test_prob_dir: Path                # the base network's nnU-Net npz outputs on the test images
    fit_workers: int
    apply_workers: int
    hgb_threads: int
    member_join_check: bool
    min_area_csv: int
    raw_train: Path = field(default_factory=lambda: fp.WORK / RAW_TRAIN)
    raw_test: Path = field(default_factory=lambda: fp.WORK / RAW_TEST)
    test_case_map: Path = field(default_factory=lambda: fp.WORK / TEST_CASE_MAP)
    train_case_map: Path = field(default_factory=lambda: fp.WORK / TRAIN_CASE_MAP)
    coco: Path = field(default_factory=lambda: fp.KAGGLE_COCO)
    selector_tables: Path = field(default_factory=lambda: fp.WORK / "analysis" / "selector_tables.npz")
    scratch: Path = field(default_factory=lambda: fp.WORK / "selector_scratch")
    reference_csv: Path = field(default_factory=lambda: fp.REFERENCE_SUBMISSIONS / "missing.csv")
    tolerance_rule: dict = field(default_factory=dict)    # reproduction_tolerance.rule (submission.compare)

    @property
    def oof_member_dirs(self) -> tuple[Path, ...]:
        """The OOF directories of the file members (the inputs a cache identity and a staging tool must cover)."""
        return tuple(m.oof_dir for m in self.members if m.is_file)


def test_image_ids(cfg: SlotConfig | None = None) -> list:
    """The 180 competition test image ids (file stems), sorted, from the shipped `configs/case_mapping_full.json`."""
    import os
    cm = json.loads((fp.CONFIGS / "case_mapping_full.json").read_text())["test"]
    return sorted(os.path.splitext(v if isinstance(v, str) else v["image"])[0] for v in cm.values())


def train_case_map() -> Path:
    """The training case mapping (case id -> competition image), under the run root; the same for every config."""
    return fp.WORK / TRAIN_CASE_MAP


def load_json(slot: str) -> dict:
    """The parsed `configs/<slot>.json`."""
    return json.loads((fp.CONFIGS / f"{slot}.json").read_text())


def load(slot: str = "nnHALO") -> SlotConfig:
    """Build the config of `configs/<slot>.json` (the only one shipped is nnHALO)."""
    j = load_json(slot)
    c = j["cpu_stage"]
    W = fp.WORK
    tp = c["test_prob"]
    assert tp["kind"] == "npz" and c["base_mask"] == "t97", (tp["kind"], c["base_mask"])
    core_t, prob_dir = float(c["core_t"]), W / tp["dir"]
    return SlotConfig(
        slot=slot, expected_md5=j["expected_md5"], expected_two_cut=float(j["selector"]["expected_two_cut"]),
        base_mask=c["base_mask"], core_t=core_t, oof_prob_dir=W / c["oof_prob_dir"],
        members=tuple(_member(m, W) for m in c["members"]), test_prob_dir=prob_dir,
        fit_workers=int(c["workers"]["fit"]), apply_workers=int(c["workers"]["apply"]),
        hgb_threads=int(c["hgb_threads"]), member_join_check=bool(c["member_join_check"]),
        min_area_csv=int(c["min_area_csv"]), scratch=W / "selector_scratch" / slot,
        reference_csv=fp.REFERENCE_SUBMISSIONS / f"{slot}.csv",
        tolerance_rule=_rule(j["reproduction_tolerance"]))


def _rule(tol: dict) -> dict:
    """`reproduction_tolerance.rule`, checked to name exactly the four limits of `submission.compare`."""
    from filament_seg.submission.compare import RULE_KEYS
    rule = tol["rule"]
    assert set(rule) == set(RULE_KEYS), sorted(rule)
    return {k: (float if k.startswith("min_") else int)(rule[k]) for k in RULE_KEYS}
