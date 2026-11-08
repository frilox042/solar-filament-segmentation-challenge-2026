"""Repository hygiene: no absolute developer paths in tracked text; the settings file resolves under the repo."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (".py", ".md", ".json", ".txt", ".toml", ".sh", ".cff")


def _repo_files():
    """The repository's own files: git-tracked when this is a clone, else everything except the environments and run
    root the README creates inside it (.venv-*, work/)."""
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return [ROOT / f for f in r.stdout.splitlines()]
    return [p for p in ROOT.rglob("*") if not any(x.startswith(".venv") or x in ("work", ".git") for x in
                                                  p.relative_to(ROOT).parts)]


def test_no_absolute_home_paths():
    bad = []
    for p in _repo_files():
        if p.is_file() and p.suffix in TEXT and "__pycache__" not in p.parts and "work" not in p.parts:
            if "/ho" "me/" in p.read_text(errors="replace"):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_settings_defaults_are_repo_relative():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FILAMENT_", "nnUNet_"))}
    code = ("import sys; sys.path.insert(0, %r); from filament_seg import filament_paths as fp; "
            "print(fp.REPO, fp.WORK, fp.NNUNET_RAW)" % str(ROOT / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.split()
    assert out[0] == str(ROOT) and all(o.startswith(str(ROOT)) for o in out)


def test_magfilo_is_never_readable():
    sys.path.insert(0, str(ROOT / "src"))
    from filament_seg import filament_paths as fp
    try:
        fp.magfilo_json()
    except RuntimeError:
        return
    raise AssertionError("magfilo_json() must refuse")
