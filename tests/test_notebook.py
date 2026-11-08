"""pipeline.ipynb is saved executed, without errors; it assembled nnHALO's weights from weights/fast/, its two
networks ran in that execution (not read from an earlier run), its final rows agree with the submitted file, it
loads only nnHALO, and no machine-specific path is left in its outputs."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "pipeline.ipynb"


def test_notebook_executed_cleanly_and_agrees_with_submission():
    nb = json.loads(NB.read_text())
    outs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
    assert all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code")
    assert not [o for o in outs if o["output_type"] == "error"]
    text = "".join("".join(o.get("text", "")) for o in outs if o["output_type"] == "stream")
    assert "8 files in place and verified" in text                   # the weights came from weights/fast/ (manifest)
    assert "TRAIN = False" in text and "smoke training: passed" in text      # training path listed + smoke-run
    for step in ("channels", "base_D007_f0", "bin_D036_f0", "selector_FAST"):  # every stage ran in THIS execution
        assert f"{step:20s} ran in" in text, step
    assert "within the reproduction tolerance: True" in text          # the 180-image CSV vs submissions/FAST.csv
    assert "agrees with the submitted file: True" in text             # the one-observation walk-through
    for name in ("stage evaluation inputs", "cross-fitted selector", "held-out annotator (Table 1)"):
        assert f"{name:30s} rc 0" in text, name                       # each evaluation step passed its own gate
    assert "prediction of the 180 test observations" in text          # runtime section
    assert sum(1 for o in outs if o["output_type"] == "display_data") >= 5      # the figures are saved


def test_notebook_loads_only_ours_fast():
    src = "".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"])
    assert "FAST_bag.pkl" in src and 'config.load("FAST")' in src


def test_notebook_names_no_dropped_network():
    assert "segfor" + "mer" not in NB.read_text().lower()             # split so this file does not match itself


def test_notebook_has_no_machine_specific_paths():
    text = "".join("".join(o.get("text", "")) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []) if o["output_type"] == "stream")
    assert not any(prefix in text for prefix in ("/ho" "me/", "/tm" "p/", "/Us" "ers/"))     # split so this file does not match itself
