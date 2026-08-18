from pathlib import Path

from scripts.q8k_e2e_experiment import checkpoint_pairs


def test_q8k_e2e_includes_layer_2_input_checkpoint():
    pairs = checkpoint_pairs(Path("qx"), Path("oracle"))
    assert pairs["layer-2-input"] == (
        Path("qx/step-0-layer-2-input.f32"),
        Path("oracle/layer-2.f32"),
    )
