import importlib.util
from pathlib import Path

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.sim2real_image_dataset import Sim2RealImageMultiDataset


REPOS = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "select_forced_trajectory_indices",
    REPOS / "UWLab/scripts_v2/tools/select_forced_trajectory_indices.py",
)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def dataset(tmp_path, lengths=(24, 7), horizon=4, val_ratio=0.0, shards=1):
    shape = {"obs": {"front_rgb": {"shape": [3, 2, 2], "type": "rgb"}},
             "action": {"shape": [7]}}
    for shard in range(shards):
        root = zarr.open_group(str(tmp_path / f"rgb_{shard:05d}.zarr"), mode="w")
        ends = np.cumsum(lengths)
        count = int(ends[-1])
        root.create_dataset("meta/episode_ends", data=ends)
        root.create_dataset("data/actions", data=np.repeat(np.arange(count)[:, None], 7, axis=1).astype(np.float32))
        root.create_dataset("data/obs/front_rgb", data=np.zeros((count, 2, 2, 3), dtype=np.uint8))
    return Sim2RealImageMultiDataset(
        dataset_dir=str(tmp_path), shape_meta=shape, horizon=horizon,
        pad_before=3, pad_after=0, n_obs_steps=4, val_ratio=val_ratio,
        use_cache=False,
    )


def test_all_targets_include_the_last_recorded_action(tmp_path):
    data = dataset(tmp_path)
    result = SELECTOR.select_windows(data, 2)
    assert [r["windows"] for r in result["selected_trajectories"]] == [24, 7]
    for window in result["window_provenance"]:
        sample = data[window["global_index"]]
        assert sample["action"][3, 0].item() == window["target_frame"]
    assert [r["target_frame"] for r in result["window_provenance"]] == list(range(31))


def test_legacy_horizon_is_rejected_as_incomplete(tmp_path):
    data = dataset(tmp_path, lengths=(24,), horizon=8)
    with pytest.raises(RuntimeError, match="missing 4 action targets"):
        SELECTOR.select_windows(data, 1)
    result = SELECTOR.select_windows(data, 1, require_complete=False)
    assert result["selected_trajectories"][0]["missing_action_targets"] == [20, 21, 22, 23]


def test_selection_uses_validation_mask_and_global_shard_offsets(tmp_path):
    data = dataset(tmp_path, lengths=(24, 7), val_ratio=0.5, shards=2)
    result = SELECTOR.select_windows(data, 2)
    assert {r["dataset_index"] for r in result["selected_trajectories"]} == {0, 1}
    for window in result["window_provenance"]:
        child = data.datasets[window["dataset_index"]]
        assert child.train_mask[window["episode"]]
        sample = data[window["global_index"]]
        assert sample["action"][3, 0].item() == window["target_frame"]


def test_production_config_covers_full_single_action_history():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = REPOS / "diffusion_policy/diffusion_policy/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train_mlp_sim2real_image_with_aux_loss_workspace")
    assert cfg.horizon == cfg.n_obs_steps == 4
    assert cfg.n_action_steps == 1
    assert cfg.task.dataset.pad_before == cfg.n_obs_steps - 1


def test_forced_indices_are_enumerated_by_position(tmp_path):
    data = dataset(tmp_path)
    data.forced_indices = (2, 17, 25)
    targets = [data[index]["action"][3, 0].item() for index in range(len(data.forced_indices))]
    assert targets == [2, 17, 25]
    # Treating the original global indices as request indices remaps them twice.
    assert [data[index]["action"][3, 0].item() for index in data.forced_indices] != targets
