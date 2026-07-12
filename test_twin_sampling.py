"""Regression tests for memory-bounded digital-twin trajectory sampling."""

from __future__ import annotations

import numpy as np
import torch

import train_flow_matching_twin as tw


class RecordingTwin(torch.nn.Module):
    """Small deterministic twin that records the largest inference batch."""

    def __init__(self) -> None:
        super().__init__()
        self.max_batch = 0

    def encode(self, surgery, patient_features, event):
        self.max_batch = max(self.max_batch, int(patient_features.shape[0]))
        return torch.cat(
            [surgery.float().unsqueeze(1), event.float().unsqueeze(1), patient_features[:, :1]],
            dim=1,
        )

    def velocity(self, x, t, cond):
        self.max_batch = max(self.max_batch, int(x.shape[0]))
        return torch.zeros_like(x)


def _arrays(n_patients: int) -> dict[str, np.ndarray]:
    return {
        "patient_features": np.arange(n_patients * 2, dtype=np.float32).reshape(n_patients, 2),
        "surgery_idx": np.arange(n_patients, dtype=np.int64) % 2,
        "y_mace": (np.arange(n_patients, dtype=np.int64) + 1) % 2,
    }


def test_sample_trajectories_bounds_each_model_batch():
    model = RecordingTwin()
    cfg = tw.TwinConfig(n_samples_per_patient=11, sample_steps=2)
    cfg.sample_batch_size = 64

    torch.manual_seed(7)
    result = tw.sample_trajectories(
        model, _arrays(41), cfg, torch.device("cpu"), x_cont_dim=3
    )

    assert result.shape == (41, 11, 3)
    assert model.max_batch <= 64


def test_batched_sampling_is_repeatable_and_restores_model_mode():
    unbatched_model = RecordingTwin()
    unbatched_model.eval()
    batched_model = RecordingTwin()
    batched_model.eval()
    unbatched_cfg = tw.TwinConfig(
        n_samples_per_patient=7, sample_steps=2, sample_batch_size=1000
    )
    batched_cfg = tw.TwinConfig(
        n_samples_per_patient=7, sample_steps=2, sample_batch_size=20
    )
    arrays = _arrays(13)

    torch.manual_seed(19)
    unbatched = tw.sample_trajectories(
        unbatched_model, arrays, unbatched_cfg, torch.device("cpu"), 2
    )
    torch.manual_seed(19)
    batched = tw.sample_trajectories(
        batched_model, arrays, batched_cfg, torch.device("cpu"), 2
    )

    np.testing.assert_array_equal(unbatched, batched)
    assert unbatched_model.training is False
    assert batched_model.training is False
