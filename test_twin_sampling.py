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


class ExponentialTwin(RecordingTwin):
    """dx/dt=x has an analytic solution and exposes integration error."""

    def velocity(self, x, t, cond):
        return x


class SurgeryShiftTwin(RecordingTwin):
    def velocity(self, x, t, cond):
        return cond[:, :1].expand_as(x)


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


def test_heun_is_more_accurate_than_euler_for_same_latent_draws():
    arrays = _arrays(1)
    cfg = tw.TwinConfig(n_samples_per_patient=1, sample_steps=10)
    noise = np.ones((1, 1, 1), dtype=np.float32)
    euler = tw.sample_trajectories(
        ExponentialTwin(), arrays, cfg, torch.device("cpu"), 1,
        initial_noise=noise, solver="euler",
    )[0, 0, 0]
    heun = tw.sample_trajectories(
        ExponentialTwin(), arrays, cfg, torch.device("cpu"), 1,
        initial_noise=noise, solver="heun",
    )[0, 0, 0]

    assert abs(heun - np.e) < abs(euler - np.e)


def test_common_noise_is_reused_for_paired_surgery_contrast():
    arrays = _arrays(2)
    cfg = tw.TwinConfig(n_samples_per_patient=3, sample_steps=4)
    noise = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    sleeve = {**arrays, "surgery_idx": np.zeros(2, dtype=np.int64)}
    rygb = {**arrays, "surgery_idx": np.ones(2, dtype=np.int64)}
    y0 = tw.sample_trajectories(
        SurgeryShiftTwin(), sleeve, cfg, torch.device("cpu"), 1,
        initial_noise=noise,
    )
    y1 = tw.sample_trajectories(
        SurgeryShiftTwin(), rygb, cfg, torch.device("cpu"), 1,
        initial_noise=noise,
    )

    np.testing.assert_allclose(y1 - y0, 1.0, atol=1e-6)
