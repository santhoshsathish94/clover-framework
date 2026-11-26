import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

from step1 import EXPECTED_SHA256, closed_boundary, sha256
from step2 import numeric_summary


TENSOR = "model.language_model.layers.0.conv.conv.weight"
FFT_SIZE = 2048
DC_CANCELLATION_CUTOFF = 0.10


def summary(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {"count": len(finite), **numeric_summary(finite)} if len(finite) else {"count": 0, "quantiles": []}


def filter_statistics(coefficients, fft_size=FFT_SIZE):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or not np.isfinite(coefficients).all():
        raise ValueError("Expected finite filters with taps along axis one")
    if fft_size < 2 * coefficients.shape[1] or fft_size % 2:
        raise ValueError("An even frequency grid at least twice the filter length is required")
    tap_count = coefficients.shape[1]
    l1 = np.abs(coefficients).sum(1)
    l2 = np.linalg.norm(coefficients, axis=1)
    active = l2 > 0
    dc = coefficients.sum(1)
    alternating = coefficients @ ((-1.0) ** np.arange(tap_count))
    response = np.fft.rfft(coefficients, n=fft_size, axis=1)
    magnitude = np.abs(response)
    frequencies = np.fft.rfftfreq(fft_size)
    peak_indices = magnitude.argmax(1)
    maximum = magnitude.max(1)
    minimum = magnitude.min(1)
    relative_ripple = np.zeros(len(coefficients))
    relative_ripple[active] = (maximum[active] - minimum[active]) / l2[active]
    flat = active & (relative_ripple <= 1e-6)
    nonflat = active & ~flat
    dc_to_l1 = np.zeros(len(coefficients))
    dc_to_l1[active] = np.abs(dc[active]) / l1[active]
    dc_to_rms = np.zeros(len(coefficients))
    dc_to_rms[active] = np.abs(dc[active]) / l2[active]
    alternating_to_rms = np.zeros(len(coefficients))
    alternating_to_rms[active] = np.abs(alternating[active]) / l2[active]
    dc_to_peak = np.zeros(len(coefficients))
    dc_to_peak[active] = np.abs(dc[active]) / maximum[active]
    tap_energy = coefficients.square() if hasattr(coefficients, "square") else coefficients ** 2
    total_energy = tap_energy.sum()
    shares = np.zeros_like(coefficients)
    shares[active] = np.abs(coefficients[active]) / l1[active, None]
    squared = magnitude ** 2
    spectral_energy = (squared[:, 0] + squared[:, -1] + 2 * squared[:, 1:-1].sum(1)) / fft_size
    parseval_error = float(np.max(np.abs(spectral_energy - l2 ** 2)))
    reverse_magnitude = np.abs(np.fft.rfft(coefficients[:, ::-1], n=fft_size, axis=1))
    reversal_error = float(np.max(np.abs(magnitude - reverse_magnitude)))
    metrics = {
        "filters": len(coefficients), "tap_count": tap_count,
        "zero_filters": int((~active).sum()), "flat_magnitude_filters": int(flat.sum()),
        "nonflat_filters": int(nonflat.sum()),
        "positive_dc": int((dc > 0).sum()), "negative_dc": int((dc < 0).sum()), "exact_zero_dc": int((dc == 0).sum()),
        "dc_signed_gain": summary(dc), "dc_absolute_gain": summary(np.abs(dc)),
        "coefficient_l1": summary(l1), "coefficient_l2": summary(l2),
        "dc_to_l1_active": summary(dc_to_l1[active]),
        "dc_to_frequency_rms_active": summary(dc_to_rms[active]),
        "alternating_to_frequency_rms_active": summary(alternating_to_rms[active]),
        "dc_to_grid_peak_active": summary(dc_to_peak[active]),
        "strong_dc_cancellation_filters": int(np.count_nonzero(active & (dc_to_l1 <= DC_CANCELLATION_CUTOFF))),
        "dc_below_frequency_rms": int(np.count_nonzero(active & (dc_to_rms < 1))),
        "alternating_gain_exceeds_dc": int(np.count_nonzero(active & (np.abs(alternating) > np.abs(dc)))),
        "dc_at_least_99pct_of_grid_peak_nonflat": int(np.count_nonzero(nonflat & (dc_to_peak >= 0.99))),
        "grid_peak_frequency_nonflat": summary(frequencies[peak_indices[nonflat]]),
        "grid_peak_bands_nonflat": {
            "0_to_0.125": int(np.count_nonzero(nonflat & (frequencies[peak_indices] < 0.125))),
            "0.125_to_0.25": int(np.count_nonzero(nonflat & (frequencies[peak_indices] >= 0.125) & (frequencies[peak_indices] < 0.25))),
            "0.25_to_0.375": int(np.count_nonzero(nonflat & (frequencies[peak_indices] >= 0.25) & (frequencies[peak_indices] < 0.375))),
            "0.375_to_0.5_inclusive": int(np.count_nonzero(nonflat & (frequencies[peak_indices] >= 0.375))),
        },
        "fraction_bank_squared_weight_norm_by_stored_tap": (tap_energy.sum(0) / total_energy).tolist() if total_energy else [0.0] * tap_count,
        "mean_absolute_share_by_stored_tap_active": shares[active].mean(0).tolist() if active.any() else [0.0] * tap_count,
        "dominant_absolute_tap_counts_active": np.bincount(np.abs(coefficients[active]).argmax(1), minlength=tap_count).tolist(),
        "parseval_max_absolute_error": parseval_error,
        "tap_reversal_max_magnitude_error": reversal_error,
    }
    arrays = {
        "coefficients": coefficients, "dc_gain": dc, "alternating_gain": alternating,
        "coefficient_l1": l1, "coefficient_l2": l2, "dc_to_l1": dc_to_l1,
        "dc_to_frequency_rms": dc_to_rms, "alternating_to_frequency_rms": alternating_to_rms,
        "dc_to_grid_peak": dc_to_peak, "peak_frequency": frequencies[peak_indices],
        "flat_magnitude_mask": flat, "active_mask": active,
        "frequencies": frequencies, "frequency_magnitude": magnitude,
    }
    return metrics, arrays


def run(root):
    root = Path(root)
    boundary = closed_boundary(root)
    checkpoint = root / "original" / "model.safetensors"
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise ValueError("Original checkpoint identity mismatch")
    output = root / "weights-only" / "step4"
    if output.exists():
        raise FileExistsError("Preserve earlier step results")
    started = time.monotonic()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        tensor = source.get_tensor(TENSOR)
    if tensor.shape != (2048, 1, 7) or tensor.dtype != torch.bfloat16:
        raise ValueError("Unexpected convolution tensor header")
    tensor_hash = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    coefficients = tensor[:, 0, :].double().numpy()
    metrics, arrays = filter_statistics(coefficients)
    selected = set(np.argsort(arrays["dc_to_l1"])[:4].tolist())
    selected.update(np.argsort(arrays["dc_to_frequency_rms"])[-4:].tolist())
    selected.update(np.argsort(arrays["coefficient_l2"])[-2:].tolist())
    examples = [{"filter_index": index, "coefficients_in_stored_order": coefficients[index].tolist(),
                 "dc_signed_gain": float(arrays["dc_gain"][index]),
                 "alternating_signed_gain": float(arrays["alternating_gain"][index]),
                 "dc_to_l1": float(arrays["dc_to_l1"][index]),
                 "grid_peak_frequency": float(arrays["peak_frequency"][index])}
                for index in sorted(selected)]
    output.mkdir()
    np.savez_compressed(output / "arrays.npz", **arrays)
    report = {
        "evidence_mode": "stored seven-tap coefficients only; analytical transfer function, no sequence inputs or model execution",
        "boundary": boundary, "tensor": TENSOR, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "tensor_sha256": tensor_hash, "parameter_count": tensor.numel(),
        "method": {"dc": "sum of coefficients", "alternating": "sum of coefficients times alternating +/-1 signs",
                   "frequency": "absolute rFFT of coefficient vector, zero-padded to2048",
                   "frequency_units": "cycles per discrete step, 0..0.5; no sample-rate or token identity assumed",
                   "frequency_rms_gain": "coefficient L2 norm by Parseval",
                   "strong_dc_cancellation_cutoff": DC_CANCELLATION_CUTOFF,
                   "flat_ripple_cutoff": 1e-6},
        "metrics": metrics, "numeric_examples_not_defect_candidates": examples,
        "elapsed_s": time.monotonic() - started,
        "checkpoint_sha256_before": EXPECTED_SHA256, "checkpoint_sha256_after": sha256(checkpoint),
        "arrays_sha256": sha256(output / "arrays.npz"),
        "limits": ["Tap time orientation is not inferred; reversing real taps preserves magnitude response",
                   "Per-filter response does not identify which preceding projection block is its input",
                   "Gates, nonlinearities, normalization, boundary padding, residuals and actual sequences were not evaluated",
                   "No claim that the embedding-family component is constant in real use, or that repetition is removed",
                   "Numeric thresholds are descriptive, not criteria for defects or weight edits"],
    }
    if report["checkpoint_sha256_after"] != EXPECTED_SHA256:
        raise ValueError("Original checkpoint changed")
    with (output / "result.json").open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print("WEIGHTS_ONLY_STEP4_COMPLETE " + json.dumps({
        "filters": metrics["filters"], "strong_dc_cancellation": metrics["strong_dc_cancellation_filters"],
        "dc_below_frequency_rms": metrics["dc_below_frequency_rms"],
        "alternating_exceeds_dc": metrics["alternating_gain_exceeds_dc"],
        "peak_bands_nonflat": metrics["grid_peak_bands_nonflat"],
        "elapsed_s": report["elapsed_s"], "original_unchanged": True,
    }), flush=True)


class FilterTests(unittest.TestCase):
    def test_constant_preserving_average(self):
        coefficients = np.ones((1, 7)) / 7
        metrics, arrays = filter_statistics(coefficients)
        self.assertAlmostEqual(arrays["dc_gain"][0], 1)
        self.assertAlmostEqual(arrays["alternating_gain"][0], 1 / 7)
        self.assertEqual(arrays["peak_frequency"][0], 0)
        self.assertEqual(metrics["strong_dc_cancellation_filters"], 0)

    def test_differencing_rejects_constant(self):
        coefficients = np.array([[1.0, -1, 0, 0, 0, 0, 0]])
        metrics, arrays = filter_statistics(coefficients)
        self.assertEqual(arrays["dc_gain"][0], 0)
        self.assertEqual(arrays["alternating_gain"][0], 2)
        self.assertEqual(arrays["peak_frequency"][0], 0.5)
        self.assertEqual(metrics["strong_dc_cancellation_filters"], 1)

    def test_flat_is_not_frequency_preference(self):
        metrics, arrays = filter_statistics(np.array([[0.0, 0, 0, 2, 0, 0, 0]]))
        np.testing.assert_allclose(arrays["frequency_magnitude"], 2, atol=1e-12)
        self.assertEqual(metrics["flat_magnitude_filters"], 1)
        self.assertEqual(sum(metrics["grid_peak_bands_nonflat"].values()), 0)

    def test_zero_filter_is_separate(self):
        metrics, _ = filter_statistics(np.zeros((1, 7)))
        self.assertEqual(metrics["zero_filters"], 1)
        self.assertEqual(metrics["flat_magnitude_filters"], 0)
        self.assertEqual(metrics["dc_to_l1_active"]["count"], 0)

    def test_parseval_and_reversal(self):
        coefficients = np.array([[0.25, -0.5, 0.125, 1.5, 0, -0.25, 0.5]])
        metrics, _ = filter_statistics(coefficients)
        self.assertLess(metrics["parseval_max_absolute_error"], 1e-12)
        self.assertLess(metrics["tap_reversal_max_magnitude_error"], 1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["step4.py"], verbosity=2)
    elif args.root:
        run(args.root)
    else:
        parser.error("--root or --self-test required")