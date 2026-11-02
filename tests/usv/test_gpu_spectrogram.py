"""The inference spectrogram on a CUDA GPU must be the CPU one.

The model was trained on scipy spectrograms, so the GPU version reproduces
scipy step for step (per-frame mean removal, periodic Hann, density scaling,
x2 one-sided PSD) and is used only after a once-per-process check that it
lands within one grey level on at most 0.01% of pixels. On Apple Silicon (MPS)
and the CPU nothing changes.

Runs under pytest, or directly.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector import mad_dataset as D

SR = 250_000
ARGS = (SR, 512, 384, 1024, -100.0, -20.0)


def signal(seconds=1.5, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = rng.normal(0, 0.01, t.size)
    x += 0.05 * np.sin(2 * np.pi * (35_000 + 15_000 * t) * t)  # a sweep
    x += 0.3                                                   # DC offset
    return x.astype(np.float32)


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return "cuda"


def test_the_gpu_image_is_the_cpu_image():
    dev = _cuda()
    x = signal()
    cpu = D.compute_full_spec_image(x, *ARGS, as_uint8=True)
    gpu = D.compute_full_spec_image_torch(x, *ARGS, device=dev)
    d = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    assert cpu.shape == gpu.shape and gpu.dtype == np.uint8
    assert d.max() <= 1
    assert np.count_nonzero(d) <= 1e-4 * d.size


def test_chunking_changes_nothing_beyond_rounding():
    """cuFFT's arithmetic varies with batch size, so a pixel can land one
    level apart -- the same tolerance the CPU comparison allows."""
    dev = _cuda()
    x = signal()
    a = D.compute_full_spec_image_torch(x, *ARGS, device=dev)
    b = D.compute_full_spec_image_torch(x, *ARGS, device=dev,
                                        chunk_frames=777)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert d.max() <= 1 and np.count_nonzero(d) <= 1e-4 * d.size


def test_too_short_is_empty_like_the_cpu():
    dev = _cuda()
    out = D.compute_full_spec_image_torch(np.zeros(100, np.float32), *ARGS,
                                          device=dev)
    assert out.shape == (513, 0)


def test_the_check_passes_on_cuda():
    dev = _cuda()
    D._TORCH_SPEC_OK.pop(dev, None)
    assert D.torch_spectrogram_agrees(dev)


def test_a_backend_that_disagrees_is_not_used(monkeypatch):
    """E.g. a missing or different FFT on some backend: the CPU is used."""
    monkeypatch.setattr(D, "_TORCH_SPEC_OK", {})
    monkeypatch.setattr(D, "compute_full_spec_image_torch",
                        lambda x, *a, **k: np.clip(
                            D.compute_full_spec_image(x, *a[:6], as_uint8=True)
                            .astype(np.int16) + 5, 0, 255).astype(np.uint8))
    assert not D.torch_spectrogram_agrees("cuda")
    img, backend = D.spec_image_for_device(signal(0.2), *ARGS, device="cuda")
    assert backend == "cpu"


def test_a_gpu_failure_falls_back_to_the_cpu(monkeypatch):
    monkeypatch.setattr(D, "_TORCH_SPEC_OK", {"cuda": True})

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(D, "compute_full_spec_image_torch", boom)
    x = signal(0.2)
    img, backend = D.spec_image_for_device(x, *ARGS, device="cuda")
    assert backend == "cpu"
    assert np.array_equal(img, D.compute_full_spec_image(x, *ARGS,
                                                         as_uint8=True))


@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_only_cuda_takes_the_gpu_path(monkeypatch, device):
    def nope(*a, **k):
        raise AssertionError(f"GPU spectrogram used on {device}")
    monkeypatch.setattr(D, "compute_full_spec_image_torch", nope)
    monkeypatch.setattr(D, "torch_spectrogram_agrees", nope)
    _img, backend = D.spec_image_for_device(signal(0.2), *ARGS, device=device)
    assert backend == "cpu"


def test_inference_reports_which_backend_built_it(monkeypatch):
    from fnt.usv.usv_detector import mad_inference as MI
    monkeypatch.setattr(MI, "load_audio",
                        lambda p, *a, **k: (signal(0.2), SR))
    cfg = MI.MADInferenceConfig(model_path="m.pt")
    assert MI.prepare_file("x.wav", cfg, {}, "cpu")["spec_backend"] == "cpu"
    cfg_off = MI.MADInferenceConfig(model_path="m.pt", gpu_spectrogram=False)
    assert MI.prepare_file("x.wav", cfg_off, {}, "cuda")["spec_backend"] == "cpu"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
