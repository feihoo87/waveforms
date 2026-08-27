import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.special as special
from scipy.signal import butter, lfilter, lfiltic, tf2sos

import waveforms as wf


PUBLIC_NAMES = {
    "D", "Waveform", "WaveVStack", "chirp", "const", "cos", "cosh",
    "coshPulse", "cosPulse", "cut", "drag", "drag_sin", "drag_sinx",
    "exp", "function",
    "gaussian", "general_cosine", "get_time_resolution", "hanning", "interp", "mixing",
    "mollifier", "one", "poly", "registerBaseFunc", "registerDerivative",
    "samplingPoints", "set_time_resolution", "sign", "sin", "sinc", "sinh", "square", "step",
    "t", "wave_eval", "zero",
}


def test_public_api_and_basic_sampling():
    assert PUBLIC_NAMES <= set(dir(wf))
    x = np.linspace(-2.0, 2.0, 1001)
    assert np.allclose(wf.cos(1.3)(x), np.cos(1.3 * x))
    assert np.allclose(wf.sin(0.7)(x), np.sin(0.7 * x))
    assert np.allclose(wf.poly([1, -0.5, 0.25])(x),
                       1 - 0.5 * x + 0.25 * x**2)


def test_binary_roundtrip_is_zero_copy_for_bytes_input():
    wav = (wf.gaussian(0.8) >> 1.2) * wf.cos(2.3) + 0.25
    data = wav.to_bytes()
    restored = wf.Waveform.from_bytes(data)
    assert restored._core.to_bytes() is data
    assert restored == wav
    assert pickle.loads(pickle.dumps(wav)) == wav
    assert wf.one() >> 5 == wf.one()
    assert (wf.const(2.5) << 7).to_bytes() == wf.const(2.5).to_bytes()


def test_operations_simplify_derivative_and_clipping():
    x = np.linspace(-1.9, 1.9, 4097)
    wav = wf.cos(1) * wf.sin(2) * wf.cos(3, 4)
    expected = np.cos(x) * np.sin(2 * x) * np.cos(3 * x + 4)
    assert np.allclose(wav(x), expected)
    assert np.allclose(wav.simplify()(x), expected)

    width = 4.0
    std_sq2 = width / 3.3302184446307908
    order = 6
    expected_d = ((-1)**order / std_sq2**order
                  * special.eval_hermite(order, x / std_sq2)
                  * np.exp(-(x / std_sq2)**2))
    assert np.allclose(wf.gaussian(width, d=order)(x), expected_d)

    clipped = 2 * wf.cos(2.1) + 0.3 * wf.sin(0.7)
    clipped.min = -0.4
    clipped.max = 0.6
    assert np.allclose(clipped(x), np.clip(2 * np.cos(2.1 * x)
                                          + 0.3 * np.sin(0.7 * x),
                                          -0.4, 0.6))

    width = 1.3
    shift = 0.7
    frequency = 2.1
    phase = 0.4
    std_sq2 = width / 3.3302184446307908
    base = (wf.gaussian(width) >> shift) * wf.cos(frequency, phase)
    inside = (x >= -0.75 * width + shift) & (x < 0.75 * width + shift)
    for order in range(4):
        expected = np.zeros_like(x)
        z = (x[inside] - shift) / std_sq2
        for gaussian_order in range(order + 1):
            carrier_order = order - gaussian_order
            gaussian_part = (
                (-1) ** gaussian_order / std_sq2 ** gaussian_order
                * special.eval_hermite(gaussian_order, z) * np.exp(-(z**2))
            )
            carrier_part = frequency ** carrier_order * np.cos(
                frequency * x[inside] + phase + carrier_order * np.pi / 2
            )
            expected[inside] += (
                special.comb(order, gaussian_order, exact=True)
                * gaussian_part * carrier_part
            )
        assert np.allclose(wf.D(base, order)(x), expected,
                           atol=2e-10, rtol=2e-10)


def test_metadata_pickle_and_removed_experimental_serializers():
    wav = (wf.gaussian(10) >> 5) + (wf.gaussian(10) >> 50)
    wav *= wf.cos(200)
    wav.start = -10
    wav.stop = 70
    wav.sample_rate = 20
    wav.label = "pulse"

    restored = pickle.loads(pickle.dumps(wav))
    assert restored == wav
    assert restored.start == wav.start
    assert restored.stop == wav.stop
    assert restored.sample_rate == wav.sample_rate
    assert restored.label == wav.label
    for name in ("tolist", "fromlist", "totree", "fromtree", "_as_v1"):
        assert not hasattr(wav, name)
    assert wav.begin == -2.5
    assert wav.end == 57.5


def test_sampling_points_chirps_and_mixing():
    x = np.linspace(0, 2, 2048, endpoint=False)
    points = np.sin(np.linspace(0, 3, 65))
    sampled = wf.samplingPoints(0, 2, points)
    assert np.allclose(sampled(x), np.interp(x, np.linspace(0, 2, 65), points))

    for kind in ("linear", "exponential", "hyperbolic"):
        wav = wf.chirp(1, 2, 2, 0.3, kind)
        if kind == "linear":
            phase = 0.3 + 2 * np.pi * ((2 - 1) / 4 * x**2 + x)
        elif kind == "exponential":
            alpha = np.log(2) / 2
            phase = 0.3 + 2 * np.pi * (np.exp(alpha * x) - 1) / alpha
        else:
            k = (1 - 2) / (2 * 2)
            phase = 0.3 + 2 * np.pi / k * np.log(1 + k * x)
        assert np.allclose(wav(x), np.sin(phase))

    i, q = wf.mixing(wf.gaussian(0.5), phase=0.2, freq=1.3,
                     DRAGScaling=0.01)
    assert np.all(np.isfinite(i(x)))
    assert np.all(np.isfinite(q(x)))


@pytest.mark.parametrize("constructor,extra", [
    (wf.drag_sin, {}),
    (wf.drag_sinx, {"tab": 0.47}),
])
def test_multi_frequency_drag_roundtrip_shift_and_parser(constructor, extra):
    parameters = dict(
        freq=5e9,
        width=22.22e-9,
        plateau=3e-9,
        delta=-13.7e6,
        block_freq=(-91e6, 37e6, 124e6),
        phase=0.31,
        t0=7e-9,
        **extra,
    )
    wav = constructor(**parameters)
    x = np.linspace(-2e-9, 40e-9, 8193)
    values = wav(x)
    assert np.all(np.isfinite(values))
    assert np.all(values[(x < parameters["t0"])
                         | (x >= parameters["t0"] + parameters["width"]
                            + parameters["plateau"])] == 0)

    restored = wf.Waveform.from_bytes(wav.to_bytes())
    assert restored == wav
    assert np.array_equal(restored(x), values)
    delay = 11e-9
    assert np.allclose((wav >> delay)(x + delay), values)

    expression = (
        f"{constructor.__name__}(5e9, 22.22e-9, plateau=3e-9, "
        "delta=-13.7e6, block_freq=(-91e6, 37e6, 124e6), "
        f"phase=0.31, t0=7e-9{', tab=0.47' if extra else ''})"
    )
    assert wf.wave_eval(expression) == wav


def test_drag_sin_scalar_block_frequency_and_argument_validation():
    scalar = wf.drag_sin(5e9, 20e-9, block_freq=80e6)
    sequence = wf.drag_sin(5e9, 20e-9, block_freq=(80e6,))
    assert scalar.to_bytes() == sequence.to_bytes()
    with pytest.raises(ValueError):
        wf.drag_sin(5e9, 0)
    with pytest.raises(ValueError):
        wf.drag_sinx(5e9, 20e-9, tab=0)


def test_multi_frequency_drag_matches_v2_numerical_behavior():
    parameters = dict(
        freq=5e9,
        width=22.22e-9,
        plateau=3e-9,
        delta=-13.7e6,
        block_freq=(-91e6, 37e6, 124e6),
        phase=0.31,
        t0=7e-9,
    )
    fractions = np.array([0, 0.07, 0.19, 0.37, 0.51,
                          0.68, 0.83, 0.96, 0.999])
    x = parameters["t0"] + fractions * (
        parameters["width"] + parameters["plateau"]
    )
    expected_sin = np.array([
        0,
        0.20532545807257174,
        0.6208384978963238,
        -0.4333357461117501,
        -0.08840393960186489,
        -0.8620379323126761,
        -0.5748645543753418,
        -0.07489572316435908,
        -0.00309835789254227,
    ])
    expected_sinx = np.array([
        0,
        0.11971095177791738,
        0.36196762048509773,
        -0.03689065391238458,
        -0.05154217041569324,
        -0.19970252200405683,
        -0.3351634210081106,
        -0.04366647169944161,
        -0.00180643635595232,
    ])
    assert np.allclose(wf.drag_sin(**parameters)(x), expected_sin,
                       rtol=2e-13, atol=2e-13)
    assert np.allclose(wf.drag_sinx(**parameters, tab=0.47)(x), expected_sinx,
                       rtol=2e-13, atol=2e-13)


def test_filters_and_chunked_sampling():
    sample_rate = 1000
    b, a = butter(3, 4.0, "lowpass", fs=sample_rate)
    zi = lfiltic(b, a, [0])
    x = np.linspace(-1, 1, 2000, endpoint=False)

    wav = wf.step(0)
    wav.start = -1
    wav.stop = 1
    wav.sample_rate = sample_rate
    wav.filters = (tf2sos(b, a), 0)
    expected = lfilter(b, a, np.heaviside(x, 1), zi=zi)[0]
    assert np.allclose(wav.sample(), expected)
    assert np.allclose(np.concatenate(list(wav.sample(chunk_size=137))), expected)


def test_wavevstack_template_sharing_operations_and_roundtrip():
    template = wf.gaussian(20e-9) * wf.cos(2 * np.pi * 5e9)
    waves = [template >> (i * 80e-9) for i in range(1000)]
    stack = wf.WaveVStack(waves)

    # One template plus three packed event arrays; substantially smaller than
    # serializing 1000 complete shifted expression trees.
    assert len(stack.to_bytes()) < 32_000
    x = np.linspace(0, 80e-6, 16_000, endpoint=False)
    assert np.allclose(stack(x), stack.simplify()(x))

    shifted = (stack + 2.0) >> 1e-6
    shifted = shifted * 0.25 + wf.sin(2 * np.pi * 1e6)
    assert np.allclose(shifted(x), shifted.simplify()(x))

    materialized = wf.WaveVStack.from_bytes(shifted.to_bytes())
    assert np.allclose(materialized(x), shifted(x))

    restored = wf.WaveVStack.from_bytes(stack.to_bytes())
    assert restored == stack
    assert pickle.loads(pickle.dumps(stack)) == stack


def test_notebook_constructors_and_wave_eval():
    x = np.linspace(-20, 60, 4001)
    linear_step = np.where(x < -1, 0, np.where(x < 1, 0.5 + x / 2, 1))
    cosine_step = np.where(
        x < -1, 0, np.where(x < 1, 0.5 + 0.5 * np.sin(np.pi * x / 2), 1)
    )
    erf_step = np.where(
        x < -2, 0, np.where(x < 2, 0.5 + 0.5 * special.erf(x / 0.4), 1)
    )
    assert np.allclose(wf.step(2, "linear")(x), linear_step)
    assert np.allclose(wf.step(2, "cos")(x), cosine_step)
    assert np.allclose(wf.step(2, "erf")(x), erf_step)

    assert np.allclose(wf.exp(0.03 + 0.2j)(x), np.exp((0.03 + 0.2j) * x))
    inside = np.array([-2.0, -1.0, 0.0, 1.5, 2.999])
    assert np.allclose(wf.interp([-2, 0, 3], [1, -1, 2])(inside),
                       np.interp(inside, [-2, 0, 3], [1, -1, 2]))

    for wav in (wf.square(8, 2), wf.gaussian(8, 3), wf.cosPulse(8, 2),
                wf.coshPulse(8, 2, 3), wf.mollifier(8, 2), wf.sinc(3)):
        with np.errstate(divide="ignore", invalid="ignore"):
            values = wav(x)
        assert np.all(np.isfinite(values))

    expression = ("(gaussian(12) >> 3) * cos(16.2, 1.63) + "
                  "0.5*(gaussian(12) >> 35) * cos(16.2, 2)")
    actual = wf.wave_eval(expression)
    expected = wf.gaussian(12) >> 3
    expected = expected * wf.cos(16.2, 1.63)
    expected += 0.5 * (wf.gaussian(12) >> 35) * wf.cos(16.2, 2)
    assert isinstance(actual, wf.Waveform)
    assert np.allclose(actual(x), expected(x))


def test_time_is_global_configuration_and_blocks_store_integer_ticks():
    assert wf.get_time_resolution() == 1e-12
    wav = wf.square(4e-9) >> 11e-9
    ticks = wav._core.get_bound_ticks()
    assert ticks.dtype == np.dtype("<i8")
    assert np.array_equal(ticks[:-1], [-2_000, 2_000])
    assert wav._delay == 11e-9
    with pytest.raises(RuntimeError):
        wf.set_time_resolution(1e-15)

    code = (
        "import waveforms as w; "
        "w.set_time_resolution(1e-15); "
        "x=w.square(4e-12); "
        "print(w.get_time_resolution(), x._core.get_bound_ticks()[0])"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True)
    assert result.stdout.strip() == "1e-15 -2000"


def test_single_packed_backend_has_no_waveform2_modules():
    package = Path(wf.__file__).parent
    python_source = (package / "waveform.py").read_text()
    cython_source = (package / "_waveform.pyx").read_text()
    assert "waveform2" not in python_source
    assert "_waveform2" not in cython_source
    assert not (package / "waveform2.py").exists()
    assert not (package / "_waveform2.pyx").exists()

    code = (
        "import sys; from waveforms import gaussian; gaussian(1); "
        "print('waveforms.waveform2' in sys.modules, "
        "'waveforms._waveform2' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True)
    assert result.stdout.strip() == "False False"


def test_custom_functions_are_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        wf.function(lambda x: x)
    with pytest.raises(NotImplementedError):
        wf.registerBaseFunc(lambda x: x)
