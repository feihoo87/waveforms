import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.special as special
from scipy.signal import butter, lfilter, lfiltic, tf2sos

import waveforms as wf
from waveforms._waveform import (
    quantize_samples, sample_clock, sample_grid, time_to_tick,
)


PUBLIC_NAMES = {
    "ComplexWaveform", "ComplexWaveVStack", "D", "RealWaveform",
    "RealWaveVStack", "Waveform", "WaveVStack",
    "chirp", "const", "cos", "cosh",
    "coshPulse", "cosPulse", "cut", "drag", "drag_sin", "drag_sinx",
    "exp", "function",
    "gaussian", "general_cosine", "get_time_resolution", "hanning", "interp", "mixing",
    "mollifier", "one", "poly", "registerBaseFunc", "registerDerivative",
    "quantize_time", "sample_clock", "sample_grid", "samplingPoints",
    "set_time_resolution", "sign", "sin", "sinc", "sinh", "square",
    "step", "t", "tick_to_time", "time_to_tick", "wave_eval", "zero",
}


def test_public_api_and_basic_sampling():
    assert PUBLIC_NAMES <= set(dir(wf))
    x = np.linspace(-2.0, 2.0, 1001)
    assert np.allclose(wf.cos(1.3)(x), np.cos(1.3 * x))
    assert np.allclose(wf.sin(0.7)(x), np.sin(0.7 * x))
    assert np.allclose(wf.poly([1, -0.5, 0.25])(x),
                       1 - 0.5 * x + 0.25 * x**2)
    assert wf.zero().is_zero()
    assert (0 * wf.gaussian(1)).is_zero()
    assert not wf.one().is_zero()


def test_binary_roundtrip_is_zero_copy_for_bytes_input():
    wav = (wf.gaussian(0.8) >> 1.2) * wf.cos(2.3) + 0.25
    data = wav.to_bytes()
    restored = wf.Waveform.from_bytes(data)
    assert restored._core.to_bytes() is data
    assert restored == wav
    assert pickle.loads(pickle.dumps(wav)) == wav
    assert wf.one() >> 5 == wf.one()
    assert (wf.const(2.5) << 7).to_bytes() == wf.const(2.5).to_bytes()


def test_affine_algebra_matches_eager_materialization():
    left = 0.37 * (wf.gaussian(0.8) >> 1.2)
    right = -1.25 * (wf.cos(2.3, 0.4) << 0.35)

    actual_add = left + right
    actual_mul = left * right

    x = np.linspace(-3, 4, 8193)
    assert np.allclose(actual_add(x), left(x) + right(x),
                       rtol=2e-15, atol=2e-15)
    assert np.allclose(actual_mul(x), left(x) * right(x),
                       rtol=2e-15, atol=2e-15)
    assert wf.Waveform.from_bytes(actual_add.to_bytes()) == actual_add
    assert wf.Waveform.from_bytes(actual_mul.to_bytes()) == actual_mul


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
                           atol=2e-9, rtol=2e-10)


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


def test_multi_frequency_drag_matches_120ghz_tick_behavior():
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
        0.2054835614941095,
        0.6212078536589116,
        -0.4335035059651672,
        -0.0884229171775666,
        -0.8622124313517241,
        -0.5746069534292936,
        -0.07479711741358212,
        -0.0026910881959570747,
    ])
    expected_sinx = np.array([
        0,
        0.1197774183686397,
        0.3621052333362499,
        -0.03599350352172509,
        -0.05154217041569324,
        -0.20015318106495275,
        -0.3349413303818123,
        -0.04359962208204045,
        -0.0015686490655034638,
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

    # One template plus three contiguous event arrays; substantially smaller than
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


def test_complex_waveform_is_two_real_channels():
    x = np.linspace(-2, 2, 2001)
    envelope = wf.gaussian(1.2)
    carrier = wf.cos(1.7, 0.3)
    wav = (2 + 3j) * envelope + (4 - 0.5j) * carrier
    expected = ((2 + 3j) * envelope(x)
                + (4 - 0.5j) * carrier(x))

    assert isinstance(wav, wf.ComplexWaveform)
    assert np.allclose(wf.ComplexWaveform(2 + 3j)(x), 2 + 3j)
    real_as_complex = wf.ComplexWaveform(envelope, wf.zero())
    assert real_as_complex == envelope
    assert hash(real_as_complex) == hash(envelope)
    assert isinstance(wav.real, wf.Waveform)
    assert isinstance(wav.imag, wf.Waveform)
    assert np.allclose(wav(x), expected)
    assert np.allclose(wav.real(x), expected.real)
    assert np.allclose(wav.imag(x), expected.imag)
    assert np.allclose((wav * (0.25 - 0.75j))(x),
                       expected * (0.25 - 0.75j))
    assert np.allclose((wav * wav)(x), expected**2)
    assert np.allclose((wav >> 0.4)(x + 0.4), expected)
    assert np.allclose(wf.D(wav)(x),
                       wf.D(wav.real)(x) + 1j * wf.D(wav.imag)(x))

    out = np.full(x.shape, 7 + 11j, dtype=np.complex128)
    assert wav(x, out=out) is out
    assert np.allclose(out, expected)
    wav(x, out=out, accumulate=True)
    assert np.allclose(out, 2 * expected)

    data = wav.to_bytes()
    assert data[:4] == b"CWF1"
    assert wf.ComplexWaveform.from_bytes(data) == wav
    assert isinstance(wf.Waveform.from_bytes(data), wf.ComplexWaveform)
    assert pickle.loads(pickle.dumps(wav)) == wav

    parsed = wf.wave_eval("(2+3j)*gaussian(1.2) + (4-0.5j)*cos(1.7, 0.3)")
    assert isinstance(parsed, wf.ComplexWaveform)
    assert np.allclose(parsed(x), expected)


def test_complex_interpolation_and_stack_roundtrip():
    x = np.linspace(0, 3, 3001, endpoint=False)
    points = np.array([1 + 2j, -0.5 + 0.25j, 2 - 3j, 0.75 + 0.5j])
    sampled = wf.samplingPoints(0, 3, points)
    expected_sampled = (
        np.interp(x, np.linspace(0, 3, len(points)), points.real)
        + 1j * np.interp(x, np.linspace(0, 3, len(points)), points.imag)
    )
    assert isinstance(sampled, wf.ComplexWaveform)
    assert np.allclose(sampled(x), expected_sampled)
    linear = wf.interp([0, 1, 3], [1 + 2j, -1 + 0.5j, 2 - 3j])
    expected_linear = (
        np.interp(x, [0, 1, 3], [1, -1, 2])
        + 1j * np.interp(x, [0, 1, 3], [2, 0.5, -3])
    )
    assert isinstance(linear, wf.ComplexWaveform)
    assert np.allclose(linear(x), expected_linear)

    template = wf.gaussian(0.2) * wf.cos(17)
    waves = [(index + 1j * (index + 1)) * (template >> (0.3 * index))
             for index in range(8)]
    stack = wf.ComplexWaveVStack(waves)
    expected = sum((wav(x) for wav in waves), np.zeros_like(x, dtype=complex))
    assert np.allclose(stack(x), expected)
    assert np.allclose(stack.simplify()(x), expected)

    data = stack.to_bytes()
    assert data[:4] == b"CWS1"
    assert isinstance(wf.WaveVStack.from_bytes(data), wf.ComplexWaveVStack)
    assert np.allclose(wf.ComplexWaveVStack.from_bytes(data)(x), expected)
    assert pickle.loads(pickle.dumps(stack)) == stack

    promoted = wf.WaveVStack([template]) * (2 - 0.25j)
    assert isinstance(promoted, wf.ComplexWaveVStack)
    assert np.allclose(promoted(x), (2 - 0.25j) * template(x))
    direct = wf.ComplexWaveVStack(waves[0])
    assert np.allclose(direct(x), waves[0](x))
    real_only = wf.ComplexWaveVStack(wf.WaveVStack([template]))
    assert real_only.begin == template.begin
    assert real_only.end == template.end
    with pytest.raises(TypeError, match="real-only"):
        wf.WaveVStack(waves)


def test_real_backend_promotes_complex_coefficients_and_scales():
    wav = wf.gaussian(1.0)
    stack = wf.WaveVStack([wav, 2 * (wav >> 3)])
    assert wav.to_bytes()[:4] == b"WNF4"
    assert stack.to_bytes()[:4] == b"WNS4"
    assert stack(np.linspace(-1, 4, 101)).dtype == np.float64

    assert isinstance(wav * 1j, wf.ComplexWaveform)
    assert isinstance(stack * 1j, wf.ComplexWaveVStack)
    with pytest.raises((TypeError, ValueError)):
        wav(np.array([0.0 + 0.0j]))


def test_time_is_global_configuration_and_blocks_store_integer_ticks():
    assert wf.get_time_resolution() == 1 / 120_000_000_000
    wav = wf.square(4e-9) >> 11e-9
    data = wav.to_bytes()
    assert data[:4] == b"WNF4"
    assert wav.begin == 9e-9
    assert wav.end == 13e-9
    with pytest.raises(RuntimeError):
        wf.set_time_resolution(1e-15)

    code = (
        "import waveforms as w; "
        "w.set_time_resolution(1e-15); "
        "x=w.square(4e-12); "
        "print(w.get_time_resolution(), round(x.begin / 1e-15))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True)
    assert result.stdout.strip() == "1e-15 -2000"


def test_device_sample_clocks_and_rational_fallback():
    expected = {
        500_000_000: 240,
        1_000_000_000: 120,
        1_200_000_000: 100,
        2_000_000_000: 60,
        2_400_000_000: 50,
        2_500_000_000: 48,
        4_000_000_000: 30,
        6_000_000_000: 20,
        8_000_000_000: 15,
        10_000_000_000: 12,
    }
    assert {rate: sample_clock(rate) for rate in expected} == {
        rate: (ticks, 1) for rate, ticks in expected.items()
    }
    assert sample_clock(7_000_000_000) == (120, 7)
    grid = sample_grid(-1200, 100, 20)
    assert grid[51] == -1.5e-9
    assert np.all(np.diff(grid) > 0)


def test_fixed_width_quantization_and_rational_sampling():
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2.0])
    assert np.array_equal(
        quantize_samples(values, 16),
        [-32768, -32768, -16384, 0, 16384, 32767, 32767],
    )
    assert np.array_equal(
        quantize_samples(values, 32),
        [-2147483648, -2147483648, -1073741824, 0,
         1073741824, 2147483647, 2147483647],
    )

    # Exercise the fused SIMD path, including its historic half-away-from-zero
    # rounding rule and multidimensional ``out`` handling.
    half_steps = np.array([0.5, -0.5, 1.5, -1.5] * 8) / 32768.0
    target = np.empty((8, 4), dtype=np.int16)
    assert quantize_samples(half_steps.reshape(8, 4), 16, out=target) is target
    assert np.array_equal(
        target.reshape(-1), np.array([1, -1, 2, -2] * 8, dtype=np.int16)
    )

    wav = 0.8 * wf.gaussian(20e-9)
    wav.start = -20e-9
    wav.stop = 20e-9
    rate = 2_400_000_000
    float_samples = wav.sample(rate)
    assert np.array_equal(
        wav.sample(rate, dtype=np.int16),
        quantize_samples(float_samples, 16),
    )
    whole = wav.sample(rate, dtype=np.int32)
    chunks = np.concatenate(list(
        wav.sample(rate, dtype=np.int32, chunk_size=17)
    ))
    assert np.array_equal(chunks, whole)

    odd_rate = 7_000_000_000
    rational_samples = wav.sample(odd_rate)
    numerator, denominator = sample_clock(odd_rate)
    rational_grid = sample_grid(
        time_to_tick(wav.start), len(rational_samples),
        numerator, denominator,
    )
    assert np.allclose(rational_samples, wav(rational_grid), rtol=0, atol=1e-15)


def test_simd_node_evaluation_matches_scalar_boundaries():
    positions = np.linspace(-2.0, 2.0, 513)
    waves = (
        wf.t(),
        wf.exp(0.2),
        wf.sinc(1.3),
        wf.cosh(0.3),
        wf.sinh(0.3),
        (wf.gaussian(4.0, 1.5) * wf.cos(2.2)) ** 2,
    )
    for wave in waves:
        expected = np.array([wave(float(position)) for position in positions])
        assert np.allclose(wave(positions), expected, rtol=1e-12, atol=1e-13)


def test_integer_sampling_fast_paths_are_bit_exact_and_pickle_safe():
    rate = 2_400_000_000
    pulse = 0.8 * wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
    stack = wf.WaveVStack(
        pulse >> (index * 40e-9) for index in range(100)
    ) + 0.13
    stack.start = -20e-9
    stack.stop = 4e-6
    float_samples = stack.sample(rate)

    for dtype, bits in ((np.int16, 16), (np.int32, 32)):
        expected = quantize_samples(float_samples, bits)
        actual = stack.sample(rate, dtype=dtype)
        assert np.array_equal(actual, expected)
        out = np.empty(len(expected), dtype=dtype)
        assert stack.sample(rate, out=out) is out
        assert np.array_equal(out, expected)

    restored = pickle.loads(pickle.dumps(stack))
    assert np.array_equal(
        restored.sample(rate, dtype=np.int16),
        quantize_samples(restored.sample(rate), 16),
    )

    varying = wf.WaveVStack(
        (0.2 + index / 1000) * (pulse >> (index * 40e-9))
        for index in range(100)
    )
    varying.start = stack.start
    varying.stop = stack.stop
    assert np.array_equal(
        varying.sample(rate, dtype=np.int16),
        quantize_samples(varying.sample(rate), 16),
    )


def test_integer_sampling_filtered_and_overlapping_fallbacks_are_exact():
    rate = 2_400_000_000
    pulse = 0.4 * wf.gaussian(20e-9)
    overlapping = wf.WaveVStack(
        pulse >> (index * 2e-9) for index in range(20)
    )
    overlapping.start = -20e-9
    overlapping.stop = 60e-9
    assert np.array_equal(
        overlapping.sample(rate, dtype=np.int16),
        quantize_samples(overlapping.sample(rate), 16),
    )

    b, a = butter(3, 50e6, "lowpass", fs=rate)
    overlapping.filters = (tf2sos(b, a), 0)
    assert np.array_equal(
        overlapping.sample(rate, dtype=np.int16),
        quantize_samples(overlapping.sample(rate), 16),
    )

    with pytest.raises(ValueError, match="non-finite"):
        quantize_samples(np.array([0.0, np.nan]), 16)


def test_wavevstack_integer_grid_template_sampling_and_complex_iq():
    rate = 6_000_000_000
    step = sample_clock(rate)
    pulse = wf.gaussian(10e-9) * wf.cos(2 * np.pi * 300e6)
    waves = [
        (0.25 + index / 100) * (
            pulse >> (index * 2e-9 + index % 3 * wf.get_time_resolution())
        )
        for index in range(50)
    ]
    stack = wf.WaveVStack(waves) + 0.13
    stack.start = -10e-9
    stack.stop = 110e-9
    actual = stack.sample(rate)
    x = sample_grid(time_to_tick(stack.start), len(actual), *step)
    expected = stack(x)
    assert np.allclose(actual, expected, rtol=2e-15, atol=2e-15)

    complex_stack = wf.ComplexWaveVStack(stack, -0.5 * stack)
    complex_stack.start = stack.start
    complex_stack.stop = stack.stop
    i_data, q_data = complex_stack.sample_iq(rate, dtype=np.int16)
    complex_samples = complex_stack.sample(rate)
    assert np.array_equal(i_data, quantize_samples(complex_samples.real, 16))
    assert np.array_equal(q_data, quantize_samples(complex_samples.imag, 16))


def test_single_c_backend_has_no_waveform2_or_packed_core():
    package = Path(wf.__file__).parent
    python_source = (package / "waveform.py").read_text()
    cython_source = (package / "_waveform.pyx").read_text()
    assert "waveform2" not in python_source
    assert "_waveform2" not in cython_source
    assert "PackedWaveform" not in python_source + cython_source
    assert "PackedStack" not in python_source + cython_source
    assert "WFM3" not in python_source + cython_source
    assert "WVS3" not in python_source + cython_source
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
