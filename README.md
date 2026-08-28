# waveforms
[![View build status](https://github.com/feihoo87/waveforms/actions/workflows/workflow.yml/badge.svg)](https://github.com/feihoo87/waveforms/)
[![Coverage Status](https://coveralls.io/repos/github/feihoo87/waveforms/badge.svg?branch=master)](https://coveralls.io/github/feihoo87/waveforms?branch=master)
[![PyPI version](https://badge.fury.io/py/waveforms.svg)](https://pypi.org/project/waveforms/)

Form waveforms used in experiment.

## Installation
We encourage installing waveforms via the pip tool (a python package manager):
```bash
python -m pip install waveforms
```

To install from the latest source, you need to clone the GitHub repository on your machine.
```bash
git clone https://github.com/feihoo87/waveforms.git
```

Then dependencies and `waveforms` can be installed in this way:
```bash
cd waveforms
python -m pip install numpy
python -m pip install -e .
```

## Usage
```python
import numpy as np
import matplotlib.pyplot as plt

from waveforms import *

pulse = cosPulse(20e-9)

x_wav = zero()
y_wav = zero()

I, Q = mixing(0.5*pulse, freq=-20e6, DRAGScaling=0.2)
x_wav += I
y_wav += Q

I, Q = mixing(pulse >> 1e-6, freq=-20e6, phase=np.pi/2, DRAGScaling=0.2)
x_wav += I
y_wav += Q

I, Q = mixing((0.5 * pulse) >> 2e-6, freq=-20e6, DRAGScaling=0.2)
x_wav += I
y_wav += Q


t = np.linspace(-1e-6, 9e-6, 10001)
plt.plot(t, x_wav(t))
plt.plot(t, y_wav(t))
plt.show()
```

### Unified binary representation

`Waveform` is the common base class for every signal object. `RealWaveform`
and `ComplexWaveform` are concrete waveforms; `WaveVStack` is the common stack
base, with `RealWaveVStack` and `ComplexWaveVStack` as its concrete forms.
The original construction style remains unchanged and automatically selects
the compact C core for common pulses and repeated-pulse stacks:

```python
import waveforms as wf

# The default tick is already the period of 120 GHz (1 / 120e9 seconds).
# To use another global tick, override it before constructing/loading waveforms:
# wf.set_time_resolution(1e-12)

pulse = (wf.gaussian(12e-9) >> 20e-9) * wf.cos(2 * wf.pi * 5e9)
data = pulse.to_bytes()
restored = wf.Waveform.from_bytes(data)

assert isinstance(pulse, wf.Waveform)
stack = wf.WaveVStack([pulse, pulse >> 40e-9])
assert isinstance(stack, wf.RealWaveVStack)
assert isinstance(stack, wf.WaveVStack)
```

The C block format stores signed 64-bit ticks using a process-wide clock. The
default is 120 GHz. A different process-wide time resolution may be selected
before the first waveform is constructed; the same C representation and
evaluator continue to be used. The setting is locked by the first object.

`sample()` has integer-grid fast paths for 500 MHz, 1 GHz, 1.2 GHz, 2 GHz,
2.4 GHz, 2.5 GHz, 4 GHz, 6 GHz, 8 GHz, and 10 GHz. Real waveforms can be
quantized directly to signed DAC buffers:

```python
pulse.start = 0
pulse.stop = 100e-9
dac16 = pulse.sample(2_400_000_000, dtype=np.int16, full_scale=1.0)
```

The low-level cores store and evaluate real-valued signals only. Complex signals
are represented in Python as independent real and imaginary channels:

```python
z = (1 + 0.25j) * wf.gaussian(12e-9)
assert isinstance(z, wf.ComplexWaveform)

stack = wf.ComplexWaveVStack([z, z >> 20e-9])
samples = stack(t)  # complex NumPy array
```

Real waveforms and stacks therefore avoid complex storage and arithmetic for
the common real-valued case. `ComplexWaveform.real` and `.imag` expose the two
real channel waveforms.

## Reporting Issues
Please report all issues [on github](https://github.com/feihoo87/waveforms/issues).

## License

[MIT](https://opensource.org/licenses/MIT)
