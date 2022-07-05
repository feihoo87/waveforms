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

## Reporting Issues
Please report all issues [on github](https://github.com/feihoo87/waveforms/issues).

## License

[MIT](https://opensource.org/licenses/MIT)
