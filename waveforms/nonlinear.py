"""Compact nonlinear transfer maps for post-sampling calibration.

The public :class:`NonlinearMap` compiles calibration samples to an NLM1
uniform-grid block.  Runtime evaluation is implemented by the C core and is
independent of SciPy; SciPy is used only while compiling monotone cubic maps.
"""

from __future__ import annotations

from math import ceil, log2

import numpy as np
from scipy.interpolate import PchipInterpolator

from ._waveform import (
    CNonlinearMapCore as _CNonlinearMapCore,
    NONLINEAR_CLIP,
    NONLINEAR_CUBIC,
    NONLINEAR_ERROR,
    NONLINEAR_FLOAT32,
    NONLINEAR_FLOAT64,
    NONLINEAR_LINEAR,
)


_METHODS = {
    "linear": NONLINEAR_LINEAR,
    "pchip": NONLINEAR_CUBIC,
    "monotone_cubic": NONLINEAR_CUBIC,
}
_METHOD_NAMES = {
    NONLINEAR_LINEAR: "linear",
    NONLINEAR_CUBIC: "monotone_cubic",
}
_EXTRAPOLATIONS = {
    "error": NONLINEAR_ERROR,
    "clip": NONLINEAR_CLIP,
}
_EXTRAPOLATION_NAMES = {
    NONLINEAR_ERROR: "error",
    NONLINEAR_CLIP: "clip",
}


def _storage_code(dtype) -> tuple[np.dtype, int]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return dtype, NONLINEAR_FLOAT32
    if dtype == np.dtype(np.float64):
        return dtype, NONLINEAR_FLOAT64
    raise TypeError("nonlinear-map dtype must be float32 or float64")


def _default_table_size(sample_count: int) -> int:
    intervals = max(256, sample_count - 1)
    return (1 << ceil(log2(intervals))) + 1


def _compiled_value(coefficients, method, x_min, x_max, point_count, value):
    """Evaluate one compiled map before reference-output subtraction."""
    intervals = point_count - 1
    coordinate = (value - x_min) * intervals / (x_max - x_min)
    coordinate = min(max(coordinate, 0.0), float(intervals))
    interval = min(int(coordinate), intervals - 1)
    fraction = 1.0 if coordinate >= intervals else coordinate - interval
    if method == NONLINEAR_LINEAR:
        left = coefficients[interval]
        return float(left + fraction * (coefficients[interval + 1] - left))
    a, b, c, d = coefficients[4 * interval:4 * interval + 4]
    return float(a + fraction * (b + fraction * (c + fraction * d)))


def _compiled_values(coefficients, method, x_min, x_max, point_count, values):
    """Vectorized reference evaluator used only while compiling a table."""
    values = np.asarray(values, dtype=np.float64)
    intervals = point_count - 1
    coordinate = np.clip(
        (values - x_min) * intervals / (x_max - x_min),
        0.0,
        float(intervals),
    )
    index = np.minimum(coordinate.astype(np.int64), intervals - 1)
    fraction = np.where(
        coordinate >= intervals, 1.0, coordinate - index,
    )
    if method == NONLINEAR_LINEAR:
        left = coefficients[index]
        return left + fraction * (coefficients[index + 1] - left)
    matrix = coefficients.reshape(intervals, 4)
    selected = matrix[index]
    return selected[:, 0] + fraction * (
        selected[:, 1] + fraction * (
            selected[:, 2] + fraction * selected[:, 3]
        )
    )


class NonlinearMap:
    """A serializable scalar transfer function evaluated after sampling.

    ``reference`` selects a centered operating point.  With a reference
    ``x_ref``, an input sample ``v`` evaluates ``g(x_ref + v) - g(x_ref)``.
    This makes a zero waveform remain zero around an idle bias.
    """

    __slots__ = ("_core",)

    def __init__(self, core: _CNonlinearMapCore):
        if not isinstance(core, _CNonlinearMapCore):
            raise TypeError("NonlinearMap requires a native nonlinear-map core")
        self._core = core

    @classmethod
    def from_samples(
        cls,
        x,
        y,
        *,
        method: str = "monotone_cubic",
        table_size: int | None = None,
        dtype=np.float64,
        reference: float | None = None,
        extrapolate: str = "error",
        max_error: float | None = None,
        max_table_size: int = 65_537,
    ) -> "NonlinearMap":
        """Compile monotone calibration samples to a uniform lookup table.

        Parameters
        ----------
        x, y:
            Finite one-dimensional calibration vectors. ``x`` must increase
            strictly and ``y`` must be monotone on the selected inverse branch.
        method:
            ``"linear"`` or ``"monotone_cubic"``/``"pchip"``.
        table_size:
            Number of uniform runtime nodes.  By default a power-of-two grid
            with at least 257 nodes is used.
        dtype:
            Serialized coefficient precision, either float32 or float64.
        reference:
            Optional idle x coordinate. Inputs then represent offsets from it,
            and the corresponding y value is subtracted from every result.
        extrapolate:
            ``"error"`` rejects out-of-domain samples; ``"clip"`` clamps to
            the nearest endpoint.
        max_error:
            Optional maximum absolute error in output units relative to the
            source linear/PCHIP interpolant. The uniform table doubles until
            quarter-interval validation points satisfy this bound.
        max_table_size:
            Largest table allowed during error-controlled compilation.
        """
        try:
            method_code = _METHODS[method]
        except KeyError as exc:
            raise ValueError(
                "method must be 'linear', 'monotone_cubic', or 'pchip'"
            ) from exc
        try:
            extrapolation_code = _EXTRAPOLATIONS[extrapolate]
        except KeyError as exc:
            raise ValueError("extrapolate must be 'error' or 'clip'") from exc
        storage_dtype, storage_code = _storage_code(dtype)

        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        if (x_values.ndim != 1 or y_values.ndim != 1
                or x_values.shape != y_values.shape):
            raise ValueError("x and y must be equal-length one-dimensional arrays")
        if len(x_values) < 2:
            raise ValueError("at least two calibration samples are required")
        if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
            raise ValueError("calibration samples must be finite")
        if not np.all(np.diff(x_values) > 0):
            raise ValueError("x calibration samples must increase strictly")
        differences = np.diff(y_values)
        if not (np.all(differences >= 0) or np.all(differences <= 0)):
            raise ValueError(
                "y calibration samples must be monotone on one inverse branch"
            )

        if table_size is None:
            table_size = _default_table_size(len(x_values))
        if (not isinstance(table_size, (int, np.integer))
                or table_size < 2 or table_size > 16_777_217):
            raise ValueError("table_size must be an integer between 2 and 16777217")
        table_size = int(table_size)
        if max_error is not None:
            max_error = float(max_error)
            if not np.isfinite(max_error) or max_error <= 0:
                raise ValueError("max_error must be a finite positive number")
            if (not isinstance(max_table_size, (int, np.integer))
                    or max_table_size < table_size
                    or max_table_size > 16_777_217):
                raise ValueError(
                    "max_table_size must be an integer between table_size and 16777217"
                )
            max_table_size = int(max_table_size)

        x_min = float(x_values[0])
        x_max = float(x_values[-1])
        if method_code == NONLINEAR_LINEAR:
            def source_values(points):
                return np.interp(points, x_values, y_values)
        else:
            source_curve = PchipInterpolator(
                x_values, y_values, extrapolate=False)
            def source_values(points):
                return np.asarray(source_curve(points), dtype=np.float64)

        def compile_table(size):
            grid = np.linspace(x_min, x_max, size, dtype=np.float64)
            ordinates = source_values(grid)
            if method_code == NONLINEAR_LINEAR:
                payload = ordinates
            else:
                compiled_curve = PchipInterpolator(
                    grid, ordinates, extrapolate=False)
                step = (x_max - x_min) / (size - 1)
                powers = compiled_curve.c
                payload = np.column_stack((
                    powers[3],
                    powers[2] * step,
                    powers[1] * step**2,
                    powers[0] * step**3,
                )).reshape(-1)
            # Round now so validation covers the serialized precision too.
            return np.asarray(payload, dtype=storage_dtype).astype(
                np.float64, copy=False)

        while True:
            coefficients = compile_table(table_size)
            if max_error is None:
                break
            check = np.linspace(
                x_min, x_max, 4 * (table_size - 1) + 1,
                dtype=np.float64,
            )
            error = np.max(np.abs(
                _compiled_values(
                    coefficients, method_code, x_min, x_max,
                    table_size, check,
                ) - source_values(check)
            ))
            if error <= max_error:
                break
            next_size = 2 * (table_size - 1) + 1
            if next_size > max_table_size:
                raise ValueError(
                    f"max_error={max_error:g} requires more than "
                    f"max_table_size={max_table_size} points; "
                    f"measured error is {error:g}"
                )
            table_size = next_size

        # Compute the reference from the rounded final payload so float32 maps
        # retain the centered semantics of their serialized coefficients.
        if reference is None:
            input_offset = 0.0
            output_offset = 0.0
        else:
            input_offset = float(reference)
            if not np.isfinite(input_offset) or not x_min <= input_offset <= x_max:
                raise ValueError("reference must be finite and inside the x domain")
            output_offset = _compiled_value(
                coefficients, method_code, x_min, x_max,
                table_size, input_offset,
            )

        core = _CNonlinearMapCore.create(
            method_code, storage_code, extrapolation_code,
            x_min, x_max, input_offset, output_offset,
            coefficients, table_size,
        )
        return cls(core)

    @classmethod
    def from_bytes(cls, data) -> "NonlinearMap":
        return cls(_CNonlinearMapCore.from_bytes(data))

    def to_bytes(self) -> bytes:
        return self._core.to_bytes()

    def apply(self, values, out=None):
        scalar = np.isscalar(values)
        result = self._core.apply(values, out=out)
        if scalar and out is None:
            return result.item()
        return result

    __call__ = apply

    def _apply_quantized(self, values, bits, full_scale=1.0, out=None):
        return self._core.apply(
            values, bits=int(bits), full_scale=full_scale, out=out)

    @property
    def method(self) -> str:
        return _METHOD_NAMES[self._core.method]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(
            np.float32 if self._core.storage == NONLINEAR_FLOAT32
            else np.float64
        )

    @property
    def extrapolate(self) -> str:
        return _EXTRAPOLATION_NAMES[self._core.extrapolation]

    @property
    def point_count(self) -> int:
        return self._core.point_count

    @property
    def absolute_domain(self) -> tuple[float, float]:
        return self._core.x_min, self._core.x_max

    @property
    def domain(self) -> tuple[float, float]:
        return (
            self._core.x_min - self._core.input_offset,
            self._core.x_max - self._core.input_offset,
        )

    @property
    def reference_input(self) -> float:
        return self._core.input_offset

    @property
    def reference_output(self) -> float:
        return self._core.output_offset

    def __eq__(self, other):
        return isinstance(other, NonlinearMap) and self._core == other._core

    def __hash__(self):
        return hash(self._core)

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))

    def __repr__(self):
        return (
            f"NonlinearMap(method={self.method!r}, dtype={self.dtype.name!r}, "
            f"points={self.point_count}, domain={self.domain!r}, "
            f"extrapolate={self.extrapolate!r})"
        )


__all__ = ["NonlinearMap"]
