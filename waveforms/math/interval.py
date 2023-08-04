import ast
import bisect
import functools

try:
    from typing import Self
except:
    from typing import TypeVar
    Self = TypeVar('Self')

import numpy as np
from pyparsing import Forward, Literal, Optional, Or, Regex, Suppress

LCLOSED = Literal("[").setParseAction(lambda: [True])
LOPEN = Literal("(").setParseAction(lambda: [False])
RCLOSED = Literal("]").setParseAction(lambda: [True])
ROPEN = Literal(")").setParseAction(lambda: [False])
inf = Literal('inf').setParseAction(lambda: [np.inf])
number = Regex(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?").setParseAction(
    lambda tokens: [ast.literal_eval(tokens[0])])

left = (LOPEN + Suppress('-') + inf).setParseAction(lambda: [False, -np.inf]) \
     | Or([LCLOSED, LOPEN]) + number

right = (Optional(Suppress('+')) + inf + ROPEN).setParseAction(lambda: [np.inf, False]) \
      | number + Or([RCLOSED, ROPEN])

interval = left + Suppress(",") + right
interval.setParseAction(lambda t: _Interval(t[1], t[2], t[0], t[3]))

expression = Forward()
expression << interval + Optional(Suppress("U") + expression)


@functools.total_ordering
class _Interval():

    __slots__ = ['start', 'end', 'start_closed', 'end_closed']

    def __init__(self,
                 start=-np.inf,
                 end=np.inf,
                 start_closed=False,
                 end_closed=False):
        assert start <= end, f"start {start} should not be greater than end {end}."
        if start == -np.inf:
            assert not start_closed, f"start {start} should not be closed."
        if end == np.inf:
            assert not end_closed, f"end {end} should not be closed."
        self.start = start
        self.end = end
        self.start_closed = start_closed
        self.end_closed = end_closed

    def __contains__(self, point) -> bool:
        if self.start < point < self.end:
            return True
        if point == self.start and self.start_closed:
            return True
        if point == self.end and self.end_closed:
            return True
        return False

    def is_subset_of(self, other: Self) -> bool:
        return ((other.start < self.start or other.start == self.start and
                 (other.start_closed or not self.start_closed))
                and (self.end < other.end or self.end == other.end and
                     (not self.end_closed or other.end_closed)))

    def intersects(self, other: Self):
        if self.start > other.end or self.end < other.start:
            return False
        if self.start == other.end and not (self.start_closed
                                            and other.end_closed):
            return False
        if self.end == other.start and not (self.end_closed
                                            and other.start_closed):
            return False
        return True

    def intersection(self, other: Self) -> Self:
        if not self.intersects(other):
            return None
        start, start_closed = max((self.start, not self.start_closed),
                                  (other.start, not other.start_closed))
        start_closed = not start_closed
        end, end_closed = min((self.end, self.end_closed),
                              (other.end, other.end_closed))

        return _Interval(start,
                         end,
                         start_closed=start_closed,
                         end_closed=end_closed)

    def union(self, other: Self) -> Self:
        if not self.intersects(other):
            return None
        start, start_closed = min((self.start, not self.start_closed),
                                  (other.start, not other.start_closed))
        start_closed = not start_closed
        end, end_closed = max((self.end, self.end_closed),
                              (other.end, other.end_closed))
        return _Interval(start,
                         end,
                         start_closed=start_closed,
                         end_closed=end_closed)

    def empty(self) -> bool:
        return self.start > self.end or (
            self.start == self.end
            and not (self.start_closed and self.end_closed))

    def __neg__(self) -> Self:
        return _Interval(-self.end, -self.start, self.end_closed,
                         self.start_closed)

    def __add__(self, other: int | float) -> Self:
        return _Interval(self.start + other, self.end + other,
                         self.start_closed, self.end_closed)

    def __sub__(self, other: int | float) -> Self:

        return _Interval(self.start - other, self.end - other,
                         self.start_closed, self.end_closed)

    def __mul__(self, other: int | float) -> Self:
        return _Interval(self.start * other, self.end * other,
                         self.start_closed, self.end_closed)

    def __truediv__(self, other: int | float) -> Self:
        return _Interval(self.start / other, self.end / other,
                         self.start_closed, self.end_closed)

    def __and__(self, other: Self) -> Self:
        return self.intersection(other)

    def __or__(self, other: Self) -> Self:
        return self.union(other)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return not self.empty() and self.start == self.end == other
        if not isinstance(other, _Interval):
            raise TypeError(
                f"Cannot compare interval with {type(other)} {other}.")
        return (self.start == other.start and self.end == other.end
                and self.start_closed == other.start_closed
                and self.end_closed == other.end_closed)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.start > other
        if not isinstance(other, _Interval):
            raise TypeError(
                f"Cannot compare interval with {type(other)} {other}.")
        return (self.start, not self.start_closed, self.end,
                not self.end_closed) > (other.start, not other.start_closed,
                                        other.end, not other.end_closed)

    def __str__(self) -> str:
        if self.start == self.end and self.start_closed and self.end_closed:
            return f"{{{self.start}}}"
        elif self.start >= self.end:
            return "{}"
        return f"{'[' if self.start_closed else '('}{self.start}, {self.end}{']' if self.end_closed else ')'}"

    def __repr__(self) -> str:
        return f"_Interval({self.start}, {self.end}, {self.start_closed}, {self.end_closed})"


class Interval():

    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], list):
                intervals = args[0]
            elif isinstance(args[0], str):
                intervals = [
                    interval.parseString(s)[0] for s in args[0].split('U')
                ]
            else:
                raise ValueError(
                    'Interval can only be initialized with a list of intervals or a string'
                )
        elif len(args) == 2 and isinstance(args[0],
                                           (int, float)) and isinstance(
                                               args[1], (int, float)):
            intervals = [_Interval(*args)]
        elif len(args) == 4 and isinstance(args[0],
                                           (int, float)) and isinstance(
                                               args[1], (int, float)):
            intervals = [_Interval(*args)]
        else:
            intervals = []
        self.intervals: list[_Interval] = sorted(
            [interval for interval in intervals if not interval.empty()])

    def empty(self) -> bool:
        return len(self.intervals) == 0

    def full(self) -> bool:
        return len(self.intervals) == 1 and self.intervals[
            0].start == -np.inf and self.intervals[0].end == np.inf

    def __contains__(self, item: int | float) -> bool:
        if isinstance(item, (int, float)):
            i = bisect.bisect_left(self.intervals, item)
            if i == 0 or i > len(self.intervals):
                return False
            return item in self.intervals[i - 1]
        else:
            raise TypeError(f"Cannot check if {item} is in interval.")

    def is_subset_of(self, other: Self) -> bool:
        if self.empty():
            return True
        if other.empty():
            return False
        for interval in self.intervals:
            if not any(
                    interval.is_subset_of(other_interval)
                    for other_interval in other.intervals):
                return False
        return True

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.intervals == other.intervals
        elif isinstance(other, _Interval):
            return self.intervals == [other]
        else:
            raise TypeError(
                f"Cannot compare interval with {type(other)} {other}.")

    def __lt__(self, other: Self) -> bool:
        return self.is_subset_of(other) and not self == other

    def __le__(self, other: Self) -> bool:
        return self.is_subset_of(other)

    def __gt__(self, other: Self) -> bool:
        return other.is_subset_of(self) and not self == other

    def __ge__(self, other: Self) -> bool:
        return other.is_subset_of(self)

    def __neg__(self) -> Self:
        return Interval([-interval for interval in self.intervals])

    def __add__(self, other: int | float) -> Self:
        if isinstance(other, (int, float)):
            return Interval([interval + other for interval in self.intervals])
        else:
            raise TypeError(f"Cannot add interval with {type(other)} {other}.")

    def __radd__(self, other: int | float) -> Self:
        return self + other

    def __sub__(self, other: int | float) -> Self:
        if isinstance(other, (int, float)):
            return Interval([interval - other for interval in self.intervals])
        else:
            raise TypeError(f"Cannot add interval with {type(other)} {other}.")

    def __mul__(self, other: int | float) -> Self:
        if isinstance(other, (int, float)):
            return Interval([interval * other for interval in self.intervals])
        else:
            raise TypeError(f"Cannot add interval with {type(other)} {other}.")

    def __rmul__(self, other: int | float) -> Self:
        return self * other

    def __truediv__(self, other: int | float) -> Self:
        if isinstance(other, (int, float)):
            return Interval([interval / other for interval in self.intervals])
        else:
            raise TypeError(f"Cannot add interval with {type(other)} {other}.")

    def __invert__(self):
        compl_intervals = []
        prev_end = -np.inf
        prev_end_closed = False

        for interval in self.intervals:
            compl_intervals.append(
                _Interval(prev_end, interval.start, prev_end_closed,
                          not interval.start_closed))
            prev_end = interval.end
            prev_end_closed = not interval.end_closed

        compl_intervals.append(
            _Interval(prev_end, np.inf, prev_end_closed, False))
        return Interval(compl_intervals)

    def __and__(self, other):
        intersections = []
        for i in self.intervals:
            for j in other.intervals:
                intersection = i.intersection(j)
                if intersection is not None:
                    intersections.append(intersection)
        return Interval(intersections)

    def __or__(self, other):
        union = sorted(self.intervals + other.intervals)
        merged = [union[0]]
        for current in union:
            last = merged[-1]
            if current.start < last.end or (current.start == last.end and
                                            (last.end_closed
                                             or current.start_closed)):
                if current.end > last.end or (current.end == last.end
                                              and current.end_closed):
                    last.end = current.end
                    last.end_closed = current.end_closed
            else:
                merged.append(current)
        return Interval(merged)

    def __repr__(self):
        if len(self.intervals) == 0:
            return '{}'
        elif len(self.intervals) == 1:
            return str(self.intervals[0])
        else:
            return ' U '.join([str(x) for x in self.intervals])
