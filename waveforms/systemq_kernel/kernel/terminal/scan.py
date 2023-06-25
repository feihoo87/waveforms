import operator
from typing import Any

import numpy as np

from ..sched.task import App

ConstType = (int, float, complex)


class Expression():

    def __init__(self):
        self.vars = set()

    def value(self, container, **kwds):
        raise NotImplemented

    def _operation(self, op, other):
        if isinstance(other, ConstType):
            if isinstance(self, Value):
                return Value(op(self._value, other))
            else:
                return Op(op, [self, Value(other)])
        elif isinstance(other, Ref):
            ret = Op(op, [self, other])
            ret.vars = self.vars | other.vars
            return ret

    def __add__(self, other):
        return self._operation(operator.add, other)

    def __sub__(self, other):
        return self._operation(operator.sub, other)

    def __mul__(self, other):
        return self._operation(operator.mul, other)

    def __truediv__(self, other):
        return self._operation(operator.truediv, other)

    def __getitem__(self, other):
        return self._operation(operator.getitem, other)


class Op(Exception):

    def __init__(self, op, nums):
        super().__init__()
        self.op = op
        self.nums = nums

    def value(self, container, **kwds):
        return self.op(*[n.value(container, **kwds) for n in self.nums])


class Ref(Expression):

    def __init__(self, key):
        super().__init__()
        self.key = key

    def value(self, container, **kwds):
        if self.key in kwds:
            return kwds[self.key]
        return container.get(self.key)


class Value(Expression):

    def __init__(self, value):
        super().__init__()
        self._value = value

    def value(self, container, **kwds):
        return self._value


class Range(Expression):

    def __init__(self, iterable):
        super().__init__()
        self.iterable = iterable


_empty = object()


class Scan(App):

    def __new__(cls, *args, mixin=None, **kwds):
        if mixin is None:
            return super().__new__(cls)
        for k in dir(mixin):
            if not hasattr(cls, k):
                try:
                    setattr(cls, k, getattr(mixin, k))
                except:
                    pass
        return super().__new__(cls)

    def __init__(self, name, *args, mixin=None, **kwds):
        super().__init__(*args, **kwds)
        self._name = name.replace(' ', '_')
        self.functions = {}
        self.consts = {}
        self.circuit = lambda: []
        self.loops = {}
        self.mapping = {}
        self._mapping_i = 0
        self.filter = None
        self.scan_info = {'loops': {}}

    @property
    def name(self):
        return f"Scan.{self._name}"

    def add_action(self, keys, func):

        def hook(task, step_index, executor, func=func):
            params = task.runtime.prog.steps[step_index].step.kwds
            func(**{k: params[k] for k in keys})

        self._hooks.append(hook)

    def _set_circuit(self, value):
        if callable(value):
            self.circuit = value
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                self.loops['__circuit_index__'] = range(len(value))
                self.circuit = lambda __circuit_index__, __circuit_list__=value: __circuit_list__[
                    __circuit_index__]
            else:
                self.circuit = lambda __circuit__=value: __circuit__
        else:
            pass

    def _mapping(self, key, value):
        tmpkey = f"__tmp_{self._mapping_i}__"
        self._mapping_i += 1
        self[tmpkey] = value
        self.mapping[key] = tmpkey

    def __setitem__(self, key, value):
        if key == 'circuit':
            self._set_circuit(value)
        elif '.' in key:
            self._mapping(key, value)
        elif callable(value):
            self.functions[key] = value
        elif isinstance(value, Ref):
            self.functions[key] = value
        elif isinstance(value, Op):
            self.functions[key] = value
        elif isinstance(value, (np.ndarray, list)):
            self.loops[key] = value
        elif isinstance(value, Range):
            self.loops[key] = lambda: value.value(self)
        else:
            self.consts[key] = value

    def __getitem__(self, key):
        if key in self.consts:
            return self.consts[key]
        if key == 'circuit':
            return self.circuit
        return Ref(key)

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError as e:
            try:
                return self[key]
            except:
                raise e

    def assemble(self):
        variables = {}

        for k, v in self.functions.items():
            if callable(v):
                variables[k] = v
            else:
                variables[k] = lambda __v__=v, **kw: __v__.value(self, **kw)

        self.scan_info = {
            'loops': self.loops,
            'functions': {
                'circuit': self.circuit
            } | variables,
            'constants': self.consts
        }

        if self.filter is not None:
            self.scan_info['filter'] = self.filter

    def main(self):
        self.assemble()
        for step in self.scan():
            for k, v in self.mapping.items():
                self.set(k, step.kwds[v])
            self.exec(step.kwds['circuit'])

    def scan_range(self):
        return self.scan_info

    def run(self, dry_run=False):
        import kernel
        from kernel.sched.sched import generate_task_id, get_system_info
        self.runtime.prog.task_arguments = {}
        self.runtime.prog.meta_info['arguments'] = {}
        self.runtime.id = generate_task_id()
        self.runtime.user = None
        self.runtime.system_info = get_system_info()
        kernel.submit(self, dry_run=dry_run)
        self.bar()

    def plot(self,
             result=None,
             fig=None,
             axis=None,
             data='population',
             aggregation: None | dict[str, tuple[str, callable]] = None,
             find_dim: None | tuple[str, int | float | complex] = None,
             T=False,
             figsize=None,
             **kwds):
        import matplotlib.pyplot as plt
        from waveforms.visualization import autoplot

        if result is None:
            result = self.result()

        x_info = list(result['index'].items())

        plot_data = result[data]
        id = result['meta']['id']

        if find_dim:
            group_dim, group_value = find_dim
            x_info = [(k, v) for k, v in x_info if k != group_dim]
            group_axis = list(result['index'].keys()).index(group_dim)
            group_index = np.abs(result['index'][group_dim] -
                                 group_value).argmin()
            plot_data = np.moveaxis(plot_data, group_axis, 0)[group_index]

        aggregation_dims = set()
        if aggregation:
            for i, (k, v) in enumerate(x_info):
                if k in aggregation:
                    aggregation_dims.add(i)
                    aggregation[k] = (i, aggregation[k])
        aggregation_dims = sorted(aggregation_dims)
        x_info = [x for i, x in enumerate(x_info) if i not in aggregation_dims]

        num_of_qubits = plot_data.shape[-1]
        row = round(np.sqrt(num_of_qubits))
        col = num_of_qubits // row
        if row * col < num_of_qubits:
            row += 1

        if fig is None:
            if figsize is None:
                figsize = (5 * col, 4 * row)
            fig, axis = plt.subplots(row, col, figsize=figsize)
            #plt.tight_layout()
            axis = np.array([axis]).reshape(row, col)

        fig.suptitle(f"{self.name[5:]}  scan{[n[0] for n in x_info]} id: {id}")

        data_label = data

        for i in range(row):
            for j in range(col):
                n = j + i * col
                try:
                    z = plot_data[..., n]
                except:
                    axis[i, j].set_axis_off()
                    continue

                agg_count = 0
                if aggregation:
                    for k, (i, (data_label, func)) in aggregation.items():
                        z = func(z, axis=i - agg_count)
                        agg_count += 1

                if len(x_info) == 2:
                    (ylabel, y), (xlabel, x) = x_info
                    if T:
                        (ylabel, y), (xlabel, x) = (xlabel, x), (ylabel, y)
                        z = z.T
                    if i != row - 1:
                        xlabel = ''
                    if j != 0:
                        ylabel = ''
                    if j != col - 1:
                        data_label = ''
                    else:
                        data_label = data
                    autoplot(np.array(x)[:z.shape[1]],
                             np.array(y)[:z.shape[0]],
                             z,
                             ax=axis[i, j],
                             xlabel=xlabel,
                             ylabel=ylabel,
                             zlabel=data_label)
                elif len(x_info) == 1:
                    (x_label, x), = x_info
                    axis[i, j].plot(x[:len(z)], z)
                    if i == row - 1:
                        axis[i, j].set_xlabel(x_label)
                    if j == 0:
                        axis[i, j].set_ylabel(data_label)
                if 'qubits' in result and isinstance(result['qubits'], tuple):
                    axis[i, j].set_title(f"{result['qubits'][n]}")
