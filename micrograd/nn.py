import random
from abc import abstractmethod
from functools import cached_property
from operator import mul

from micrograd.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    @cached_property
    @abstractmethod
    def parameters(self) -> list[Value]:
        pass


class Neuron(Module):
    def __init__(self, nin: int, activation='ReLU'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x: list[float]) -> Value:
        act = sum(map(mul, self.w, x)) + self.b

        match self.activation:
            case None | '':
                return act
            case 'ReLU':
                return act.relu()
            case 'Sigmoid':
                return act.sigmoid()
            case 'Tanh':
                return act.tanh()
            case _:
                raise ValueError(f'activation function={self.activation} is not recognised')

    @cached_property
    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.activation else 'Linear'}-Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x: list[float]):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    @cached_property
    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in: int, n_outs: list[int]):
        sz = [n_in] + n_outs
        last_layer_index = len(n_outs) - 1

        self.layers = [Layer(sz[i], sz[i + 1], activation='ReLU' if i != last_layer_index else None)
                       for i in range(len(n_outs))]

    def __call__(self, x: list[float]) -> list[float]:
        for layer in self.layers:
            x = layer(x)
        return x

    @cached_property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
