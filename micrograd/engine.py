from collections import deque
from dataclasses import dataclass, field
from itertools import count
from math import cos, exp, sin
from typing import Callable, Self

ID_CREATOR = count().__next__


def default_backward_fun():
    pass


@dataclass
class Value:
    id: int = field(init=False, default_factory=ID_CREATOR)
    data: float
    parents: tuple[Self, ...] = ()
    ops: str = ''

    grad: float = field(init=False, default=0)
    update_grad: Callable[[], None] = field(init=False)

    def __post_init__(self):
        self.update_grad = default_backward_fun

    def __add__(self, other: Self | float) -> Self:
        other = self.to_value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def update_grad():
            # for understanding grad in more detail; https://colah.github.io/posts/2015-08-Backprop/
            self.grad += out.grad
            other.grad += out.grad

        out.update_grad = update_grad

        return out

    def __mul__(self, other: Self | float) -> Self:
        other = self.to_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def update_grad():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.update_grad = update_grad

        return out

    def __pow__(self, p: int | float) -> Self:
        assert isinstance(p, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** p, (self,), f'**{p}')

        def update_grad():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out.update_grad = update_grad

        return out

    def relu(self) -> Self:
        return self.leaky_relu(0, 'ReLU')

    def leaky_relu(self, s: float = 0.01, name='LeakyRelu') -> Self:
        assert 0 <= s <= 1

        f = 1 if self.data > 0 else s

        out = Value(self.data * f, (self,), name)

        def update_grad():
            self.grad += f * out.grad

        out.update_grad = update_grad

        return out

    def sin(self) -> Self:
        out = Value(sin(self.data), (self,), 'Sin')

        def update_grad():
            self.grad += cos(self.data) * out.grad

        out.update_grad = update_grad

        return out

    def cos(self) -> Self:
        out = Value(cos(self.data), (self,), 'Cos')

        def update_grad():
            self.grad += -sin(self.data) * out.grad

        out.update_grad = update_grad

        return out

    def sigmoid(self) -> Self:
        z = 1 / (1 + exp(-self.data))

        out = Value(z, (self,), 'Sigmoid')

        def update_grad():
            self.grad += z * (1 - z) * out.grad

        out.update_grad = update_grad

        return out

    def tanh(self) -> Self:
        z = exp(self.data)
        z = (z - 1 / z) / (z + 1 / z)
        out = Value(z, (self,), 'Tanh')

        def update_grad():
            self.grad += (1 - z * z) * out.grad

        out.update_grad = update_grad

        return out

    def backward(self):
        stk = deque()
        visited_nodes = set()

        def build_topo(child: Value):
            if child not in visited_nodes:
                visited_nodes.add(child)

                for parent in child.parents:
                    build_topo(parent)

                stk.appendleft(child)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1

        for v in stk:
            v.update_grad()

    def __neg__(self) -> Self:  # -self
        return self * -1

    def __radd__(self, other: Self | float) -> Self:  # other + self
        return self + other

    def __sub__(self, other: Self | float) -> Self:  # self - other
        return self + (-other)

    def __rsub__(self, other: Self | float) -> Self:  # other - self
        return other + (-self)

    def __rmul__(self, other: Self | float) -> Self:  # other * self
        return self * other

    def __truediv__(self, other: Self | float) -> Self:  # self / other
        return self * other ** -1

    def __rtruediv__(self, other: Self | float) -> Self:  # other / self
        return other * self ** -1

    # overriding eq and hash function as required for topological sorting
    def __eq__(self, other) -> bool:
        return isinstance(other, Value) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        if self.ops:
            return f'Value(data={round(self.data, 3)}, grad={round(self.grad, 3)}, grad_fn={self.ops})'
        else:
            return f'Value(data={round(self.data, 3)}, grad={round(self.grad, 3)})'

    @staticmethod
    def to_value(x: float) -> 'Value':
        if not isinstance(x, Value):
            x = Value(x)

        return x
