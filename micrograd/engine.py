from collections import deque
from typing import Self


class Value:
    def __init__(self, data: float, parents: tuple[Self, ...] = (), ops=''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(parents)
        self.ops = ops  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Self | float) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Self | float) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, p: int | float) -> Self:
        assert isinstance(p, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** p, (self,), f'**{p}')

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> Self:
        out = Value(max(0., self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self, dx=1):
        topo = deque()
        visited_nodes = set()

        def build_topo(node: Value):
            if node not in visited_nodes:
                visited_nodes.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = dx

        for v in reversed(topo):
            v._backward()

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

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
