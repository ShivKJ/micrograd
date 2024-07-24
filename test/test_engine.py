import pytest
import torch

from micrograd.engine import Value

T = Value | torch.Tensor


def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def generate1(a: float, b: float) -> tuple:
    def evaluate(A: T, B: T) -> tuple[T, T, T]:
        c = A + B
        d = A * B + B ** 3
        c += c + 1
        c += 1 + c + (-A)
        d += d * 2 + (B + A).relu()
        d += 3 * d + (B - A).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g += 10.0 / f

        g.backward()

        return A, B, g

    return (
        evaluate(Value(a), Value(b)),
        evaluate(
            torch.tensor(a, requires_grad=True, dtype=torch.float),
            torch.tensor(b, requires_grad=True, dtype=torch.float)
        ),
    )


def generate2(a: float, b: float) -> tuple:
    def evaluate(A: T, B: T) -> tuple[T, T, T]:
        c = A + B
        c = c.tanh()

        c = ((A * c).sin()
             .relu()
             .cos()
             .sigmoid())

        c.backward()

        return A, B, c

    return (
        evaluate(Value(a), Value(b)),
        evaluate(
            torch.tensor(a, requires_grad=True, dtype=torch.float),
            torch.tensor(b, requires_grad=True, dtype=torch.float)
        ),
    )


def generate():
    return [generate1(-4., 2.), generate2(1., 2.)]


@pytest.mark.parametrize('v, t', generate())
def test_more_ops(v, t):
    amg, bmg, gmg = v
    apt, bpt, gpt = t

    tol = 1e-3

    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) <= tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) <= tol
    assert abs(bmg.grad - bpt.grad.item()) <= tol
