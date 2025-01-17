# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import math

e = math.e
# a safe zip that errors when the objects are of different length
_zip = zip


def sz(a, b):
    if len(a) != len(b):
        raise ValueError("lists must be of same length")
    return _zip(a, b)


zip = sz
# %%


class Node:
    def __init__(self):
        self.grad = None

    def backward(self, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def all_children_rts(self):
        # TODO make this implementation more efficient
        cs = []
        cs += self.children
        for child in self.children:
            cs += child.all_children_rts()
        return list(reversed(list(dict.fromkeys(reversed(cs)))))

    def backwards(self):
        children = self.all_children_rts()
        self.grad = np.ones_like(self.val)
        children.insert(0, self)
        for node in children:
            subnodes = node.children
            grads = node.backward(node.grad)
            for subnode, grad in zip(subnodes, grads):
                if subnode.grad is None:
                    subnode.grad = np.zeros_like(subnode.val, dtype=subnode.val.dtype)
                subnode.grad += grad

    def __repr__(self):
        return f"{self.val}"

    def __add__(self, other):
        return Plus(self, other)

    def __mul__(self, other):
        return Mult(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __matmul__(self, other):
        return MatMul(self, other)

    def sum(self):
        return Sum(self)

    def prod(self):
        return Prod(self)


class Matrix(Node):
    def __init__(self, val: np.ndarray):
        super().__init__()
        self.children = []
        self.val = val
        self.shape = val.shape

    def backward(self, grad_output):
        # no children, so just return
        return ()


class Plus(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.val = a.val + b.val
        self.shape = self.val.shape

    def backward(self, grad_output):
        return grad_output, grad_output


class Sub(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.val = a.val - b.val
        self.shape = self.val.shape

    def backward(self, grad_output):
        return grad_output, -grad_output


class Mult(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.saved = a.val, b.val
        self.val = self.saved[0] * self.saved[1]
        self.shape = self.val.shape

    def backward(self, grad_output):
        return grad_output * self.saved[1], grad_output * self.saved[0]


class MatMul(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.saved = a.val, b.val
        arows, acols = a.shape[-2], a.shape[-1]
        brows, bcols = b.shape[-2], b.shape[-1]
        if acols != brows:
            raise ValueError(f"invalid shapes in matmul {a.shape}, {b.shape}")
        self.val = a.val @ b.val
        self.shape = self.val.shape

    def backward(self, grad_output):
        grad_a = grad_output @ self.saved[1].T
        grad_b = self.saved[0].T @ grad_output
        return grad_a, grad_b


class Pow(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.saved = a.val, b.val
        self.val = self.saved[0] ** self.saved[1]
        self.shape = self.val.shape

    def backward(self, grad_output):
        return self.saved[1] * self.saved[0] ** (self.saved[1] - 1), np.log(
            self.saved[0]
        ) * self.val


class Sum(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.val = a.val.sum()
        self.saved = a.val
        self.shape = ()

    def backward(self, grad_output):
        return (grad_output * np.ones_like(self.saved),)


class Prod(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.val = a.val.prod()
        self.saved = a.val
        self.shape = ()

    def backward(self, grad_output):
        return (grad_output * self.saved,)


class ReLU(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = np.where(a.val > 0.0, a.val, np.zeros_like(a.val, dtype=a.val.dtype))
        self.shape = self.val.shape

    def backward(self, grad_output):
        return (grad_output * np.where(self.saved > 0.0, 1.0, 0.0),)


class Sigmoid(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = 1 / (1 + e ** (-a.val))
        self.shape = self.val.shape

    def backward(self, grad_output):
        return (grad_output * self.val * (1 - self.val),)


class Square(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = a.val**2
        self.shape = self.val.shape

    def backward(self, grad_output):
        return (grad_output * 2 * self.saved,)


class Layer:
    def __init__(self, in_features, out_features, activate=None):
        self.weight = Matrix(np.random.randn(out_features, in_features))
        self.bias = Matrix(np.zeros((out_features, 1)))
        self.activate = activate

    def __call__(self, x):
        x = self.weight @ x
        x = x + self.bias
        if self.activate is not None:
            x = self.activate(x)
        return x

    def params(self):
        return [self.weight, self.bias]


net = [
    Layer(1, 10, activate=ReLU),
    Layer(10, 1),
]
params = []
for layer in net:
    params += layer.params()


def run_net(x):
    x = Matrix(np.array([[x]]))
    for layer in net:
        x = layer(x)
    return x


def ground_truth(x):
    return np.sin(x)


def get_loss(yhat, y):
    return Square(yhat - Matrix(np.array([[y]])))


for iter in range(5001):
    n_samps = 200
    samps = np.linspace(-np.pi, np.pi, n_samps)
    yhats = [run_net(x) for x in samps]
    ys = ground_truth(samps)
    losses = [get_loss(yhat, y) for yhat, y in zip(yhats, ys)]
    l = Matrix(np.array([[0.0]]))
    for los in losses:
        l += los
    l *= Matrix(np.array([[1 / n_samps]]))
    l.backwards()
    if iter % 10 == 0:
        print("iter", iter, "loss", l.val)
    for param in params:
        param.val -= 0.001 * param.grad
        param.grad = None
    if iter % 100 == 0:
        plt.plot([y.val[0][0] for y in yhats], label="predicted")
        plt.plot(ys, label="ground truth")
        plt.show()


# %%
