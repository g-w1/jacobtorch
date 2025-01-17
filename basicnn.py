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


class Node:
    def __init__(self):
        self.grad = 0

    def backward(self, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        self.grad = 0

    def all_children_rts(self):
        # TODO make this implementation more efficient
        cs = []
        cs += self.children
        for child in self.children:
            cs += child.all_children_rts()
        return list(reversed(list(dict.fromkeys(reversed(cs)))))

    def backwards(self):
        children = self.all_children_rts()
        self.grad = 1
        children.insert(0, self)
        for node in children:
            subnodes = node.children
            grads = node.backward(node.grad)
            for subnode, grad in zip(subnodes, grads):
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


class Num(Node):
    def __init__(self, val: int):
        super().__init__()
        self.children = []
        self.val = val

    def backward(self, grad_output):
        # no inputs, so just return
        return ()


class Plus(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.val = a.val + b.val

    def backward(self, grad_output):
        return grad_output, grad_output


class Sub(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.val = a.val - b.val

    def backward(self, grad_output):
        return grad_output, -grad_output


class Mult(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.saved = a.val, b.val
        self.val = self.saved[0] * self.saved[1]

    def backward(self, grad_output):
        return grad_output * self.saved[1], grad_output * self.saved[0]


class Pow(Node):
    def __init__(self, a, b):
        super().__init__()
        self.children = [a, b]
        self.saved = a.val, b.val
        self.val = self.saved[0] ** self.saved[1]

    def backward(self, grad_output):
        return self.saved[1] * self.saved[0] ** (self.saved[1] - 1), np.log(
            self.saved[0]
        ) * self.val


class ReLU(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = max(a.val, 0)

    def backward(self, grad_output):
        return (grad_output * float(self.saved > 0),)


class Sigmoid(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = 1 / (1 + e ** (-a.val))

    def backward(self, grad_output):
        return (grad_output * self.val * (1 - self.val),)


class Square(Node):
    def __init__(self, a):
        super().__init__()
        self.children = [a]
        self.saved = a.val
        self.val = a.val**2

    def backward(self, grad_output):
        return (grad_output * 2 * self.saved,)


class Matrix:
    def __init__(self, vals):
        self.vals = [[Num(x) for x in y] for y in vals]

    @staticmethod
    def from_array_of_nodes(ar_o_nodes):
        m = Matrix([])
        m.vals = ar_o_nodes
        return m

    @staticmethod
    def placeholder_of_size(rows, cols):
        m = Matrix([])
        m.vals = [list(range(cols)) for _ in range(rows)]
        return m

    def mul(self, other):
        r1, c1 = len(self.vals), len(self.vals[0])
        r2, c2 = len(other.vals), len(other.vals[0])
        assert c1 == r2, f"Invalid matmul shapes: {r1, c1}, {r2, c2}"
        res = Matrix.placeholder_of_size(r1, c2)
        slots = res.vals
        for i in range(r1):
            for j in range(c2):
                slots[i][j] = self.vals[i][0] * other.vals[0][j]
                for k in range(1, c1):
                    slots[i][j] += self.vals[i][k] * other.vals[k][j]
        return res

    def __add__(self, other):
        ar_o_nodes = [
            [x1 + y2 for x1, y2 in zip(y1, y2)] for y1, y2 in zip(self.vals, other.vals)
        ]
        return Matrix.from_array_of_nodes(ar_o_nodes)

    def __repr__(self):
        return f"Matrix({self.vals})"

    def apply(self, unop):
        return Matrix.from_array_of_nodes([[unop(x) for x in y] for y in self.vals])

    @property
    def shape(self):
        return len(self.vals), len(self.vals[0])

    def sum(self):
        s = Num(0)
        for row in self.vals:
            for elem in row:
                s = s + elem
        return s

    def params(self):
        ps = []
        for row in self.vals:
            for elem in row:
                ps.append(elem)
        return ps


class Layer:
    def __init__(self, in_neurons, out_neurons, activate=True):
        self.weight = Matrix(
            [
                [random.gauss(mu=0, sigma=1) for _ in range(in_neurons)]
                for _ in range(out_neurons)
            ]
        )
        self.bias = Matrix([[0] for _ in range(out_neurons)])
        self.activate = activate

    def __call__(self, x):
        x = self.weight.mul(x) + self.bias
        if self.activate:
            return x.apply(Sigmoid)
        else:
            return x

    def params(self):
        return self.weight.params() + self.bias.params()


net = [
    Layer(1, 50, activate=True),
    Layer(50, 1, activate=False),
]

params = net[0].params() + net[1].params()


def run_net(n):
    x = Matrix([[n]])
    for layer in net:
        x = layer(x)
    return x


def get_loss(input, expected):
    y = Num(expected)
    yhat = run_net(input)
    loss = Square(yhat.sum() - y)
    return loss


def ground_truth(x):
    return np.sin(1.5 * x)


iters = 1_001
for iter in range(iters):
    l = Num(0)
    low = -math.pi
    high = math.pi
    n_samps = 200
    for i in np.linspace(low, high, num=n_samps):
        l += get_loss(i, ground_truth(i))
    l *= Num(1 / n_samps)
    print(f"{iter=} loss={l.val}")
    l.backwards()
    for param in params:
        lr = 0.010
        param.val -= lr * param.grad
    for param in params:
        param.grad = 0
    if iter % 100 == 0:
        inp = np.linspace(low, high, num=n_samps)
        vals = [run_net(i).sum().val for i in inp]
        plt.plot(inp, ground_truth(inp), label="ground truth")
        plt.plot(inp, vals, label="predicted")
        plt.legend()
        plt.show()

# %%
