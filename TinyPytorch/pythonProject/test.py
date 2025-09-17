import unittest

import numpy as np

from base import Variable, square, add, mul
from config import Config, using_config

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([2]))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()

    def test_square(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        print(z.data)
        print(x.grad)
        print(y.grad)

    def test_clear(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        print(x.grad)
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        print(x.grad)

    def test_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))
        y = add(mul(a, b), c)
        y.backward()
        print(y)
        print(a.grad)
        print(b.grad)

        y = a * b + c
        print(y)

    def test_mul_np(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        print(y)
        y = 3.0 * x + 1.0
        print(y)

    def test_square_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        print(x.grad)
        print(y.data)

    def test_middle_res(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()

        print(y.grad, t.grad)
        print(x0.grad, x1.grad)

    def test_config(self):
        Config.enable_backprop = True
        x = Variable(np.ones((100, 100, 100))+1)
        y = square(square(x))
        y.backward()
        print(y.data)

        Config.enable_backprop = False
        x = Variable(np.ones((100, 100, 100))+1)
        y = square(square(x))
        print(y.data)

        with using_config('enable_backprop', False):
            x = Variable(np.array(2.0))
            y = square(x)
            # y.backward()
            print(y.data)

    def test_neg(self):
        x = Variable(np.array(2.0))
        y = -x
        print(y)

    def test_sub(self):
        x = Variable(np.array(2.0))
        y2 = x - 1.0
        y1 = 2.0 - x
        print(y1)
        print(y2)

    def test_div(self):
        x = Variable(np.array(2.0))
        y1 = x / 2.0
        y2 = 4 / x
        print(y1)
        print(y2)

    def test_pow(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        print(y)