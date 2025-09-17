import unittest

import numpy as np
from matplotlib import pyplot as plt

import TinyPytorch
from TinyPytorch import Variable, layers, optimizers
from TinyPytorch.functions import transpose, sin, reshape, matmul, sigmoid, mean_squared_error, get_item, \
    softmax_cross_entropy_simple, sum, softmax_simple
from TinyPytorch.layers import Layer, Linear, RNN
from TinyPytorch.models import Model, MLP, SimpleRNN


class TorchTest(unittest.TestCase):
    def test_sphere(self):
        def sphere(x, y):
            z = x ** 2 + y ** 2
            return z

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        print(x.grad, y.grad)

    def test_matyas(self):
        def matyas(x, y):
            z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
            return z
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        print(x.grad, y.grad)

    def test_second(self):
        def second(x):
            y = x ** 4 - 2 * x ** 2
            return y
        x = Variable(np.array(2.0))
        y = second(x)
        y.backward(create_graph=True)
        print(x.grad, y.grad)
        gx = x.grad
        gx.backward()
        print(x.grad, y.grad)

    def test_sin(self):
        x = Variable(np.array(1.0))
        y = sin(x)
        y.backward(create_graph=True)

        for _ in range(3):
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
            print(x.grad)

    def test_reshape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = reshape(x, (6,))
        y.backward(create_graph=True)
        print(x.grad)

    def test_transpose(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = transpose(x)
        y.backward()
        print(x)
        print(x.grad)
        print(y.grad)
        x = Variable(np.random.rand(2, 3))
        y = x.transpose()
        y = x.T
        print(x)
        print(y)
        print(y)

    def test_sum(self):
        x = Variable(np.array([1, 2, 3 ,4, 5, 6]))
        y = sum(x)
        y.backward()
        print(y)
        print(x.grad)

    def test_broadcast(self):
        x0 = Variable(np.array([1, 2, 3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        print(y)
        y.backward()
        print(x1.grad)

    def test_matmul(self):
        x = Variable(np.random.rand(2, 3))
        W = Variable(np.random.rand(3, 4))
        y = matmul(x, W)
        y.backward()
        print(x.grad.shape)
        print(W.grad.shape)

    def test_linear(self):
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = 5 + 2 * x + np.random.rand(100, 1)
        x, y = Variable(x), Variable(y)

        W = Variable(np.zeros((1, 1)))
        b = Variable(np.zeros(1))

        def predict(x):
            y = matmul(x, W) + b
            return y

        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        # Hyperparameters
        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        # Model definition
        class TwoLayerNet(Model):
            def __init__(self, hidden_size, out_size):
                super().__init__()
                self.l1 = Linear(hidden_size)
                self.l2 = Linear(out_size)

            def forward(self, x):
                y = sigmoid(self.l1(x))
                y = self.l2(y)
                return y

        model = TwoLayerNet(hidden_size, 1)

        for i in range(max_iter):
            y_pred = model(x)
            loss = mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad.data
            if i % 1000 == 0:
                print(loss)

        lr = 0.1
        iters = 100

        for i in range(iters):
            y_pred = predict(x)
            loss = mean_squared_error(y, y_pred)

            W.cleargrad()
            b.cleargrad()
            loss.backward()

            # Update .data attribute (No need grads when updating params)
            W.data -= lr * W.grad.data
            b.data -= lr * b.grad.data
            print(W, b, loss)

    def test_layer(self):
        model = Layer()
        model.l1 = layers.Linear(5)
        model.l2 = layers.Linear(3)
        def predict(model, x):
            y = model.l1(x)
            y = sigmoid(y)
            y = model.l2(y)
            return y
        for p in model.params():
            print(p)
        model.cleargrads()

    def test_model(self):
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        # Hyperparameters
        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        # Model definition
        class TwoLayerNet(Model):
            def __init__(self, hidden_size, out_size):
                super().__init__()
                self.l1 = Linear(hidden_size)
                self.l2 = Linear(out_size)

            def forward(self, x):
                y = sigmoid(self.l1(x))
                y = self.l2(y)
                return y

        model = TwoLayerNet(hidden_size, 1)

        for i in range(max_iter):
            y_pred = model(x)
            loss = mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad.data
            if i % 1000 == 0:
                print(loss)

    def test_optimizer(self):
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        model = MLP((hidden_size, 1))
        optimizer = optimizers.SGD(lr).setup(model)

        for i in range(max_iter):
            y_pred = model(x)
            loss = mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()

            optimizer.update()
            if i % 1000 == 0:
                print(loss)

    def test_get_item(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = get_item(x, 1)
        print(y)
        y.backward()
        print(x.grad)

    def test_softmax(self):
        x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
        t = np.array([2, 0, 1, 0])

        model = MLP((10, 3))

        y = model(x)
        p = softmax_simple(y)
        print(y)
        print(p)

        loss = softmax_cross_entropy_simple(y, t)
        loss.backward()
        print(loss)

    def test_mnist(self):
        train_set = TinyPytorch.datasets.MNIST(train=True, transform=True)
        test_set = TinyPytorch.datasets.MNIST(train=False, transform=True)
        print(len(train_set), len(test_set))

    def test_dropout(self):
        x = np.ones(5)
        print(x)

        # When training
        y = TinyPytorch.functions.dropout(x)
        print(y)

        # When testing (predicting)
        with TinyPytorch.test_mode():
            y = TinyPytorch.functions.dropout(x)
            print(y)

    def test_rnn(self):
        rnn = RNN(10)
        x = np.random.rand(1, 1)
        h = rnn(x)
        print(h)
        print(h.shape)

    def test_rnn_model(self):
        seq_data = [np.random.randn(1, 1) for _ in range(1000)]
        xs = seq_data[0:-1]
        ts = seq_data[1:]

        model = SimpleRNN(10, 1)

        loss, cnt = 0, 0
        for x, t in zip(xs, ts):
            y = model(x)
            loss += mean_squared_error(y, t)
            print("loss:", loss)
            cnt += 1
            if cnt == 2:
                model.cleargrads()
                loss.backward()
                break

    def test_rnn_model_train(self):
        # Hyperparameters
        max_epoch = 100
        hidden_size = 100
        bptt_length = 30

        train_set = TinyPytorch.datasets.SinCurve(train=True)
        seqlen = len(train_set)

        class SimpleRNN(Model):
            def __init__(self, hidden_size, out_size):
                super().__init__()
                self.rnn = RNN(hidden_size)
                self.fc = Linear(out_size)

            def reset_state(self):
                self.rnn.reset_state()

            def __call__(self, x):
                h = self.rnn(x)
                y = self.fc(h)
                return y

        model = SimpleRNN(hidden_size, 1)
        optimizer = TinyPytorch.optimizers.Adam().setup(model)

        # Start training.
        for epoch in range(max_epoch):
            model.reset_state()
            loss, count = 0, 0

            for x, t in train_set:
                x = x.reshape(1, 1)
                y = model(x)
                loss += mean_squared_error(y, t)
                count += 1

                if count % bptt_length == 0 or count == seqlen:
                    model.cleargrads()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()

            avg_loss = float(loss.data) / count
            print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

        # Plot
        xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
        model.reset_state()
        pred_list = []

        with TinyPytorch.no_grad():
            for x in xs:
                x = np.array(x).reshape(1, 1)
                y = model(x)
                pred_list.append(float(y.data))

        plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
        plt.plot(np.arange(len(xs)), pred_list, label='predict')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()