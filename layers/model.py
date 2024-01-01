import numpy as np


class Model:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        t = x
        for layer in self.layers:
            t = layer.forward(t)
        return t

    def fit(self, x, y, batch_size=64, lr=0.001, n_epochs=5):

        for epoch in range(n_epochs):
            n_data, n_feature = x.shape
            indices = np.arange(n_data)
            # shuffle
            np.random.shuffle(indices)

            for i in range(0, n_data, batch_size):
                idx = indices[i: i + batch_size]
                x_train = x[idx]
                y_train = y[idx]

                pred = self.predict(x_train)
                loss = np.sum((y_train - pred) ** 2) / len(idx)
                print("now loss = ", loss, ' accuracy = ', np.sum((pred > 0.5) == y_train) / y_train.shape[0])


                L = 2 * (pred - y_train)
                for layer in self.layers[::-1]:
                    L = layer.backward(L)
                for layer in self.layers:
                    layer.update(lr)



