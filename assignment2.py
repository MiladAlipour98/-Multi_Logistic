from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def append_bias(X): return np.append(X, np.ones((X.shape[0], 1)), axis=1)


def hypothesis(w, X): return X @ w


def predict(w, X): return (sigmoid(hypothesis(w, append_bias(X))) > 0.5).astype(int)


def sigmoid(x): return 1 / (1 + np.exp(-x))


def load_susysubset(path):
    df = pd.read_csv(path)
    ys = df["Label"]
    xs = df.drop(columns=["Label"])
    return xs.to_numpy(), ys.to_numpy().reshape(-1, 1)


def split_ds(xs, ys, batch_size):
    x_batches = [xs[i: i + batch_size] for i in range(0, len(xs), batch_size)]
    y_batches = [ys[i: i + batch_size] for i in range(0, len(ys), batch_size)]
    return x_batches, y_batches


def shuffle_ds(xs, ys):
    ds = list(zip(xs, ys))
    shuffle(ds)
    xs, ys = list(zip(*ds))
    return np.array(xs), np.array(ys)


def initialize_with_zeros(n_features): return np.zeros((n_features + 1, 1))


def propagate(w, X, Y):
    X = append_bias(X)
    n_features = X.shape[1]

    A = sigmoid(hypothesis(w, X))
    cost = - (1 / n_features) * np.sum((Y.T @ np.log(A)) + ((1 - Y).T @ np.log(1 - A)))
    dw = (1 / n_features) * (X.T @ (A - Y))

    return dw, cost


def sgd_fit(xs_train, ys_train, lr=5e-3, epochs=200):
    n_features = xs_train.shape[1]
    batch_size = 32
    costs = []
    w = initialize_with_zeros(n_features)
    for i in range(epochs):
        xs_train, ys_train = shuffle_ds(xs_train, ys_train)
        xs_train_batch, ys_train_batch = split_ds(xs_train, ys_train, batch_size)
        for x, y in zip(xs_train_batch, ys_train_batch):
            dw, cost = propagate(w, x, y)
            w -= lr * dw
            costs.append(cost)

    return w, costs


def momentum_fit(xs_train, ys_train, lr=5e-5, epochs=200):
    n_features = xs_train.shape[1]
    batch_size = 32
    costs = []
    w = initialize_with_zeros(n_features)
    mu = 0.9
    v = 0
    for i in range(epochs):
        xs_train, ys_train = shuffle_ds(xs_train, ys_train)
        xs_train_batch, ys_train_batch = split_ds(xs_train, ys_train, batch_size)
        for x, y in zip(xs_train_batch, ys_train_batch):
            dw, cost = propagate(w, x, y)
            v = mu * v - lr * dw
            w += v
            costs.append(cost)

    return w, costs


def accuracy(preds, ys): return np.sum(preds == ys) / len(ys)


def main():
    xs, ys = load_susysubset("susysubset_physics.csv")
    xs = normalize(xs)
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2)

    w, _ = sgd_fit(xs_train, ys_train)
    print(accuracy(predict(w, xs_test), ys_test))

    w, _ = momentum_fit(xs_train, ys_train)
    print(accuracy(predict(w, xs_test), ys_test))


if __name__ == '__main__':
    main()
