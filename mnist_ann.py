"""
A simple Artificial Neural Network (ANN) for MNIST handwritten digits,
written from scratch with NumPy only (no deep-learning framework).

Network:  784 inputs -> 256 (ReLU) -> 128 (ReLU) -> 10 (softmax)
Training: mini-batch gradient descent with the Adam optimizer,
          cross-entropy loss, small L2 weight decay.

Run:
    pip install -r requirements.txt
    python mnist_ann.py
"""

import gzip
import json
import os
import time
import urllib.request

import numpy as np

# --------------------------------------------------------------------------
# Settings (feel free to change these and re-run)
# --------------------------------------------------------------------------
LAYER_SIZES = [784, 256, 128, 10]
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 5000
SEED = 42

DATA_DIR = "data"
RESULTS_DIR = "results"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]


# --------------------------------------------------------------------------
# 1. Data: download and read the MNIST files
# --------------------------------------------------------------------------
def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            continue
        for mirror in MIRRORS:
            try:
                print(f"Downloading {filename} from {mirror} ...")
                urllib.request.urlretrieve(mirror + filename, path)
                break
            except Exception as err:  # try the next mirror
                print(f"  failed ({err})")
        else:
            raise RuntimeError(f"Could not download {filename}")


def read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def load_mnist():
    download_mnist()
    p = {k: os.path.join(DATA_DIR, v) for k, v in MNIST_FILES.items()}
    x_train = read_images(p["train_images"]).astype(np.float32) / 255.0
    y_train = read_labels(p["train_labels"]).astype(np.int64)
    x_test = read_images(p["test_images"]).astype(np.float32) / 255.0
    y_test = read_labels(p["test_labels"]).astype(np.int64)
    return x_train, y_train, x_test, y_test


# --------------------------------------------------------------------------
# 2. The neural network
# --------------------------------------------------------------------------
class ANN:
    def __init__(self, layer_sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation works well with ReLU
            w = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out)).astype(np.float32)
            self.weights.append(w)
            self.biases.append(np.zeros(n_out, dtype=np.float32))
        self.params = self.weights + self.biases
        # Adam optimizer state
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, x):
        """Returns class probabilities and the activations of every layer."""
        activations = [x]
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            last = i == len(self.weights) - 1
            a = self.softmax(z) if last else np.maximum(z, 0)  # ReLU
            activations.append(a)
        return a, activations

    def backward(self, activations, y):
        """Backpropagation: gradients of cross-entropy loss w.r.t. every parameter."""
        n = y.shape[0]
        grad_w, grad_b = [None] * len(self.weights), [None] * len(self.biases)
        delta = activations[-1].copy()
        delta[np.arange(n), y] -= 1  # d(loss)/d(logits) for softmax + cross-entropy
        delta /= n
        for i in reversed(range(len(self.weights))):
            grad_w[i] = activations[i].T @ delta + WEIGHT_DECAY * self.weights[i]
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)  # ReLU derivative
        return grad_w + grad_b

    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x, batch_size=2000):
        out = [self.forward(x[i:i + batch_size])[0] for i in range(0, len(x), batch_size)]
        return np.vstack(out)

    def save(self, path):
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **arrays)


def loss_and_accuracy(model, x, y):
    probs = model.predict(x)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    acc = np.mean(probs.argmax(axis=1) == y)
    return float(loss), float(acc)


# --------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------
def train():
    rng = np.random.default_rng(SEED)
    x_train, y_train, x_test, y_test = load_mnist()

    # Hold out part of the training set to check for overfitting
    idx = rng.permutation(len(x_train))
    val_idx, tr_idx = idx[:VALIDATION_SIZE], idx[VALIDATION_SIZE:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    print(f"Train: {len(x_tr)}  Validation: {len(x_val)}  Test: {len(x_test)}")

    model = ANN(LAYER_SIZES, seed=SEED)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_params = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        lr = LEARNING_RATE * (0.5 ** ((epoch - 1) // 8))  # halve the step size every 8 epochs
        order = rng.permutation(len(x_tr))
        for i in range(0, len(order), BATCH_SIZE):
            b = order[i:i + BATCH_SIZE]
            _, acts = model.forward(x_tr[b])
            model.adam_step(model.backward(acts, y_tr[b]), lr)

        tr_loss, tr_acc = loss_and_accuracy(model, x_tr, y_tr)
        va_loss, va_acc = loss_and_accuracy(model, x_val, y_val)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        if va_acc > best_val:  # keep the best model seen on validation data
            best_val = va_acc
            best_params = [p.copy() for p in model.params]
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss {tr_loss:.4f}  train acc {tr_acc:.4f}  "
              f"val loss {va_loss:.4f}  val acc {va_acc:.4f}  ({time.time() - start:.1f}s)")

    for p, best in zip(model.params, best_params):
        p[...] = best

    test_loss, test_acc = loss_and_accuracy(model, x_test, y_test)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%  (test loss {test_loss:.4f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model.save(os.path.join(RESULTS_DIR, "mnist_ann_weights.npz"))
    preds = model.predict(x_test).argmax(axis=1)
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "best_val_accuracy": round(best_val, 4),
        "epochs": EPOCHS,
        "layers": LAYER_SIZES,
        "history": history,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    make_plots(history, x_test, y_test, preds)
    return metrics


# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------
def make_plots(history, x_test, y_test, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="%")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=120)
    plt.close(fig)

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_test, preds), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted digit",
           ylabel="true digit", title="Confusion matrix (test set)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for k, ax in enumerate(axes.flat):
        ax.imshow(x_test[k].reshape(28, 28), cmap="gray")
        ok = preds[k] == y_test[k]
        ax.set_title(f"pred {preds[k]}", color="green" if ok else "red", fontsize=10)
        ax.axis("off")
    fig.suptitle("Predictions on the first 16 test images")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train()
