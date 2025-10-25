import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data_banknote_authentication.txt"


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(y_pm1, f):
    z = y_pm1 * f
    return np.logaddexp(0.0, -z)


def logistic_grad_wrt_f(y_pm1, f):
    return -y_pm1 * (1.0 / (1.0 + np.exp(y_pm1 * f)))


def exp_loss(y_pm1, f):
    z = -y_pm1 * f
    z = np.clip(z, -50, 50)
    return np.exp(z)


def exp_grad_wrt_f(y_pm1, f):
    z = -y_pm1 * f
    z = np.clip(z, -50, 50)
    return -y_pm1 * np.exp(z)


def bce_loss(y01, p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return -(y01 * np.log(p) + (1 - y01) * np.log(1 - p))


def bce_grad_wrt_f(y01, f):
    p = sigmoid(f)
    return p - y01


def compute_loss(loss_name, f, y01=None, y_pm1=None):
    if loss_name == "logistic":
        return logistic_loss(y_pm1, f)
    elif loss_name == "exp":
        return exp_loss(y_pm1, f)
    elif loss_name == "bce":
        p = sigmoid(f)
        return bce_loss(y01, p)
    else:
        raise ValueError("Unknown loss")


def compute_grad_df(loss_name, f, y01=None, y_pm1=None):
    if loss_name == "logistic":
        return logistic_grad_wrt_f(y_pm1, f)
    elif loss_name == "exp":
        return exp_grad_wrt_f(y_pm1, f)
    elif loss_name == "bce":
        return bce_grad_wrt_f(y01, f)
    else:
        raise ValueError("Unknown loss")


def train_linear_gd(X_tr, y01_tr, ypm1_tr, X_te, y01_te, ytepm1,
                    loss_name="logistic", lr=0.05, epochs=600, l2=0.0, seed=42):
    rng = np.random.default_rng(seed)
    n_features = X_tr.shape[1]
    w = rng.normal(scale=0.01, size=n_features)
    b = 0.0

    train_losses, test_losses, test_accs = [], [], []

    for ep in range(epochs):
        f_tr = X_tr @ w + b

        if loss_name in ("logistic", "exp"):
            grad_f = compute_grad_df(loss_name, f_tr, y_pm1=ypm1_tr)
            cur_loss = compute_loss(loss_name, f_tr, y_pm1=ypm1_tr).mean()
        else:
            grad_f = compute_grad_df(loss_name, f_tr, y01=y01_tr)
            cur_loss = compute_loss(loss_name, f_tr, y01=y01_tr).mean()

        n = X_tr.shape[0]
        gw = (X_tr.T @ grad_f) / n + l2 * w
        gb = grad_f.mean()

        w -= lr * gw
        b -= lr * gb

        train_losses.append(cur_loss)

        f_te = X_te @ w + b
        if loss_name in ("logistic", "exp"):
            te_loss = compute_loss(loss_name, f_te, y_pm1=ytepm1).mean()
        else:
            te_loss = compute_loss(loss_name, f_te, y01=y01_te).mean()

        p_te = sigmoid(f_te)
        y_pred = (p_te >= 0.5).astype(int)
        acc = (y_pred == y01_te).mean()

        test_losses.append(te_loss)
        test_accs.append(acc)

    history = {
        "train_loss": np.array(train_losses),
        "test_loss": np.array(test_losses),
        "test_acc": np.array(test_accs),
    }
    return w, b, history


def main():
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y01 = df.iloc[:, -1].values.astype(int)
    y_pm1 = 2 * y01 - 1

    X_train, X_test, y_train01, y_test01, y_train_pm1, y_test_pm1 = train_test_split(
        X, y01, y_pm1, test_size=0.25, random_state=42, stratify=y01
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    configs = [
        ("Logistic loss", "logistic", y_train01, y_train_pm1, y_test01, y_test_pm1),
        ("Adaboost (exp) loss", "exp", y_train01, y_train_pm1, y_test01, y_test_pm1),
        ("Binary cross-entropy", "bce", y_train01, y_train_pm1, y_test01, y_test_pm1),
    ]

    results = {}
    for title, key, ytr01, ytrpm1, yte01, ytepm1 in configs:
        w, b, hist = train_linear_gd(
            X_train, ytr01, ytrpm1, X_test, yte01, ytepm1,
            loss_name=key, lr=0.05 if key != "exp" else 0.02, epochs=600, l2=0.000
        )
        results[key] = {"title": title, "w": w, "b": b, "hist": hist}

    for key in ["logistic", "exp", "bce"]:
        title = results[key]["title"]
        hist = results[key]["hist"]

        plt.figure()
        plt.plot(hist["train_loss"], label="train loss")
        plt.plot(hist["test_loss"], label="test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Learning curves — {title}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("Test accuracy (last epoch / best)")
    for key in ["logistic", "exp", "bce"]:
        title = results[key]["title"]
        acc_last = results[key]["hist"]["test_acc"][-1]
        acc_best = results[key]["hist"]["test_acc"].max()
        print(f"{title:22s}  last={acc_last:.4f}  best={acc_best:.4f}")


main()
