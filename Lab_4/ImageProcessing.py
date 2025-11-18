import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG19, vgg19
from tensorflow.keras.applications import ResNet50, resnet50

# GPU SETUP
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[WARN] Could not set memory growth for GPU:", e)

    print(f"[INFO] Found GPU devices: {gpus}")

    try:
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Using mixed precision policy: mixed_float16")
        _USE_MIXED = True
    except Exception as e:
        print("[WARN] Could not enable mixed precision:", e)
        _USE_MIXED = False
else:
    print("[WARN] No GPU found. Running on CPU.")
    _USE_MIXED = False


# CONFIGURATION
@dataclass
class Config:
    data_dir: str = "data"
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42

    # amount of training epochs
    epochs_base: int = 12
    # amount of overfit epochs
    epochs_overfit: int = 25
    # amount of steps for overfit
    overfit_steps: int = 50

    lr: float = 1e-3
    out_dir: str = "outputs"


CFG = Config()

tf.keras.utils.set_random_seed(CFG.seed)


def ensure_outdir():
    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "logs"), exist_ok=True)


def make_datasets(img_size=(224, 224), batch_size=32, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE

    def mk(split):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(CFG.data_dir, split),
            labels="inferred",
            label_mode="binary",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=seed
        )

    train_ds = mk("train")
    val_ds = mk("val")
    test_ds = mk("test")

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


# train arguments
def build_augmenter():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ], name="augment")


# MODELS
# MLP
def build_mlp(input_shape, augment=True):
    """(a) Повнозв'язна мережа: Flatten -> Dense x3 -> Sigmoid"""
    inputs = keras.Input(shape=input_shape)
    x = layers.Resizing(*CFG.img_size)(inputs)
    x = layers.Rescaling(1. / 255)(x)
    if augment:
        x = build_augmenter()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name="MLP")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# CNN
def build_cnn(input_shape, augment=True):
    """(b) CNN: [Conv+ReLU -> Pool] x2 -> Dense -> Sigmoid"""
    inputs = keras.Input(shape=input_shape)
    x = layers.Resizing(*CFG.img_size)(inputs)
    x = layers.Rescaling(1. / 255)(x)
    if augment:
        x = build_augmenter()(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name="CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_transfer(base_name: str, input_shape, augment=True):
    """
    (c) Transfer learning: VGG19 / ResNet50 (conv-base frozen), GAP -> Dense -> Sigmoid
    To match preprocess_input:
      [0..1] -> (augment) -> *255 -> preprocess_input -> base
    """
    assert base_name in ("VGG19", "ResNet50")
    if base_name == "VGG19":
        preprocess = vgg19.preprocess_input
        base = VGG19(include_top=False, weights="imagenet", input_shape=(*CFG.img_size, 3))
    else:
        preprocess = resnet50.preprocess_input
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*CFG.img_size, 3))

    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = layers.Resizing(*CFG.img_size)(inputs)
    x = layers.Rescaling(1. / 255)(x)
    if augment:
        x = build_augmenter()(x)
    x = layers.Lambda(lambda t: t * 255.0)(x)
    x = layers.Lambda(preprocess, name="preprocess")(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name=f"{base_name}_TL")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# TRAIN / EVAL
def fit_model(model, train_ds, val_ds, name: str, epochs: int, early_stop=True) -> keras.callbacks.History:
    weights_path = os.path.join(CFG.out_dir, "weights", f"{name}_best.weights.h5")
    log_path = os.path.join(CFG.out_dir, "logs", f"{name}.csv")

    cbs = [
        ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        CSVLogger(log_path, append=False),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    if early_stop:
        cbs.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1))

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        verbose=2
    )
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return hist


def evaluate_model(model, test_ds, name: str) -> Dict[str, float]:
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"model": name, "test_loss": float(loss), "test_acc": float(acc)}


# PLOTS
def plot_history(hist: keras.callbacks.History, name: str, suffix: str = ""):
    h = hist.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title(f"{name} loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(h["accuracy"], label="train")
    ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title(f"{name} accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("acc")
    ax[1].legend()
    ax[1].grid(True)

    out_path = os.path.join(CFG.out_dir, "plots", f"{name}{suffix}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def maybe_take_steps(ds, steps: int):
    if steps is None:
        return ds
    it = iter(ds)
    batches = []
    try:
        for _ in range(steps):
            batches.append(next(it))
    except StopIteration:
        pass
    x = tf.data.Dataset.from_tensor_slices((tf.concat([b[0] for b in batches], axis=0),
                                            tf.concat([b[1] for b in batches], axis=0)))
    x = x.batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)
    return x


# MAIN
def main():
    ensure_outdir()
    with open(os.path.join(CFG.out_dir, "config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("Loading datasets ...")
    train_ds, val_ds, test_ds = make_datasets(CFG.img_size, CFG.batch_size, CFG.seed)
    input_shape = (CFG.img_size[0], CFG.img_size[1], 3)

    # (a) MLP
    mlp = build_mlp(input_shape, augment=True)
    print(mlp.summary())
    hist_mlp = fit_model(mlp, train_ds, val_ds, "MLP", CFG.epochs_base, early_stop=True)
    plot_history(hist_mlp, "MLP", suffix="_base")
    res_mlp = evaluate_model(mlp, test_ds, "MLP")

    # (b) CNN
    cnn = build_cnn(input_shape, augment=True)
    print(cnn.summary())
    hist_cnn = fit_model(cnn, train_ds, val_ds, "CNN", CFG.epochs_base, early_stop=True)
    plot_history(hist_cnn, "CNN", suffix="_base")
    res_cnn = evaluate_model(cnn, test_ds, "CNN")

    # (c) Transfer: VGG19 / ResNet50
    vgg = build_transfer("VGG19", input_shape, augment=True)
    print(vgg.summary())
    hist_vgg = fit_model(vgg, train_ds, val_ds, "VGG19_TL", CFG.epochs_base, early_stop=True)
    plot_history(hist_vgg, "VGG19_TL", suffix="_base")
    res_vgg = evaluate_model(vgg, test_ds, "VGG19_TL")

    resnet = build_transfer("ResNet50", input_shape, augment=True)
    print(resnet.summary())
    hist_res = fit_model(resnet, train_ds, val_ds, "ResNet50_TL", CFG.epochs_base, early_stop=True)
    plot_history(hist_res, "ResNet50_TL", suffix="_base")
    res_res = evaluate_model(resnet, test_ds, "ResNet50_TL")

    # OVERFITTING
    print("\nOverfitting demo (MLP/CNN)")
    train_small = maybe_take_steps(train_ds, CFG.overfit_steps)

    print(">>> Starting MLP_overfit")
    mlp_over = build_mlp(input_shape, augment=False)
    hist_mlp_over = fit_model(mlp_over, train_small, val_ds,
                              "MLP_overfit", CFG.epochs_overfit, early_stop=False)
    plot_history(hist_mlp_over, "MLP", suffix="_overfit")
    print(">>> Finished MLP_overfit")

    print(">>> Starting CNN_overfit")
    cnn_over = build_cnn(input_shape, augment=False)
    hist_cnn_over = fit_model(cnn_over, train_small, val_ds,
                              "CNN_overfit", CFG.epochs_overfit, early_stop=False)
    plot_history(hist_cnn_over, "CNN", suffix="_overfit")
    print(">>> Finished CNN_overfit")

    # Summary table of test accuracies (for base models)
    summary = [res_mlp, res_cnn, res_vgg, res_res]
    print("\nTest accuracy summary")
    for row in summary:
        print(f"{row['model']:14s}  acc={row['test_acc']:.4f}  loss={row['test_loss']:.4f}")

    with open(os.path.join(CFG.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


main()
