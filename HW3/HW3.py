import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import EuroSAT
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
NUM_CLASSES = 10
DATA_ROOT = "./data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Dataset + loaders
# -----------------------------
def get_loaders():
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full = EuroSAT(root=DATA_ROOT, download=True)
    class_names = full.classes

    N = len(full)
    train_size = int(0.7 * N)
    val_size = int(0.15 * N)
    test_size = N - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Attach transforms after split
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform = eval_tfms
    test_ds.dataset.transform = eval_tfms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_names


# -----------------------------
# Model builders
# -----------------------------
def build_linear_probe_model(num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=True)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)  # trainable head

    return model


def build_scratch_model(num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# -----------------------------
# Train / eval helpers
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS, tag="run"):
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"[{tag}] Epoch {epoch:02d}/{epochs} | "
              f"train acc {train_acc:.4f} loss {train_loss:.4f} | "
              f"val acc {val_acc:.4f} loss {val_loss:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    return model, history


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)


def plot_accuracy(history, title, out_path):
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    train_loader, val_loader, test_loader, class_names = get_loaders()
    criterion = nn.CrossEntropyLoss()

    results = {}

    # ---- Part 1: Linear probing ----
    lp_model = build_linear_probe_model(NUM_CLASSES)
    lp_optimizer = optim.Adam(lp_model.fc.parameters(), lr=LR)  # ONLY head params
    lp_model, lp_hist = train_model(
        lp_model, train_loader, val_loader,
        lp_optimizer, criterion,
        epochs=EPOCHS, tag="LinearProbe"
    )
    plot_accuracy(lp_hist, "Linear Probing: Accuracy vs Epochs", os.path.join(PLOTS_DIR, "acc_linear_probe.png"))

    y_true, y_pred = predict_all(lp_model, test_loader)
    lp_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    lp_acc = (y_true == y_pred).mean()

    print("\n=== Linear Probing Test Results ===")
    print(f"Test accuracy: {lp_acc:.4f}")
    print(lp_report)

    results["linear_probe"] = {"test_acc": float(lp_acc), "report": lp_report}

    # ---- Part 2: From scratch ----
    sc_model = build_scratch_model(NUM_CLASSES)
    sc_optimizer = optim.Adam(sc_model.parameters(), lr=LR)  # all params
    sc_model, sc_hist = train_model(
        sc_model, train_loader, val_loader,
        sc_optimizer, criterion,
        epochs=EPOCHS, tag="Scratch"
    )
    plot_accuracy(sc_hist, "From Scratch: Accuracy vs Epochs", os.path.join(PLOTS_DIR, "acc_scratch.png"))

    y_true, y_pred = predict_all(sc_model, test_loader)
    sc_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    sc_acc = (y_true == y_pred).mean()

    print("\n=== Scratch Test Results ===")
    print(f"Test accuracy: {sc_acc:.4f}")
    print(sc_report)

    results["scratch"] = {"test_acc": float(sc_acc), "report": sc_report}

    # ---- Part 3: Comparison summary ----
    print("\n=== Comparison Summary ===")
    print(f"Linear Probing Test Acc: {results['linear_probe']['test_acc']:.4f}")
    print(f"Scratch       Test Acc: {results['scratch']['test_acc']:.4f}")

    # Optional: save reports to text
    with open("results.txt", "w") as f:
        f.write("=== Linear Probing ===\n")
        f.write(f"Test accuracy: {results['linear_probe']['test_acc']:.4f}\n")
        f.write(results["linear_probe"]["report"])
        f.write("\n\n=== Scratch ===\n")
        f.write(f"Test accuracy: {results['scratch']['test_acc']:.4f}\n")
        f.write(results["scratch"]["report"])
    print("\nSaved: plots/acc_linear_probe.png, plots/acc_scratch.png, results.txt")


if __name__ == "__main__":
    main()


