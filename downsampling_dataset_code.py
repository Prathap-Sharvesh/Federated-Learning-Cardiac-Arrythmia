import pandas as pd
import numpy as np
from sklearn.utils import shuffle, resample
from collections import Counter

# -------------------------
# Load dataset
# -------------------------
train_df = pd.read_csv("mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values.astype(int)

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

# -------------------------
# Balanced split with intermediate oversampling
# -------------------------
def split_clients_intermediate_balance(X, y, n_clients=3, balance_factor=0):
    """
    balance_factor: 0 → downsample to smallest class
                    1 → oversample to largest class
                    0.5 → intermediate target
    """
    clients_data = {f"client_{i+1}": {"X": [], "y": []} for i in range(n_clients)}
    class_counts = Counter(y)
    print("Original class distribution:", class_counts)

    max_class = max(class_counts.values())
    min_class = min(class_counts.values())
    target = int(min_class + balance_factor * (max_class - min_class))
    print(f"Target samples per class: {target}")

    classes = np.unique(y)
    X_bal, y_bal = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        cls_X, cls_y = X[idx], y[idx]
        cls_X, cls_y = shuffle(cls_X, cls_y, random_state=42)

        if len(cls_X) > target:
            # downsample large classes
            cls_X, cls_y = cls_X[:target], cls_y[:target]
        else:
            # moderate oversample smaller classes
            cls_X, cls_y = resample(cls_X, cls_y, replace=True, n_samples=target, random_state=42)
        
        X_bal.append(cls_X)
        y_bal.append(cls_y)

    X_bal = np.vstack(X_bal)
    y_bal = np.hstack(y_bal)
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)

    # -------------------------
    # Split balanced data among clients
    # -------------------------
    classes = np.unique(y_bal)
    cls_indices = {cls: np.where(y_bal == cls)[0] for cls in classes}

    for cls in classes:
        cls_X = X_bal[cls_indices[cls]]
        cls_y = y_bal[cls_indices[cls]]
        cls_splits_X = np.array_split(cls_X, n_clients)
        cls_splits_y = np.array_split(cls_y, n_clients)
        for i in range(n_clients):
            clients_data[f"client_{i+1}"]["X"].append(cls_splits_X[i])
            clients_data[f"client_{i+1}"]["y"].append(cls_splits_y[i])

    # reshape for Conv1D
    for i in range(n_clients):
        clients_data[f"client_{i+1}"]["X"] = np.vstack(clients_data[f"client_{i+1}"]["X"])[..., np.newaxis]
        clients_data[f"client_{i+1}"]["y"] = np.hstack(clients_data[f"client_{i+1}"]["y"])

    return clients_data

# -------------------------
# Create intermediate-balanced client datasets
# -------------------------
clients_data = split_clients_intermediate_balance(X, y, n_clients=3, balance_factor=0.5)

# -------------------------
# Check results
# -------------------------
for i in range(3):
    Xc, yc = clients_data[f"client_{i+1}"]["X"], clients_data[f"client_{i+1}"]["y"]
    print(f"Client {i+1}: {Xc.shape}, Class distribution: {Counter(yc)}")
