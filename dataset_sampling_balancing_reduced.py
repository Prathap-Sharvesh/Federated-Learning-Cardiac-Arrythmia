import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd

# -------------------------
# Load dataset
# -------------------------
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df  = pd.read_csv("mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.astype(int)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.astype(int)

# Shuffle for randomness
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# -------------------------
# Standardize input features
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add channel dimension (CNN expects 3D input)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# -------------------------
# Reduce and balance train data
# -------------------------
X_train_flat = X_train.reshape(len(X_train), -1)
MAX_SAMPLES_PER_CLASS = 1200  # ~7k samples total, ESP32 safe

unique, counts = np.unique(y_train, return_counts=True)
print("Original training class distribution:", dict(zip(unique, counts)))

X_balanced, y_balanced = [], []

for cls in unique:
    cls_idx = np.where(y_train == cls)[0]
    cls_X = X_train_flat[cls_idx]
    cls_y = y_train[cls_idx]

    # Resample per class
    if len(cls_idx) < MAX_SAMPLES_PER_CLASS:
        cls_X, cls_y = resample(cls_X, cls_y, replace=True,
                                n_samples=MAX_SAMPLES_PER_CLASS, random_state=42)
    else:
        cls_X, cls_y = resample(cls_X, cls_y, replace=False,
                                n_samples=MAX_SAMPLES_PER_CLASS, random_state=42)

    X_balanced.append(cls_X)
    y_balanced.append(cls_y)

X_reduced = np.vstack(X_balanced)
y_reduced = np.hstack(y_balanced)

# Reshape back to 3D
X_reduced = X_reduced.reshape(-1, X_train.shape[1], 1)
print("Reduced train shape:", X_reduced.shape)
print("New training class counts:", np.bincount(y_reduced))

# -------------------------
# Reduce and balance test data
# -------------------------
X_test_flat = X_test.reshape(len(X_test), -1)
MAX_TEST_SAMPLES_PER_CLASS = 300  # smaller balanced test set

unique_t, counts_t = np.unique(y_test, return_counts=True)
print("Original test class distribution:", dict(zip(unique_t, counts_t)))

X_test_bal, y_test_bal = [], []

for cls in unique_t:
    cls_idx = np.where(y_test == cls)[0]
    cls_X = X_test_flat[cls_idx]
    cls_y = y_test[cls_idx]

    if len(cls_idx) < MAX_TEST_SAMPLES_PER_CLASS:
        cls_X, cls_y = resample(cls_X, cls_y, replace=True,
                                n_samples=MAX_TEST_SAMPLES_PER_CLASS, random_state=42)
    else:
        cls_X, cls_y = resample(cls_X, cls_y, replace=False,
                                n_samples=MAX_TEST_SAMPLES_PER_CLASS, random_state=42)

    X_test_bal.append(cls_X)
    y_test_bal.append(cls_y)

X_test_reduced = np.vstack(X_test_bal)
y_test_reduced = np.hstack(y_test_bal)

# Reshape back to 3D
X_test_reduced = X_test_reduced.reshape(-1, X_test.shape[1], 1)
print("Reduced test shape:", X_test_reduced.shape)
print("New test class counts:", np.bincount(y_test_reduced))

# -------------------------
# Save all files
# -------------------------
np.save("ecg_train_data_reduced.npy", X_reduced)
np.save("ecg_train_labels_reduced.npy", y_reduced)
np.save("ecg_test_data_reduced.npy", X_test_reduced)
np.save("ecg_test_labels_reduced.npy", y_test_reduced)

print("\n✅ Saved all reduced .npy datasets successfully.")
