import numpy as np
from sklearn.model_selection import train_test_split

# Load your preprocessed, balanced arrays
X = np.load("ecg_train_data_reduced.npy")
y = np.load("ecg_train_labels_reduced.npy")

# # Split into 3 client datasets
num_clients = 3
split_size = len(X) // num_clients
splits = [(i * split_size, (i + 1) * split_size) for i in range(num_clients)]

for i, (start, end) in enumerate(splits, 1):
    X_part, y_part = X[start:end], y[start:end]
    np.save(f"client{i}_data_reduced.npy", X_part)
    np.save(f"client{i}_labels_reduced.npy", y_part)
    print(f"✅ Saved client{i}_data.npy with shape {X_part.shape}")
print(X)
print(y)