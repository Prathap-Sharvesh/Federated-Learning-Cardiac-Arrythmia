import pandas as pd
import numpy as np
from sklearn.utils import shuffle
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
# Balanced, mutually exclusive split among 3 clients
# -------------------------
def split_dataset_exclusive(X, y, n_clients=3):
    clients_data = {f"client_{i+1}": {"X": [], "y": []} for i in range(n_clients)}
    classes = np.unique(y)
    
    for cls in classes:
        # Get all indices of this class
        idx = np.where(y == cls)[0]
        cls_X = X[idx]
        cls_y = y[idx]
        
        # Shuffle this class subset before splitting
        cls_X, cls_y = shuffle(cls_X, cls_y, random_state=42)
        
        # Split class data into n_clients exclusive parts
        cls_splits_X = np.array_split(cls_X, n_clients)
        cls_splits_y = np.array_split(cls_y, n_clients)
        
        # Assign each chunk to a different client
        for i in range(n_clients):
            clients_data[f"client_{i+1}"]["X"].append(cls_splits_X[i])
            clients_data[f"client_{i+1}"]["y"].append(cls_splits_y[i])
    
    # Concatenate splits for each client and reshape for Conv1D
    for i in range(n_clients):
        clients_data[f"client_{i+1}"]["X"] = np.vstack(clients_data[f"client_{i+1}"]["X"])[..., np.newaxis]
        clients_data[f"client_{i+1}"]["y"] = np.hstack(clients_data[f"client_{i+1}"]["y"])
    
    return clients_data

# -------------------------
# Create client datasets
# -------------------------
clients_data = split_dataset_exclusive(X, y, n_clients=3)

X_client1, y_client1 = clients_data["client_1"]["X"], clients_data["client_1"]["y"]
X_client2, y_client2 = clients_data["client_2"]["X"], clients_data["client_2"]["y"]
X_client3, y_client3 = clients_data["client_3"]["X"], clients_data["client_3"]["y"]

# -------------------------
# Print info
# -------------------------
print("Client 1:", X_client1.shape, Counter(y_client1))
print("Client 2:", X_client2.shape, Counter(y_client2))
print("Client 3:", X_client3.shape, Counter(y_client3))

# -------------------------
# Verify mutual exclusivity
# -------------------------
# (Convert indices back to sets to ensure no overlap)
ids_1 = set(map(tuple, X_client1.reshape(X_client1.shape[0], -1)))
ids_2 = set(map(tuple, X_client2.reshape(X_client2.shape[0], -1)))
ids_3 = set(map(tuple, X_client3.reshape(X_client3.shape[0], -1)))

overlap = len(ids_1 & ids_2) + len(ids_1 & ids_3) + len(ids_2 & ids_3)
print(f"\n✅ Overlap between clients: {overlap} samples (should be 0)")
