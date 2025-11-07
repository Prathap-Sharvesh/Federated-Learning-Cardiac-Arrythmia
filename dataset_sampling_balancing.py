import json
import time
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import paho.mqtt.client as mqtt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_utils import create_model
from imblearn.over_sampling import RandomOverSampler
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df  = pd.read_csv("mitbih_test.csv", header=None)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.astype(int)

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.astype(int)

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Train shape after reshape:", X_train.shape)
print("Test shape after reshape:", X_test.shape)

# Flatten temporarily for oversampling
X_train_flat = X_train.reshape(len(X_train), -1)

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train_flat, y_train)

# Reshape back to 3D
X_res = X_res.reshape(-1, X_train.shape[1], 1)

print("After oversampling class counts:", np.bincount(y_res))

np.save("ecg_train_data.npy", X_res)
np.save("ecg_train_labels.npy", y_res)
np.save("ecg_test_data.npy", X_test)
np.save("ecg_test_labels.npy", y_test)