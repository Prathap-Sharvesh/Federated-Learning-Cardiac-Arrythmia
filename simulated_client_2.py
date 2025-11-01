import json
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
from model_utils import create_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd

# -------------------------
# MQTT Setup
# -------------------------
BROKER_IP = "127.0.0.1"  # localhost because both server & client are on same PC
MQTT_PORT = 1883
CLIENT_ID = "Sim_Client_2"

mqtt_client = mqtt.Client(CLIENT_ID)
mqtt_client.connect(BROKER_IP, MQTT_PORT)

# -------------------------
# Load dataset (small subset for simulation)
# -------------------------
train_df = pd.read_csv("mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values.astype(int)
X, y = shuffle(X, y, random_state=42)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # small local data

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = X_train[..., np.newaxis]

# -------------------------
# Load or create model
# -------------------------
try:
    global_model = tf.keras.models.load_model("global_model.h5")
    print("✅ Loaded global model.")
except:
    global_model = create_model()
    print("🆕 Created new model.")

# -------------------------
# Local training
# -------------------------
global_model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=1)  # 1 epoch per round

# -------------------------
# Send weights to server
# -------------------------
weights = [w.tolist() for w in global_model.get_weights()]
payload = json.dumps({"client_id": CLIENT_ID, "weights": weights})
mqtt_client.publish("fl/update", payload)
print("📤 Sent local update to server.")
