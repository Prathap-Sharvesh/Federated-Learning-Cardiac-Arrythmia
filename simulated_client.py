import json
import numpy as np
import paho.mqtt.client as mqtt
from model_utils import create_model
import tensorflow as tf

# -------------------------
# Initialize global model
# -------------------------
try:
    model = tf.keras.models.load_model("global_model.h5")
    print("✅ Loaded existing global model.")
except:
    model = create_model()
    print("🆕 Created new global model.")

# -------------------------
# Load and prepare local data (client 1's portion)
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

train_df = pd.read_csv("mitbih_train.csv", header=None)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values.astype(int)
X, y = shuffle(X, y, random_state=42)

# Split data between 3 clients (client 1)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.8, random_state=42)  # Client 1 gets 1/3 of the data

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Reshape data to 3D for Conv1D (samples, timesteps, channels)
X_train = X_train[..., np.newaxis]

# Handle class imbalance by oversampling
X_train_flat = X_train.reshape(len(X_train), -1)
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train_flat, y_train)
X_res = X_res.reshape(-1, X_train.shape[1], 1)

# -------------------------
# Differential Privacy
# -------------------------
def add_dp_noise(weights, epsilon=1.0):
    noisy = []
    for w in weights:
        noise = np.random.normal(0, 1/epsilon, w.shape)
        noisy.append(w + noise)
    return noisy

# -------------------------
# MQTT callback
# -------------------------
def on_connect(client, userdata, flags, rc):
    print("Connected to server with result code " + str(rc))
    client.subscribe("fl/update")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    if "weights" in data:
        weights = [np.array(w) for w in data["weights"]]
        weights = add_dp_noise(weights, epsilon=1.0)  # Add differential privacy noise
        model.set_weights(weights)
        print(f"📥 Received updated global model from server.")

        # Train the model with local data
        model.fit(X_res, y_res, epochs=5, batch_size=32)

        # Send the updated weights back to the server
        new_weights = model.get_weights()
        update_data = {"client_id": "client_1", "weights": [w.tolist() for w in new_weights]}
        client.publish("fl/update", json.dumps(update_data))
        print("📤 Sent updated model weights to server.")

# -------------------------
# MQTT setup
# -------------------------
BROKER_IP = "127.0.0.1"  # Local broker IP
MQTT_PORT = 1883

mqtt_client = mqtt.Client("client_1")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(BROKER_IP, MQTT_PORT, keepalive=60)

print("🚀 Client 1 Started. Waiting for updates from server...")
mqtt_client.loop_forever()
