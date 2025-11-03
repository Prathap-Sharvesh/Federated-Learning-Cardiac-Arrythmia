import json
import time
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
from model_utils import create_model

BROKER = "localhost"
CLIENT_ID = "client2"      # change to client2 / client3
PUB_TOPIC = f"fl/{CLIENT_ID}"
SUB_TOPIC = "fl/global_model"

# --- Load local ECG dataset ---
X_local = np.load(f"{CLIENT_ID}_data_reduced.npy")
y_local = np.load(f"{CLIENT_ID}_labels_reduced.npy")

print(f"📊 {CLIENT_ID}: Loaded data {X_local.shape}, labels {y_local.shape}")

# --- Build model ---
model = create_model()

def train_local_model(global_weights):
    """Train using global weights and return updated local weights."""
    model.set_weights(global_weights)
    model.fit(X_local, y_local, epochs=2, batch_size=32, verbose=1)
    return model.get_weights()

def on_message(client, userdata, msg):
    """Triggered when server publishes new global model."""
    data = json.loads(msg.payload.decode())
    global_weights = [np.array(w) for w in data["global_weights"]]
    print(f"📥 {CLIENT_ID} received global model. Training locally...")
    updated_weights = train_local_model(global_weights)
    payload = json.dumps({"weights": [w.tolist() for w in updated_weights]})
    client.publish(PUB_TOPIC, payload)
    print(f"📤 {CLIENT_ID} sent updated weights to server.")

client = mqtt.Client(CLIENT_ID)
client.on_message = on_message
client.connect(BROKER, 1883)
client.subscribe(SUB_TOPIC)

print(f"✅ {CLIENT_ID} connected and waiting for global model...")
client.loop_start()

# Kickstart the first round (only client1 does this)
if CLIENT_ID == "client2":
    payload = json.dumps({"weights": [w.tolist() for w in model.get_weights()]})
    client.publish(PUB_TOPIC, payload)
    print("🚀 Initial model weights sent to server.")

while True:
    time.sleep(10)
