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

client_updates = []

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
# Aggregate updates
# -------------------------
def aggregate_updates():
    global client_updates, model
    if not client_updates:
        return

    print(f"📦 Aggregating {len(client_updates)} client updates...")
    new_weights = [
        np.mean(np.stack(layer_weights, axis=0), axis=0)
        for layer_weights in zip(*client_updates)
    ]
    model.set_weights(new_weights)
    model.save("global_model.h5")
    print("✅ Global model updated and saved.")
    client_updates = []

# -------------------------
# Send updated model to client
# -------------------------
def send_updated_model():
    # Convert model weights to a list and prepare the payload
    weights = model.get_weights()
    payload = json.dumps({
        "client_id": "server",
        "weights": [w.tolist() for w in weights]
    })
    
    # Publish the updated model to clients
    mqtt_client.publish("fl/global_model_update", payload)
    print("📤 Sent updated global model to client.")

# -------------------------
# MQTT callback
# -------------------------
def on_message(client, userdata, msg):
    global client_updates
    data = json.loads(msg.payload.decode())

    if "weights" in data:
        weights = [np.array(w) for w in data["weights"]]
        weights = add_dp_noise(weights, epsilon=1.0)
        client_updates.append(weights)
        print(f"📥 Received update from {data['client_id']} ({len(weights)} layers).")

        # Aggregate after receiving updates from 3 clients (adjust as needed)
        if len(client_updates) >= 3:
            aggregate_updates()
            send_updated_model()  # Send updated model back to clients

# -------------------------
# MQTT setup
# -------------------------
# Replace with your PC's local IP
BROKER_IP = "127.0.0.1"  # e.g., your PC LAN IP
MQTT_PORT = 1883

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER_IP, MQTT_PORT, keepalive=60)

mqtt_client.subscribe("fl/update")
print("🚀 Federated Server Started on PC. Waiting for client updates...")

# Start the MQTT client loop in the background
mqtt_client.loop_start()

# Periodically send the updated global model to clients
# You could use this as a mechanism to keep clients up to date with the model
try:
    while True:
        pass  # Server keeps running, receiving updates and sending the updated model
except KeyboardInterrupt:
    print("Server shutting down.")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
