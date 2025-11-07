import json
import numpy as np
import paho.mqtt.client as mqtt
from model_utils import create_model
import tensorflow as tf
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
BROKER = "localhost"
TOPICS = ["fl/client1", "fl/client2", "fl/client3"]
AGG_TOPIC = "fl/global_model"
NUM_CLIENTS = len(TOPICS)

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
client_updates = {}
round_number = 1

# Load or initialize global model
try:
    model = tf.keras.models.load_model("global_model.h5")
    print("✅ Loaded existing global model.")
except:
    model = create_model()
    print("🆕 Created new global model.")

global_weights = model.get_weights()


# ----------------------------
# MODEL AGGREGATION
# ----------------------------

def aggregate_models():
    """Aggregate client models and evaluate the global model."""
    global round_number

    if len(client_updates) < NUM_CLIENTS:
        return None  # Wait for all clients

    print(f"\n🧮 Aggregating model weights for Round {round_number} ...")

    # Get reference global weights
    global_weights = model.get_weights()

    # Verify all clients sent weights with same structure
    first_client = list(client_updates.keys())[0]
    reference_shapes = [np.shape(w) for w in client_updates[first_client]]

    for cid, weights in client_updates.items():
        shapes = [np.shape(w) for w in weights]
        for i, (ref, s) in enumerate(zip(reference_shapes, shapes)):
            if ref != s:
                raise ValueError(
                    f"❌ Shape mismatch in client {cid}, layer {i}: expected {ref}, got {s}"
                )

    # Average layer-by-layer
    new_weights = []
    for layer_idx in range(len(global_weights)):
        try:
            layer_stack = np.stack([client_updates[c][layer_idx] for c in client_updates], axis=0)
        except ValueError as e:
            raise ValueError(f"❌ Could not stack weights for layer {layer_idx}: {e}")
        layer_mean = np.mean(layer_stack, axis=0)
        new_weights.append(layer_mean)

    # Clear old updates
    client_updates.clear()

    # Apply new global weights safely
    try:
        model.set_weights(new_weights)
    except ValueError as e:
        for i, (gw, nw) in enumerate(zip(model.get_weights(), new_weights)):
            print(f"Layer {i}: expected {gw.shape}, got {nw.shape}")
        raise ValueError(f"❌ Failed to set global weights: {e}")

    # Evaluate the global model
    try:
        X_test = np.load("ecg_test_data_reduced.npy")
        y_test = np.load("ecg_test_labels_reduced.npy")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Global Model Evaluation -> Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    except Exception as e:
        print(f"⚠️ Evaluation skipped (test data not found): {e}")

    # Save global model
    model.save("global_model.h5")
    round_number += 1

    return new_weights

# ----------------------------
# MQTT CALLBACKS
# ----------------------------
def on_message(client, userdata, msg):
    global global_weights
    topic = msg.topic
    client_name = topic.split('/')[-1]

    print(f"📩 Received model update from {client_name}")

    try:
        data = json.loads(msg.payload.decode())
        weights = [np.array(w) for w in data["weights"]]
        client_updates[client_name] = weights
    except Exception as e:
        print(f"⚠️ Error decoding weights from {client_name}: {e}")
        return

    # Aggregate when all clients have sent their updates
    agg = aggregate_models()
    if agg is not None:
        global_weights = agg
        payload = json.dumps({"global_weights": [w.tolist() for w in global_weights]})
        print("📤 Broadcasting new global model to all clients...")
        client.publish(AGG_TOPIC, payload)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Server connected to MQTT broker.")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"🔔 Subscribed to: {topic}")
    else:
        print(f"❌ Connection failed with code {rc}")


# ----------------------------
# MAIN LOOP
# ----------------------------
client = mqtt.Client("Server_Aggregator")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)

print("\n🚀 Federated ECG Server started...")
print("Waiting for client updates...\n")

client.loop_forever()
