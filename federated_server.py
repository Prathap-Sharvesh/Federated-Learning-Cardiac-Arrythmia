import json
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
import matplotlib.pyplot as plt

from model_utils import create_model

# ----------------------------
# CONFIGURATION
# ----------------------------
BROKER = "localhost"
TOPICS = ["fl/client1", "fl/client2", "fl/client3"]
AGG_TOPIC = "fl/global_model"
NUM_CLIENTS = len(TOPICS)

# ----------------------------
# METRICS & GLOBAL STATE
# ----------------------------
server_losses = []
server_accuracies = []

client_updates = {}   # {client_id: {"weights": [...], "n": int}}
round_number = 1

def save_server_graph():
    plt.figure(figsize=(8, 4))
    plt.plot(server_losses, label="Loss")
    plt.plot(server_accuracies, label="Accuracy")
    plt.title("Server Global Model Metrics (FedAvg)")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("server_metrics.png")
    plt.close()

# Load or initialize global model
try:
    model = tf.keras.models.load_model("global_model.h5")
    print("Loaded existing global model.")
except Exception as e:
    print(f"Could not load model, creating new one. Reason: {e}")
    model = create_model()
    print("Created new global model.")

global_weights = model.get_weights()


# ----------------------------
# FEDAVG AGGREGATION
# ----------------------------
def aggregate_models_fedavg():
    """Aggregate client models using FedAvg and evaluate global model."""
    global round_number, model

    if len(client_updates) < NUM_CLIENTS:
        return None  # Wait for all clients

    print(f"\n[SERVER] Aggregating model weights for Round {round_number} ...")

    # Reference global weights to know shapes
    global_weights = model.get_weights()
    num_layers = len(global_weights)

    # Collect sample counts
    n_list = np.array([client_updates[cid]["n"] for cid in client_updates])
    total_n = np.sum(n_list).astype(float)

    print("[SERVER] Client sample counts:", {cid: client_updates[cid]["n"] for cid in client_updates})
    print("[SERVER] Total samples:", total_n)

    # FedAvg: layer-wise weighted average by n_k
    new_weights = []
    for layer_idx in range(num_layers):
        agg_layer = np.zeros_like(global_weights[layer_idx], dtype=np.float32)

        for cid, update in client_updates.items():
            w_k = update["weights"][layer_idx]   # masked weights from client
            n_k = update["n"]
            agg_layer += (n_k / total_n) * w_k

        new_weights.append(agg_layer)

    # Clear old updates
    client_updates.clear()

    # ---------------------------
    # ✅ SERVER-SIDE DP (optional)
    # ---------------------------
    DP_STD = 0.0005  # same as before
    dp_weights = []
    for w in new_weights:
        noise = np.random.normal(0, DP_STD, size=w.shape)
        dp_weights.append(w + noise)

    new_weights = dp_weights
    print(f"[SERVER] 🔐 Applied Server-Side DP Noise (std={DP_STD})")

    # Apply new global weights safely
    try:
        model.set_weights(new_weights)
    except ValueError as e:
        for i, (gw, nw) in enumerate(zip(model.get_weights(), new_weights)):
            print(f"[SERVER] Layer {i}: expected {gw.shape}, got {nw.shape}")
        raise ValueError(f"[SERVER] Failed to set global weights: {e}")

    # Evaluate the global model
    try:
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[SERVER] Global Model Evaluation -> Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

        # Log metrics
        server_losses.append(loss)
        server_accuracies.append(acc)

        # Save graph
        save_server_graph()

    except Exception as e:
        print(f"[SERVER] Evaluation skipped (test data not found): {e}")

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

    print(f"[SERVER] Received model update from {client_name}")

    try:
        data = json.loads(msg.payload.decode())
        weights = [np.array(w) for w in data["weights"]]
        n_samples = int(data["n"])
        client_updates[client_name] = {"weights": weights, "n": n_samples}
    except Exception as e:
        print(f"[SERVER] Error decoding weights from {client_name}: {e}")
        return

    # Aggregate when all clients have sent their updates
    agg = aggregate_models_fedavg()
    if agg is not None:
        global_weights = agg
        payload = json.dumps({
            "model_json": model.to_json(),                     # IMPORTANT: always send
            "global_weights": [w.tolist() for w in global_weights]
        })
        print("[SERVER] Broadcasting new global model to all clients...")
        client.publish(AGG_TOPIC, payload)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[SERVER] Server connected to MQTT broker.")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"[SERVER] Subscribed to: {topic}")
    else:
        print(f"[SERVER] Connection failed with code {rc}")


# ----------------------------
# MAIN LOOP
# ----------------------------
client = mqtt.Client("Server_Aggregator_FedAvg")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)

print("\n[SERVER] Federated ECG Server (FedAvg) started...")
print("[SERVER] Sending initial global model to all clients...")

initial_payload = json.dumps({
    "model_json": model.to_json(),
    "global_weights": [w.tolist() for w in global_weights]
})
client.publish(AGG_TOPIC, initial_payload)
print("[SERVER] Initial global model sent. Waiting for client updates...\n")

client.loop_forever()
