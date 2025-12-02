import json
import time
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import hashlib
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
BROKER = "localhost"
CLIENT_ID = "client3"  # change for client1 / client3
PUB_TOPIC = f"fl/{CLIENT_ID}"
SUB_TOPIC = "fl/global_model"

# Metrics history
client_losses = []
client_accuracies = []

def save_client_graph():
    plt.figure(figsize=(8,4))
    plt.plot(client_losses, label="Loss")
    plt.plot(client_accuracies, label="Accuracy")
    plt.title(f"Client Training Metrics - {CLIENT_ID}")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{CLIENT_ID}_metrics.png")
    plt.close()

# All participating clients (must be identical in all scripts)
ALL_CLIENTS = ["client1", "client2", "client3"]

# Secure aggregation sign config:
PAIR_SIGNS = {
    "client1": {
        "client1-client2": +1,
        "client1-client3": +1,
    },
    "client2": {
        "client1-client2": -1,
        "client2-client3": +1,
    },
    "client3": {
        "client1-client3": -1,
        "client2-client3": -1,
    },
}

# ----------------------------
# LOAD LOCAL DATA
# ----------------------------
X_local = np.load("X_client3.npy")
y_local = np.load("y_client3.npy")
print(f"📊 {CLIENT_ID}: Loaded data {X_local.shape}, labels {y_local.shape}")
n_local = len(X_local)

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
model = None
global_model_received = False
local_round = 0  # increments each time we receive a new global model


# ----------------------------
# SECURE AGGREGATION HELPERS
# ----------------------------
def derive_seed(pair_id: str, round_number: int) -> int:
    """Deterministic 32-bit seed from pair_id + round_number."""
    h = hashlib.sha256(f"{pair_id}-{round_number}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def get_my_pairs(client_id: str):
    """Compute all pair IDs involving this client, in sorted order."""
    pairs = []
    for other in ALL_CLIENTS:
        if other == client_id:
            continue
        a, b = sorted([client_id, other])
        pairs.append(f"{a}-{b}")
    return pairs


def mask_weights_for_secure_agg(weights, round_number: int):
    """
    Create mask tensor for each weight tensor such that masks cancel in the sum.
    (Note: exact cancellation is guaranteed for simple averaging; with FedAvg
    weighting, this is an approximation.)
    """
    my_pair_signs = PAIR_SIGNS[CLIENT_ID]
    my_pairs = get_my_pairs(CLIENT_ID)

    masked_weights = []
    for layer_idx, w in enumerate(weights):
        total_mask = np.zeros_like(w, dtype=np.float32)

        for pair_id in my_pairs:
            sign = my_pair_signs[pair_id]
            seed = derive_seed(pair_id, round_number)
            rng = np.random.default_rng(seed)
            mask = rng.normal(loc=0.0, scale=1e-3, size=w.shape).astype(np.float32)
            total_mask += sign * mask

        masked_weights.append((w + total_mask).astype(np.float32))

    return masked_weights


# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_local_model(global_weights):
    """Train using global weights and return masked local weights."""
    global local_round

    # Start from global weights
    model.set_weights(global_weights)

    print(f"🧪 {CLIENT_ID}: Training on local data (round {local_round})...")
    history = model.fit(X_local, y_local, epochs=2, batch_size=32, verbose=1)

    # Log metrics (last epoch)
    client_losses.append(history.history["loss"][-1])
    client_accuracies.append(history.history["accuracy"][-1])

    # Save/update graph
    save_client_graph()

    # True updated local weights
    updated_weights = model.get_weights()

    # Apply secure aggregation mask
    print(f"🔐 {CLIENT_ID}: Applying secure aggregation masks for round {local_round}...")
    masked_weights = mask_weights_for_secure_agg(updated_weights, local_round)

    return masked_weights


# ----------------------------
# MQTT CALLBACKS
# ----------------------------
def on_message(mqtt_client, userdata, msg):
    global model, global_model_received, local_round

    print(f"📥 {CLIENT_ID}: Received global model message.")
    data = json.loads(msg.payload.decode())

    model_json = data.get("model_json", None)
    global_weights = [np.array(w) for w in data["global_weights"]]

    # Initialize model once
    if model is None:
        if model_json is None:
            print(f"❌ {CLIENT_ID}: No model_json received; cannot build model.")
            return
        model = tf.keras.models.model_from_json(model_json)
        print(f"🧩 {CLIENT_ID}: Model architecture created from server JSON.")

        # 5-class classification model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    print(f"📥 {CLIENT_ID}: Global model received. Starting local training...")
    updated_weights = train_local_model(global_weights)

    # Increase local round AFTER finishing this round
    local_round += 1

    # Send masked weights + sample count for FedAvg
    payload = json.dumps({
        "weights": [w.tolist() for w in updated_weights],
        "n": int(n_local)
    })
    mqtt_client.publish(PUB_TOPIC, payload)
    print(f"📤 {CLIENT_ID}: Sent masked weights to server.")

    global_model_received = True


def on_connect(mqtt_client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ {CLIENT_ID} connected to MQTT broker.")
        mqtt_client.subscribe(SUB_TOPIC)
        print(f"🔔 {CLIENT_ID} subscribed to global model topic: {SUB_TOPIC}")
    else:
        print(f"❌ {CLIENT_ID} connection failed with code {rc}")


# ----------------------------
# MAIN LOOP
# ----------------------------
client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()

print(f"🚀 {CLIENT_ID} started and waiting for model from server...")

# Wait until first model is received
while not global_model_received:
    time.sleep(2)

# Keep client alive for subsequent rounds
while True:
    time.sleep(10)
