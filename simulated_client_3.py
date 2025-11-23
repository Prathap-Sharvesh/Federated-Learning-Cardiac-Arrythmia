import json
import time
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt

# ----------------------------
# CONFIGURATION
# ----------------------------
BROKER = "localhost"
CLIENT_ID = "client3"  # change to client2 / client3
PUB_TOPIC = f"fl/{CLIENT_ID}"
SUB_TOPIC = "fl/global_model"

# ----------------------------
# LOAD LOCAL DATA
# ----------------------------
X_local = np.load("X_client3.npy")
y_local = np.load("y_client3.npy")
print(f"📊 {CLIENT_ID}: Loaded data {X_local.shape}, labels {y_local.shape}")

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
model = None
global_model_received = False

# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_local_model(global_weights):
    """Train using global weights and return updated local weights."""
    model.set_weights(global_weights)
    model.fit(X_local, y_local, epochs=2, batch_size=32, verbose=1)
    return model.get_weights()

# ----------------------------
# MQTT CALLBACKS
# ----------------------------
def on_message(client, userdata, msg):
    global model, global_model_received

    data = json.loads(msg.payload.decode())

    model_json = data.get("model_json", None)
    global_weights = [np.array(w) for w in data["global_weights"]]

    if model is None:
        if model_json is None:
            print("❌ No model architecture received from server.")
            return
        model = tf.keras.models.model_from_json(model_json)
        print(f"🧩 {CLIENT_ID}: Model architecture created from server JSON.")

        # ✅ Compile the model before using it
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',  # 5-class classification
            metrics=['accuracy']
        )

    print(f"📥 {CLIENT_ID} received global model. Training locally...")

    updated_weights = train_local_model(global_weights)
    payload = json.dumps({"weights": [w.tolist() for w in updated_weights]})
    client.publish(PUB_TOPIC, payload)
    print(f"📤 {CLIENT_ID} sent updated weights to server.")

    global_model_received = True


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ {CLIENT_ID} connected to MQTT broker.")
        client.subscribe(SUB_TOPIC)
        print(f"🔔 Subscribed to global model topic: {SUB_TOPIC}")
    else:
        print(f"❌ Connection failed with code {rc}")

# ----------------------------
# MAIN LOOP
# ----------------------------
client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_start()

print(f"🚀 {CLIENT_ID} started and waiting for model from server...")

# Wait until model is received
while not global_model_received:
    time.sleep(2)

# Keep the client running for future rounds
while True:
    time.sleep(10)
