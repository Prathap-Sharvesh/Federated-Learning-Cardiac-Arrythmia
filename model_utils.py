import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model():
    model = models.Sequential([
        # 🔹 SAME as your original first layer (keeps (5,1,16) weights)
        layers.Conv1D(16, 5, activation='relu', padding='same', input_shape=(187, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # 🔹 Second convolutional block
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # 🔹 Third convolutional block (new)
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # 🔹 Dense layers for classification
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(5, activation='softmax')
    ])

    # 🔹 Adam optimizer (generally improves convergence)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
