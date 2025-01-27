import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Enhanced Cache Simulator (integrating concepts from BarnOwl)
class CacheSimulator:
    def __init__(self):
        self.cache = {}

    def flush(self, address):
        if address in self.cache:
            del self.cache[address]

    def reload(self, address):
        start = time.time()
        if address in self.cache:
            access_time = np.random.normal(loc=0.1, scale=0.02)  # Cache hit
            cache_hit = True
        else:
            access_time = np.random.normal(loc=0.5, scale=0.05)  # Cache miss
            cache_hit = False
            self.cache[address] = True
        end = time.time()
        total_time = end - start + access_time
        return total_time, cache_hit

    def simulate_flush_reload_attack(self, address, victim_access=False):
        self.flush(address)
        if victim_access:
            self.cache[address] = True
        return self.reload(address)

# Compile or create a new CNN model
def build_or_load_model():
    try:
        model = load_model('cache_attack_model_cnn.h5')
        print("Loaded existing model.")
    except:
        print("Creating a new model.")
        model = Sequential()
        
        # Enhanced model architecture
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the data scaler used during training
scaler = MinMaxScaler()

def detect_attack(model, simulated_data):
    scaled_data = scaler.transform(simulated_data)
    predictions = model.predict(scaled_data.reshape(-1, 3, 1))  # Reshape for CNN input
    return predictions

# Simulate cache behavior using BarnOwl-inspired detection
simulator = CacheSimulator()
address = "0xABCDEF"
data = []

# Simulate balanced normal and attack cache access patterns
for _ in range(500):
    access_time, cache_hit = simulator.simulate_flush_reload_attack(address, victim_access=False)
    cache_hit_ratio = 1.0 if cache_hit else 0.0  # Cache hit ratio: 1 if hit, 0 if miss
    cache_miss_ratio = 1 - cache_hit_ratio
    data.append([access_time, cache_hit_ratio, cache_miss_ratio])

for _ in range(500):
    access_time, cache_hit = simulator.simulate_flush_reload_attack(address, victim_access=True)
    cache_hit_ratio = 1.0 if cache_hit else 0.0  # Cache hit ratio: 1 if hit, 0 if miss
    cache_miss_ratio = 1 - cache_hit_ratio
    data.append([access_time, cache_hit_ratio, cache_miss_ratio])

# Shuffle data to randomize the sequence
random.shuffle(data)
simulated_data = np.array(data)

# Split the data into training and testing sets using KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Load or create the model
model = build_or_load_model()

# Callbacks to improve training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Prepare to store results for plotting
cache_hit_ratios = []
cache_miss_ratios = []
train_losses = []
val_losses = []

# Train and evaluate the model with new data over multiple epochs
for train_index, test_index in kf.split(simulated_data):
    train_data, test_data = simulated_data[train_index], simulated_data[test_index]
    
    # Scale the data
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Incrementally train the model
    history = model.fit(train_data.reshape(-1, 3, 1), np.random.randint(0, 2, len(train_data)),
                        validation_data=(test_data.reshape(-1, 3, 1), np.random.randint(0, 2, len(test_data))),
                        epochs=100, batch_size=64, verbose=1,
                        callbacks=[early_stopping, reduce_lr])

    # Collect training and validation losses for plotting
    train_losses.extend(history.history['loss'])
    val_losses.extend(history.history['val_loss'])

    # Collect cache hit and miss ratios for plotting
    cache_hit_ratios.append(np.mean(train_data[:, 1]))  # Cache hit ratio from training data
    cache_miss_ratios.append(np.mean(train_data[:, 2]))  # Cache miss ratio from training data

    # Predict attacks on test data
    test_predictions = detect_attack(model, test_data)
    binary_predictions = (test_predictions > 0.5).astype(int)

    # Print classification report
    true_labels = np.random.randint(0, 2, len(test_data))  # Use real labels here
    print(classification_report(true_labels, binary_predictions))
    print(confusion_matrix(true_labels, binary_predictions))

    # Save the model after each fold
    model.save('cache_attack_model_cnn.h5')

# Final success rate evaluation
success_rate = np.mean(binary_predictions) * 100
print(f"Attack detection success rate: {success_rate:.2f}%")

# Plotting function is the same as before
