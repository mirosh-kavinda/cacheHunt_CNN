import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Enhanced Cache Simulator (integrating concepts from BarnOwl)
class CacheSimulator:
    def __init__(self):
        self.cache = {}

    def flush(self, address):
        """Simulate flushing a memory address from the cache"""
        if address in self.cache:
            del self.cache[address]

    def reload(self, address):
        """Simulate reloading a memory address and measuring access time"""
        start = time.time()
        if address in self.cache:
            access_time = np.random.normal(loc=0.1, scale=0.02)  # Cache hit
        else:
            access_time = np.random.normal(loc=0.5, scale=0.05)  # Cache miss
            self.cache[address] = True
        end = time.time()
        total_time = end - start + access_time
        return total_time

    def simulate_flush_reload_attack(self, address, victim_access=False):
        """Simulate the flush-reload attack process"""
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
        model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(3, 1)))  # Adjust kernel size
        # Removed MaxPooling1D layer
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the data scaler used during training
scaler = StandardScaler()

def detect_attack(model, simulated_data):
    """Detect if cache access patterns indicate an attack using the trained model"""
    scaled_data = scaler.transform(simulated_data)
    predictions = model.predict(scaled_data.reshape(-1, 3, 1))  # Reshape for CNN input
    return predictions

# Simulate cache behavior using BarnOwl-inspired detection
simulator = CacheSimulator()
address = "0xABCDEF"
data = []

# Simulate normal cache access patterns (no victim access)
for _ in range(500):
    access_time = simulator.simulate_flush_reload_attack(address, victim_access=False)
    cache_hit_ratio = random.uniform(0.7, 0.9)
    cache_miss_ratio = 1 - cache_hit_ratio
    data.append([access_time, cache_hit_ratio, cache_miss_ratio])

# Simulate attack cache access patterns (victim accessed cache)
for _ in range(500):
    access_time = simulator.simulate_flush_reload_attack(address, victim_access=True)
    cache_hit_ratio = random.uniform(0.3, 0.5)
    cache_miss_ratio = 1 - cache_hit_ratio
    data.append([access_time, cache_hit_ratio, cache_miss_ratio])

# Convert data to numpy array
simulated_data = np.array(data)

# Split the data into training and testing sets
train_data, test_data = train_test_split(simulated_data, test_size=0.2, random_state=42)

# Fit the scaler on the training data
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Load or create the model
model = build_or_load_model()

# Train and evaluate the model with new data over multiple epochs
for epoch in range(5):
    print(f"--- Epoch {epoch + 1} ---")

    # Re-simulate attack scenarios for each epoch
    for _ in range(500):
        victim_access = random.choice([True, False])
        access_time = simulator.simulate_flush_reload_attack(address, victim_access=victim_access)
        cache_hit_ratio = random.uniform(0.3, 0.9)
        cache_miss_ratio = 1 - cache_hit_ratio
        data.append([access_time, cache_hit_ratio, cache_miss_ratio])

    # Convert new data and re-split
    simulated_data = np.array(data)
    train_data, test_data = train_test_split(simulated_data, test_size=0.2, random_state=42)
    
    # Update scaler and scale the data
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Incrementally train the model
    model.fit(train_data.reshape(-1, 3, 1), np.random.randint(0, 2, len(train_data)), epochs=1, verbose=1)

    # Predict attacks
    train_predictions = detect_attack(model, train_data)
    test_predictions = detect_attack(model, test_data)

    # Count the number of detected attacks
    train_attack_detected = np.sum(train_predictions > 0.5)
    test_attack_detected = np.sum(test_predictions > 0.5)
    
    # Print results for this epoch
    print(f"Number of attacks detected in training data: {train_attack_detected}")
    print(f"Number of attacks detected in testing data: {test_attack_detected}")
    
    # Save the model after each epoch
    model.save('cache_attack_model_cnn.h5')

def plot_results(history, data, success_rate):
    plt.figure(figsize=(15, 5))  # Adjusted figsize for three plots

    # Add figure title
    plt.suptitle('Cache Attack Summary Results(CNN Based)', fontsize=16)

    # Plot cache hit and miss ratios
    plt.subplot(1, 3, 1)
    plt.plot([x[1] for x in data], label='Cache Hit Ratio')
    plt.plot([x[2] for x in data], label='Cache Miss Ratio')
    plt.title('Cache Hit and Miss Ratios')
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')  # Ensure validation loss is included if present
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot success rate
    plt.subplot(1, 3, 3)
    plt.bar(['Success Rate'], [success_rate])
    plt.title(f'Success Rate: {success_rate:.2f}%')
    plt.ylim([0, 100])

    # Save plot to file and display
    plt.tight_layout()
    plt.savefig('cache_attack_summary_results.png')
    plt.show()


history = model.fit(train_data.reshape(-1, 3, 1), np.random.randint(0, 2, len(train_data)), epochs=5, verbose=1)

# Final success rate
success_rate = test_attack_detected / len(test_data) * 100
print(f"Attack detection success rate: {success_rate:.2f}%")
# After training, pass the training history to the function
plot_results(history, test_data,success_rate)

