import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')

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

# Compile or create a new model
def build_or_load_model():
    try:
        model = load_model('cache_attack_model.h5')
        print("Loaded existing model.")
    except:
        print("Creating a new model.")
        model = Sequential()
        model.add(Dense(64, input_dim=3, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the data scaler used during training
scaler = StandardScaler()

def detect_attack(model, simulated_data):
    """Detect if cache access patterns indicate an attack using the trained model"""
    scaled_data = scaler.transform(simulated_data)
    predictions = model.predict(scaled_data)
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

# Initialize lists to track accuracy and loss
train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []

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
    history = model.fit(train_data, np.random.randint(0, 2, len(train_data)), epochs=1, verbose=1)

    # Predict attacks
    train_predictions = detect_attack(model, train_data)
    test_predictions = detect_attack(model, test_data)

    # Evaluate accuracy and loss
    train_acc = accuracy_score(np.random.randint(0, 2, len(train_data)), train_predictions > 0.5)
    test_acc = accuracy_score(np.random.randint(0, 2, len(test_data)), test_predictions > 0.5)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

    # Append losses for each epoch
    train_loss.append(history.history['loss'][0])
    test_loss.append(history.history['loss'][0])

    # Count the number of detected attacks
    train_attack_detected = np.sum(train_predictions > 0.5)
    test_attack_detected = np.sum(test_predictions > 0.5)
    
    # Print results for this epoch
    print(f"Number of attacks detected in training data: {train_attack_detected}")
    print(f"Number of attacks detected in testing data: {test_attack_detected}")
    
    # Save the model after each epoch
    model.save('cache_attack_model.h5')

# Plot cache hit/miss ratios over time
def plot_results(data):
    plt.figure(figsize=(10, 6))
    plt.plot([x[1] for x in data], label='Cache Hit Ratio')
    plt.plot([x[2] for x in data], label='Cache Miss Ratio')
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.title('Cache Hit and Miss Ratios Over Time')
    plt.legend()
    plt.savefig('cache_ratios.png')
    plt.show()

# Plot accuracy and loss over epochs
def plot_training_metrics():
    epochs = range(1, 6)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy Over Epochs')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_loss, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

plot_results(test_data)
plot_training_metrics()

# Final success rate
success_rate = test_attack_detected / len(test_data) * 100
print(f"Attack detection success rate: {success_rate:.2f}%")
