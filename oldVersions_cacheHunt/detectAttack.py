from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Plot results of cache hits and misses over time
def plot_results(data, predictions=None):
    """Plot cache hits and misses over time, with optional predictions."""
    plt.figure(figsize=(10, 6))
    
    # Plot cache hits and misses
    plt.plot([x[0] for x in data], label='Cache Hits')
    plt.plot([x[1] for x in data], label='Cache Misses')
    
    # If predictions are provided, plot them
    if predictions is not None:
        plt.plot(predictions, label='Predicted Attacks', linestyle='--', color='red')

    plt.xlabel('Time')
    plt.ylabel('Count / Prediction')
    plt.title('Cache Hits, Misses, and Predicted Attacks Over Time')
    plt.legend()
    plt.show()

class CacheSimulator:
    def __init__(self):
        self.cache = {}

    def flush(self, address):
        """Simulate flushing a memory address from the cache"""
        if address in self.cache:
            del self.cache[address]

    def reload(self, address):
        """Simulate reloading a memory address to the cache and measuring access time"""
        start = time.time()

        # Simulate cache hit or miss
        if address in self.cache:
            access_time = np.random.normal(loc=0.1, scale=0.02)  # Cache hit
        else:
            access_time = np.random.normal(loc=0.5, scale=0.05)  # Cache miss
            self.cache[address] = True  # Reload cache

        end = time.time()
        total_time = end - start + access_time  # Include measured and simulated time
        return total_time

    def simulate_flush_reload_attack(self, address, victim_access=False):
        """Simulate the flush-reload attack process"""
        self.flush(address)
        if victim_access:
            self.cache[address] = True  # Victim accesses and reloads cache
        access_time = self.reload(address)
        return access_time

# Load the trained model
model = load_model('cache_attack_model.h5')

# Load the data scaler used during training
scaler = StandardScaler()

def detect_attack(simulated_data):
    """Detect if cache access patterns indicate an attack using the trained model"""
    scaled_data = scaler.transform(simulated_data)
    predictions = model.predict(scaled_data)
    return predictions

# Simulate cache behavior and predict attacks
simulator = CacheSimulator()
address = "0xABCDEF"
data = []

# Simulate cache behavior with and without victim access
for _ in range(500):
    access_time = simulator.simulate_flush_reload_attack(address, victim_access=False)
    hits = random.randint(500, 1000)
    misses = random.randint(50, 100)
    cache_hit_ratio = hits / (hits + misses)
    data.append([hits, misses, cache_hit_ratio])

for _ in range(500):
    access_time = simulator.simulate_flush_reload_attack(address, victim_access=True)
    hits = random.randint(100, 500)
    misses = random.randint(100, 500)
    cache_hit_ratio = hits / (hits + misses)
    data.append([hits, misses, cache_hit_ratio])

# Convert data to numpy array
simulated_data = np.array(data)

# Split data into training and testing sets
train_data, test_data = train_test_split(simulated_data, test_size=0.2, random_state=42)

# Fit the scaler on the training data
scaler.fit(train_data)

# Transform the training and testing data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Continuous evaluation over epochs
for epoch in range(5):  # Adjust number of epochs as needed
    print(f"--- Epoch {epoch + 1} ---")

    # Re-simulate attack scenarios for each epoch
    for _ in range(500):
        victim_access = random.choice([True, False])
        access_time = simulator.simulate_flush_reload_attack(address, victim_access=victim_access)
        hits = random.randint(100, 1000)
        misses = random.randint(100, 1000)
        cache_hit_ratio = hits / (hits + misses)
        data.append([hits, misses, cache_hit_ratio])

    # Convert new data and re-split
    simulated_data = np.array(data)
    train_data, test_data = train_test_split(simulated_data, test_size=0.2, random_state=42)
    
    # Update the scaler and scale the data
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Predict attacks
    train_predictions = detect_attack(train_data)
    test_predictions = detect_attack(test_data)

    # Count the number of detected attacks
    train_attack_detected = np.sum(train_predictions > 0.5)
    test_attack_detected = np.sum(test_predictions > 0.5)
    
    # Print results for this epoch
    print(f"Number of attacks detected in training data: {train_attack_detected}")
    print(f"Number of attacks detected in testing data: {test_attack_detected}")
    
    # Plot results for this epoch (Cache hits, misses, and predictions)
    plot_results(test_data, predictions=test_predictions)

# Final plot of overall results after all epochs
plot_results(test_data, predictions=test_predictions)
