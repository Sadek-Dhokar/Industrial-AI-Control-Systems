"""
Simulates temperature, humidity, and vibration readings from an industrial sensor.
"""
import random
import csv
from datetime import datetime, timedelta

def create_sensor_data(num_readings=100):
    """Generates a list of simulated sensor readings."""
    readings = []
    base_time = datetime.now()
    
    for i in range(num_readings):
        # Generate somewhat realistic values
        timestamp = base_time - timedelta(minutes=i*5)
        temperature = random.uniform(20.0, 40.0)  # Celsius
        humidity = random.uniform(30.0, 70.0)     # Percentage
        vibration = random.uniform(0.1, 2.5)      # mm/s
        
        # Occasionally create a high vibration "anomaly"
        if random.random() > 0.9:  # 10% chance
            vibration = random.uniform(3.5, 5.0)
        
        readings.append({
            'timestamp': timestamp,
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'vibration': round(vibration, 2)
        })
    
    return readings

def save_to_csv(readings, filename='sensor_data.csv'):
    """Saves the sensor readings to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'temperature', 'humidity', 'vibration'])
        writer.writeheader()
        for reading in readings:
            # Convert datetime to string for CSV
            reading_copy = reading.copy()
            reading_copy['timestamp'] = reading['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow(reading_copy)
    print(f"✅ Data saved to {filename}")

if __name__ == "__main__":
    print("🔧 Generating sensor data...")
    data = create_sensor_data(200)
    print(f"📊 Created {len(data)} sensor readings")
    save_to_csv(data)
    print("🎯 Done! Open 'sensor_data.csv' to view the data.")