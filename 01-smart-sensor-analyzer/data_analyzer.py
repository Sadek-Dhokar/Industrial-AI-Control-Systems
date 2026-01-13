"""
Analyzes the sensor data and creates basic statistics.
"""
import pandas as pd
import matplotlib.pyplot as plt

def analyze_sensor_data(filename='sensor_data.csv'):
    """Loads and analyzes the sensor CSV file."""
    print(f"📖 Reading data from {filename}...")
    
    # Read the CSV file
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df)} rows of data")
    print("\n📈 Basic Statistics:")
    print(df[['temperature', 'humidity', 'vibration']].describe().round(2))
    
    # Find high vibration readings (potential anomalies)
    high_vibration = df[df['vibration'] > 3.0]
    if not high_vibration.empty:
        print(f"\n⚠️  Found {len(high_vibration)} high vibration readings (potential issues):")
        for _, row in high_vibration.head().iterrows():
            print(f"   {row['timestamp']}: Vibration = {row['vibration']}")
    else:
        print("\n✅ No high vibration readings detected.")
    
    return df

def create_visualizations(df):
    """Creates basic plots of the sensor data."""
    print("\n🎨 Creating visualizations...")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot temperature
    axes[0].plot(df['timestamp'], df['temperature'], 'b-', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Temperature Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot humidity
    axes[1].plot(df['timestamp'], df['humidity'], 'g-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Humidity (%)')
    axes[1].set_title('Humidity Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # Plot vibration with high values highlighted
    axes[2].plot(df['timestamp'], df['vibration'], 'r-', linewidth=1, alpha=0.7, label='Vibration')
    
    # Highlight high vibration points
    high_vib = df[df['vibration'] > 3.0]
    if not high_vib.empty:
        axes[2].scatter(high_vib['timestamp'], high_vib['vibration'], 
                       color='orange', s=50, zorder=5, label='High Vibration')
    
    axes[2].set_ylabel('Vibration (mm/s)')
    axes[2].set_xlabel('Timestamp')
    axes[2].set_title('Vibration Over Time (High values highlighted)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_plots.png', dpi=150, bbox_inches='tight')
    print("✅ Saved plot as 'sensor_plots.png'")
    plt.show()

if __name__ == "__main__":
    df = analyze_sensor_data()
    create_visualizations(df)
    print("\n✨ Analysis complete!")