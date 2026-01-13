"""
A very simple dashboard to show the sensor data.
"""
import pandas as pd

def show_dashboard():
    """Displays a simple text-based dashboard."""
    print("\n" + "="*50)
    print("        📊 SMART SENSOR ANALYZER DASHBOARD")
    print("="*50)
    
    try:
        df = pd.read_csv('sensor_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Latest reading
        latest = df.iloc[-1]
        print(f"\n🕒 Latest Reading ({latest['timestamp']}):")
        print(f"   🌡️  Temperature: {latest['temperature']}°C")
        print(f"   💧 Humidity: {latest['humidity']}%")
        print(f"   📳 Vibration: {latest['vibration']} mm/s")
        
        # Statistics
        print(f"\n📈 Statistics (last {len(df)} readings):")
        print(f"   Temperature: {df['temperature'].mean():.1f}°C avg "
              f"(min {df['temperature'].min():.1f}, max {df['temperature'].max():.1f})")
        print(f"   Humidity: {df['humidity'].mean():.1f}% avg "
              f"(min {df['humidity'].min():.1f}, max {df['humidity'].max():.1f})")
        print(f"   Vibration: {df['vibration'].mean():.2f} mm/s avg "
              f"(min {df['vibration'].min():.2f}, max {df['vibration'].max():.2f})")
        
        # Check for high vibration
        high_vib_count = len(df[df['vibration'] > 3.0])
        if high_vib_count > 0:
            print(f"\n⚠️  ALERT: {high_vib_count} high vibration readings detected!")
            print("   Check 'sensor_plots.png' for visualization.")
        else:
            print(f"\n✅ System Status: NORMAL")
            
    except FileNotFoundError:
        print("\n❌ Error: No sensor data found!")
        print("   Run 'python sensor_simulator.py' first to generate data.")
    
    print("\n" + "="*50)
    print("💡 Next steps: Run 'python data_analyzer.py' for detailed analysis")
    print("="*50)

if __name__ == "__main__":
    show_dashboard()