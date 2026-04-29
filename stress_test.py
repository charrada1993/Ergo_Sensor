import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

BASE_URL = "http://127.0.0.1:5000"
# Must match EXPECTED_SENSORS in config.py exactly
SENSORS = [
    'UPPER_BACK', 'NECK',
    'R_BICEPS',   'L_BICEPS',
    'R_FOREARM',  'L_FOREARM',
    'R_HAND',     'L_HAND',
    'R_THIGH',    'L_THIGH',
    'R_SHANK',    'L_SHANK',
]

# Realistic angle ranges per sensor (roll, pitch, yaw) in degrees
SENSOR_RANGES = {
    'UPPER_BACK': ((-15, 15), (-30, 30), (-20, 20)),
    'NECK':       ((-20, 20), (-40, 40), (-30, 30)),
    'R_BICEPS':   ((-20, 20), (-90, 90), (-20, 20)),
    'L_BICEPS':   ((-20, 20), (-90, 90), (-20, 20)),
    'R_FOREARM':  ((-20, 20), (-130, 0),  (-20, 20)),
    'L_FOREARM':  ((-20, 20), (-130, 0),  (-20, 20)),
    'R_HAND':     ((-30, 30), (-45, 45), (-30, 30)),
    'L_HAND':     ((-30, 30), (-45, 45), (-30, 30)),
    'R_THIGH':    ((-10, 10), (-90, 10), (-15, 15)),
    'L_THIGH':    ((-10, 10), (-90, 10), (-15, 15)),
    'R_SHANK':    ((-10, 10), (-130, 0),  (-10, 10)),
    'L_SHANK':    ((-10, 10), (-130, 0),  (-10, 10)),
}

def send_sensor_data(sensor_id):
    """Send one IMU frame for the given sensor using anatomically realistic angle ranges."""
    roll_r, pitch_r, yaw_r = SENSOR_RANGES.get(sensor_id,
                                                ((-30, 30), (-45, 45), (-30, 30)))
    data = {
        "sensor_id": sensor_id,
        "roll":      round(random.uniform(*roll_r),  4),
        "pitch":     round(random.uniform(*pitch_r), 4),
        "yaw":       round(random.uniform(*yaw_r),   4),
        "timestamp": time.time()
    }
    try:
        r = requests.post(f"{BASE_URL}/api/data", json=data, timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False

def stress_test_data_collection(duration_seconds=15, frequency_hz=10):
    print(f"--- Starting Data Ingestion Stress Test ({duration_seconds}s at {frequency_hz}Hz per sensor) ---")
    print(f"Total expected data points: {duration_seconds * frequency_hz * len(SENSORS)}")
    start_time = time.time()
    total_requests = 0
    successful_requests = 0
    
    delay = 1.0 / frequency_hz

    while time.time() - start_time < duration_seconds:
        loop_start = time.time()
        
        # Send data for all sensors concurrently to maximize throughput
        with ThreadPoolExecutor(max_workers=len(SENSORS)) as executor:
            results = executor.map(send_sensor_data, SENSORS)
            
            for result in results:
                total_requests += 1
                if result:
                    successful_requests += 1

        elapsed = time.time() - loop_start
        sleep_time = delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Data Collection Test Complete.")
    print(f"Total Requests Sent: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    success_rate = (successful_requests/total_requests)*100 if total_requests else 0
    print(f"Success Rate: {success_rate:.2f}%\n")
    return success_rate

def test_api_responsiveness():
    print("--- Testing API Responsiveness (AI & Data Retrieval) ---")
    start = time.time()
    try:
        r = requests.get(f"{BASE_URL}/api/sensors", timeout=5)
        latency = time.time() - start
        print(f"Sensors Endpoint Latency: {latency*1000:.2f} ms")
        if r.status_code == 200:
            data = r.json()
            # The API returns a list of dictionaries with 'sensor_id', 'last_seen', 'online'
            print(f"Number of Active Sensors: {len([s for s in data if s.get('online')])}/{len(data)}")
            print(f"Sensor Status Array Length: {len(data)}")
        else:
            print(f"Failed API Response: {r.status_code}")
    except Exception as e:
        print(f"Failed to fetch sensors data: {e}")
    print()

def test_report_generation():
    print("--- Testing Report Generation Performance ---")
    # Generates PDF Report locally to benchmark the performance
    try:
        from report_generator import ReportGenerator
        from config import Config
        import os
        
        rg = ReportGenerator(Config())
        csv_dir = Config.CSV_DIR
        files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        if files:
            files.sort(reverse=True)
            latest_csv = os.path.join(csv_dir, files[0])
            file_size_kb = os.path.getsize(latest_csv) / 1024
            print(f"Found latest CSV: {latest_csv} (Size: {file_size_kb:.2f} KB)")
            
            start_time = time.time()
            pdf_path = rg.generate(latest_csv)
            generation_time = time.time() - start_time
            
            print(f"Report successfully generated in {generation_time:.2f} seconds.")
            print(f"Report File: {pdf_path}")
        else:
            print("No CSV files found to generate report. Run the system to collect data first.")
    except Exception as e:
        print(f"Failed to generate report: {e}")
    print()

if __name__ == "__main__":
    print("=====================================================")
    print("   Ergo Sensor - System Performance & Load Tester   ")
    print("=====================================================\n")
    
    # Optional parameters based on CLI arguments
    duration = 360
    freq = 10
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            freq = int(sys.argv[2])
        except ValueError:
            pass
            
    # Check if backend is actually running before stressing it
    try:
        requests.get(f"{BASE_URL}/api/time", timeout=2)
    except requests.RequestException:
        print(f"ERROR: Backend server does not appear to be running on {BASE_URL}.")
        print("Please start 'python app.py' in another terminal first.")
        sys.exit(1)

    # 1. Stress the data ingestion
    stress_test_data_collection(duration_seconds=duration, frequency_hz=freq)
    
    # Wait for the AI pipelines to catch up if needed
    time.sleep(2)
    
    # 2. Check responsiveness after load
    test_api_responsiveness()
    
    # 3. Test report generation
    test_report_generation()
    
    print("Performance testing complete.")
