import requests
import time
import json

BASE_URL = "http://127.0.0.1:5000"

def test_backend():
    print("Testing backend...")
    
    # 1. Test /api/time
    try:
        r = requests.get(f"{BASE_URL}/api/time")
        print(f"/api/time: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error testing /api/time: {e}")

    # 2. Test /api/data (POST)
    try:
        data = {
            "sensor_id": "UPPER_BACK",
            "roll": 10.0,
            "pitch": 5.0,
            "yaw": 0.0,
            "timestamp": time.time()
        }
        r = requests.post(f"{BASE_URL}/api/data", json=data)
        print(f"/api/data (UPPER_BACK): {r.status_code} - {r.text}")

        # Send more data to trigger processing
        for sid in ['NECK', 'R_BICEPS', 'R_FOREARM', 'R_HAND', 'L_BICEPS', 'L_FOREARM', 'L_HAND']:
            data["sensor_id"] = sid
            r = requests.post(f"{BASE_URL}/api/data", json=data)
            print(f"/api/data ({sid}): {r.status_code} - {r.text}")
            
    except Exception as e:
        print(f"Error testing /api/data: {e}")

    # 3. Test /api/sensors
    try:
        r = requests.get(f"{BASE_URL}/api/sensors")
        print(f"/api/sensors: {r.status_code} - {r.text[:100]}...")
    except Exception as e:
        print(f"Error testing /api/sensors: {e}")

    # 4. Test /api/csv/list and /api/report/generate
    try:
        r = requests.get(f"{BASE_URL}/api/csv/list")
        print(f"/api/csv/list: {r.status_code} - {r.text[:100]}")
        
        # NOTE: /api/report/generate requires roles in session normally. But wait, it uses @login_required. I will test generating it via script directly.
        from report_generator import ReportGenerator
        from config import Config
        import os
        from data_processor import DataProcessor
        from ai_engine import AIModels
        from socket_manager import socketio

        rg = ReportGenerator(Config())
        csv_dir = Config.CSV_DIR
        files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if files:
            files.sort(reverse=True)
            pdf_path = rg.generate(os.path.join(csv_dir, files[0]))
            print(f"Report generated successfully at: {pdf_path}")
        else:
            print("No CSV files to generate report from.")
    except Exception as e:
        print(f"Error testing report generation: {e}")

if __name__ == '__main__':
    # Wait for server to start
    time.sleep(2)
    test_backend()
