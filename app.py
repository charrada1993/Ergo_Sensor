from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, abort, session, redirect, url_for
from flask_socketio import SocketIO
from config import Config
from data_processor import DataProcessor
from socket_manager import socketio, register_socket_events
from report_generator import ReportGenerator
import time
import os
import csv
from functools import wraps

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'your-secret-key-here-change-in-production'  # Important for sessions
socketio.init_app(app)

# Initialize components
data_processor = DataProcessor(Config, socketio)
report_gen = ReportGenerator(Config)

# ===============================
# LOGIN DECORATOR
# ===============================

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session:
                return redirect(url_for('login'))
            if role and session['user_role'] != role:
                # Patient trying to access doctor page: redirect to dashboard
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ===============================
# LOGIN/LOGOUT ROUTES
# ===============================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        # Hardcoded credentials
        if role == 'doctor' and email == 'doctor@exemple.com' and password == 'doctor123':
            session['user_role'] = 'doctor'
            return redirect(url_for('index'))
        elif role == 'patient' and email == 'patient@exemple.com' and password == 'patient123':
            session['user_role'] = 'patient'
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials or role mismatch')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_role', None)
    return redirect(url_for('login'))

# ===============================
# PAGE ROUTES (protected)
# ===============================

@app.route('/')
@login_required()  # Any logged-in user
def index():
    return render_template('index.html')

@app.route('/system')
@login_required()
def system():
    return render_template('system.html')

@app.route('/csv-view')
@login_required(role='doctor')  # Only doctors
def csv_view():
    return render_template('csv_view.html')

@app.route('/reports')
@login_required(role='doctor')
def reports():
    return render_template('reports.html')

@app.route('/view-csv')
@login_required(role='doctor')
def view_csv():
    files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)
    if not files:
        return "No CSV files yet."
    latest = os.path.join(Config.CSV_DIR, files[0])
    with open(latest, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return render_template('csv_table.html', rows=rows)

@app.route('/rula')
@login_required(role='doctor')
def rula_page():
    return render_template('rula.html')

@app.route('/reba')
@login_required(role='doctor')
def reba_page():
    return render_template('reba.html')

# ===============================
# API ROUTES (remain public for ESP32)
# ===============================

@app.route('/api/time', methods=['GET'])
def get_time():
    print("ESP32 requested TIME SYNC")
    return str(int(time.time()))

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json(silent=True)
    if not data:
        print("Invalid JSON received")
        return jsonify({'error': 'Invalid JSON'}), 400
    sensor_id = data.get('sensor_id')
    roll = data.get('roll')
    pitch = data.get('pitch')
    yaw = data.get('yaw')
    timestamp = data.get('timestamp', time.time())
    if not all([sensor_id, roll is not None, pitch is not None, yaw is not None]):
        return jsonify({'error': 'Missing fields'}), 400
    print("\n=========== ESP32 DATA RECEIVED ===========")
    print("Sensor ID:", sensor_id)
    print("Roll:", roll)
    print("Pitch:", pitch)
    print("Yaw:", yaw)
    print("Timestamp:", timestamp)
    print("===========================================\n")
    data_processor.process_incoming(sensor_id, roll, pitch, yaw, timestamp)
    return jsonify({'status': 'ok'}), 200

@app.route('/api/sensors', methods=['GET'])
def get_sensors_status():
    return jsonify(data_processor.get_sensor_status())

# ===============================
# CSV MANAGEMENT (protected for doctors only)
# ===============================

@app.route('/api/csv/list', methods=['GET'])
@login_required(role='doctor')
def list_csv():
    files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)
    return jsonify(files)

@app.route('/api/csv/delete/<filename>', methods=['DELETE'])
@login_required(role='doctor')
def delete_csv(filename):
    if '..' in filename or filename.startswith('/'):
        abort(400)
    current_log = data_processor.get_current_log_filename()
    if current_log and filename == current_log:
        return jsonify({'error': 'Cannot delete active log file'}), 403
    filepath = os.path.join(Config.CSV_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({'status': 'ok'}), 200
        except PermissionError:
            return jsonify({'error': 'File in use'}), 409
    else:
        abort(404)

@app.route('/api/csv/latest', methods=['GET'])
@login_required(role='doctor')
def get_latest_csv():
    files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)
    if not files:
        return jsonify({'error': 'No CSV files'}), 404
    return send_from_directory(Config.CSV_DIR, files[0], as_attachment=True)

@app.route('/api/csv/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_csv(filename):
    return send_from_directory(Config.CSV_DIR, filename, as_attachment=True)

# ===============================
# REPORT MANAGEMENT (protected for doctors only)
# ===============================

@app.route('/api/reports/list', methods=['GET'])
@login_required(role='doctor')
def list_reports():
    files = [f for f in os.listdir(Config.REPORTS_DIR) if f.endswith('.pdf')]
    files.sort(reverse=True)
    return jsonify(files)

@app.route('/api/reports/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_report(filename):
    return send_from_directory(Config.REPORTS_DIR, filename, as_attachment=True)

@app.route('/api/report/generate', methods=['POST'])
@login_required(role='doctor')
def generate_report():
    csv_dir = Config.CSV_DIR
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    files.sort(reverse=True)
    if not files:
        return jsonify({'error': 'No data available'}), 404
    pdf_file = report_gen.generate(os.path.join(csv_dir, files[0]))
    return send_file(pdf_file, as_attachment=True)

# ===============================
# SERVER START
# ===============================

if __name__ == '__main__':
    register_socket_events(socketio)
    print("\n=======================================")
    print("   MSD SENTINEL SERVER STARTED")
    print("=======================================")
    print("Server Address: http://192.168.1.217:5000")
    print("")
    print("ESP32 API Endpoint: http://192.168.1.217:5000/api/data")
    print("Time Sync Endpoint: http://192.168.1.217:5000/api/time")
    print("=======================================\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)