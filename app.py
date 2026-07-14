import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import (Flask, request, jsonify, render_template,
                   send_file, send_from_directory, abort, session, redirect, url_for)
from flask_socketio import SocketIO
from config import Config
from data_processor import DataProcessor
from socket_manager import socketio, register_socket_events
from report_generator import ReportGenerator
from firebase_listener import FirebaseListener
from ai_engine import AIModels
import time
import csv
import socket
import base64
import io
from functools import wraps

# ===============================
# APP SETUP
# ===============================

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'your-secret-key-here-change-in-production'

socketio.init_app(app)

# ===============================
# SAFE INITIALISATION (FIXED)
# ===============================

# Simple guard to prevent double loading (optional, safe with use_reloader=False)
if not hasattr(app, '_ai_models_loaded'):
    ai_models = AIModels(model_dir='models')
    print("[OK] AI Models loaded")

    data_processor = DataProcessor(Config, socketio, ai_models)
    report_gen = ReportGenerator(Config)

    firebase_listener = FirebaseListener(data_processor)
    firebase_listener.start(
        Config.FIREBASE_CREDENTIALS_PATH,
        Config.FIREBASE_DATABASE_URL
    )
    import atexit
    atexit.register(firebase_listener.stop)
    print("[OK] Firebase listener started")

    app._ai_models_loaded = True
else:
    ai_models = getattr(app, '_ai_models', None)
    data_processor = getattr(app, '_data_processor', None)
    report_gen = getattr(app, '_report_gen', None)
    firebase_listener = getattr(app, '_firebase_listener', None)

# Store references on app to avoid garbage collection
app._ai_models = ai_models
app._data_processor = data_processor
app._report_gen = report_gen
app._firebase_listener = firebase_listener

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
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ===============================
# LOGIN / LOGOUT
# ===============================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        assessment_method = request.form.get('assessment_method', 'rula')

        if role == 'doctor' and email == 'doctor@exemple.com' and password == 'doctor123':
            session['user_role'] = 'doctor'
            session['assessment_method'] = assessment_method
            if assessment_method == 'rula':
                return redirect(url_for('rula_page'))
            elif assessment_method == 'reba':
                return redirect(url_for('reba_page'))
            elif assessment_method == 'imu_posture':
                return redirect(url_for('imu_posture_page'))
            else:
                return redirect(url_for('rula_page'))

        elif role == 'patient' and email == 'patient@exemple.com' and password == 'patient123':
            session['user_role'] = 'patient'
            session['assessment_method'] = assessment_method
            return redirect(url_for('index'))

        else:
            return render_template('login.html', error='Invalid credentials or role mismatch')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_role', None)
    return redirect(url_for('login'))

# ===============================
# PAGE ROUTES
# ===============================

@app.route('/')
@login_required()
def index():
    return render_template('index.html')


@app.route('/system')
@login_required()
def system():
    return render_template('system.html')


@app.route('/sensors')
@login_required()
def sensors_page():
    return render_template('sensors.html')


@app.route('/ai')
@login_required(role='doctor')
def ai_page():
    return render_template('ai.html')


@app.route('/plots/<path:filename>')
@login_required(role='doctor')
def serve_plot(filename):
    return send_from_directory('plots', filename)


@app.route('/api/ai-metrics')
@login_required(role='doctor')
def ai_metrics():
    import json, os
    meta_path = 'models/model_metadata.json'
    if not os.path.exists(meta_path):
        return jsonify({'error': 'metrics not found'}), 404
    with open(meta_path) as f:
        meta = json.load(f)
    m = meta.get('metrics', {})
    reg = m.get('LightGBM_Regression', {})
    cls = m.get('LightGBM_Classifier', {})
    sev = m.get('LightGBM_Severity',   {})
    return jsonify({
        'version':    meta.get('version', '3.0-Production'),
        'n_features': meta.get('n_features', 75),
        'n_samples':  meta.get('n_samples', 20000),
        'created':    meta.get('created', ''),
        'regression': {
            'r2':   reg.get('r2',   0),
            'mae':  reg.get('mae',  0),
            'rmse': reg.get('rmse', 0),
        },
        'condition': {
            'accuracy':  cls.get('accuracy',  0),
            'precision': cls.get('precision', 0),
            'recall':    cls.get('recall',    0),
            'f1_macro':  cls.get('f1',        0),
        },
        'severity': {
            'accuracy': sev.get('accuracy', 0),
            'f1_macro': sev.get('f1',       0),
        },
    })


@app.route('/csv-view')
@login_required(role='doctor')
def csv_view():
    return render_template('csv_view.html')


@app.route('/reports')
@login_required(role='doctor')
def reports():
    return render_template('reports.html')


@app.route('/calibration')
@login_required()
def calibration():
    return render_template('calibration.html')


@app.route('/rula')
@login_required(role='doctor')
def rula_page():
    return render_template('rula.html')


@app.route('/reba')
@login_required(role='doctor')
def reba_page():
    return render_template('reba.html')


@app.route('/imu-posture')
@login_required(role='doctor')
def imu_posture_page():
    return render_template('imu_posture.html')


@app.route('/history')
@login_required(role='doctor')
def history():
    """Session history — shows the CSV data viewer."""
    return render_template('csv_view.html')


@app.route('/settings')
@login_required()
def settings():
    """Settings page — shows the system status page."""
    return render_template('system.html')


@app.route('/view-csv')
@login_required(role='doctor')
def view_csv():
    files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)

    if not files:
        return "No CSV files yet."

    latest = os.path.join(Config.CSV_DIR, files[0])

    with open(latest, 'r') as f:
        rows = list(csv.reader(f))

    return render_template('csv_table.html', rows=rows)

# ===============================
# AI API
# ===============================

@app.route('/api/predict', methods=['POST'])
@login_required(role='doctor')
def predict_risk():
    try:
        data = request.get_json(silent=True) or {}
        features = data.get('features', {})

        if not features:
            return jsonify({'error': 'No features provided'}), 400

        if ai_models is None:
            return jsonify({'error': 'AI not ready'}), 500

        result = ai_models.predict(features)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Prediction API Error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
@login_required()
def calibrate():
    data_processor.calibrate()
    return jsonify({'status': 'ok'}), 200


@app.route('/api/streaming/start', methods=['POST'])
@login_required()
def streaming_start():
    filename = data_processor.start_streaming()
    return jsonify({'status': 'ok', 'filename': filename}), 200


@app.route('/api/streaming/stop', methods=['POST'])
@login_required()
def streaming_stop():
    filename = data_processor.stop_streaming()
    return jsonify({'status': 'ok', 'filename': filename}), 200


@app.route('/api/streaming/status', methods=['GET'])
@login_required()
def streaming_status():
    return jsonify({
        'active': data_processor.is_streaming(),
        'filename': data_processor.get_current_log_filename()
    }), 200

# ===============================
# ESP32 API
# ===============================

@app.route('/api/time', methods=['GET'])
def get_time():
    return str(int(time.time()))


@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    sensor_id = data.get('sensor_id')
    roll = data.get('roll')
    pitch = data.get('pitch')
    yaw = data.get('yaw')
    timestamp = data.get('timestamp', time.time())

    if not all([sensor_id, roll is not None, pitch is not None, yaw is not None]):
        return jsonify({'error': 'Missing fields'}), 400

    data_processor.process_incoming(sensor_id, roll, pitch, yaw, timestamp)
    return jsonify({'status': 'ok'}), 200


@app.route('/api/sensors', methods=['GET'])
def get_sensors_status():
    return jsonify(data_processor.get_sensor_status())

# ===============================
# CSV & REPORTS
# ===============================

def get_firebase_db_files(path):
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        return storage.list_files(path) or []
    except Exception as e:
        print(f"Error accessing Firebase RTDB: {e}")
    return []

def get_firebase_db_files_metadata(path):
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        return storage.get_files_metadata(path) or {}
    except Exception as e:
        print(f"Error accessing Firebase RTDB metadata: {e}")
    return {}


@app.route('/api/csv/list', methods=['GET'])
@login_required(role='doctor')
def list_csv():
    fb_meta = get_firebase_db_files_metadata('/files/csv')
    local_files = {}
    if os.path.exists(Config.CSV_DIR):
        for f in os.listdir(Config.CSV_DIR):
            if f.endswith('.csv'):
                filepath = os.path.join(Config.CSV_DIR, f)
                try:
                    local_files[f] = {
                        'timestamp': os.path.getmtime(filepath),
                        'size': os.path.getsize(filepath)
                    }
                except Exception:
                    pass

    all_filenames = list(set(list(fb_meta.keys()) + list(local_files.keys())))
    all_filenames.sort(reverse=True)

    result = []
    for fname in all_filenames:
        in_local = fname in local_files
        in_firebase = fname in fb_meta
        
        timestamp = None
        size = None
        
        if in_local:
            timestamp = local_files[fname]['timestamp']
            size = local_files[fname]['size']
        elif in_firebase:
            timestamp = fb_meta[fname]['timestamp']
            size = fb_meta[fname]['size']

        result.append({
            'filename': fname,
            'timestamp': timestamp,
            'size': size,
            'in_local': in_local,
            'in_firebase': in_firebase
        })

    return jsonify(result)


@app.route('/api/csv/delete/<filename>', methods=['DELETE'])
@login_required(role='doctor')
def delete_csv(filename):
    if '..' in filename or filename.startswith('/'):
        abort(400)

    current_log = data_processor.get_current_log_filename()
    if current_log and filename == current_log:
        return jsonify({'error': 'Cannot delete active log file'}), 403

    # Delete from Firebase
    deleted_from_firebase = False
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        deleted_from_firebase = storage.delete_file('/files/csv', filename)
    except Exception as e:
        print(f"Error deleting CSV from Firebase: {e}")

    filepath = os.path.join(Config.CSV_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({'status': 'ok'}), 200
        except PermissionError:
            return jsonify({'error': 'File in use'}), 409

    if deleted_from_firebase:
        return jsonify({'status': 'ok'}), 200
    abort(404)


@app.route('/api/csv/latest/download', methods=['GET'])
@app.route('/api/csv/latest', methods=['GET'])
@login_required(role='doctor')
def get_latest_csv():
    custom_filename = request.args.get('filename')
    fb_files = get_firebase_db_files('/files/csv') or []
    local_files = []
    if os.path.exists(Config.CSV_DIR):
        local_files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    all_files = list(set(fb_files + local_files))
    all_files.sort(reverse=True)
    
    if not all_files:
        return jsonify({'error': 'No CSV files'}), 404
        
    latest_file = all_files[0]
    download_name = custom_filename or latest_file
    
    if latest_file in fb_files:
        try:
            from firebase_storage import FirebaseStorage
            storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
            file_bytes = storage.get_file('/files/csv', latest_file)
            if file_bytes:
                return send_file(io.BytesIO(file_bytes), download_name=download_name, as_attachment=True)
        except Exception as e:
            print(f"Error fetching CSV from Firebase: {e}")
            
    return send_file(os.path.join(Config.CSV_DIR, latest_file), download_name=download_name, as_attachment=True)


@app.route('/api/csv/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_csv(filename):
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        file_bytes = storage.get_file('/files/csv', filename)
        if file_bytes:
            return send_file(io.BytesIO(file_bytes), download_name=filename, as_attachment=True)
    except Exception as e:
        print(f"Error fetching CSV from Firebase: {e}")

    return send_from_directory(Config.CSV_DIR, filename, as_attachment=True)


@app.route('/api/csv/preview/<filename>', methods=['GET'])
@login_required(role='doctor')
def preview_csv(filename):
    if '..' in filename or filename.startswith('/'):
        abort(400)
    
    csv_data = None
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        file_bytes = storage.get_file('/files/csv', filename)
        if file_bytes:
            csv_data = file_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error fetching CSV for preview: {e}")

    if not csv_data:
        filepath = os.path.join(Config.CSV_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    if not csv_data:
        return jsonify({'error': 'File not found'}), 404

    try:
        reader = csv.reader(io.StringIO(csv_data))
        rows = []
        for i, row in enumerate(reader):
            if i >= 25:  # Let's show up to 25 rows for a richer preview
                break
            rows.append(row)
        return jsonify({'filename': filename, 'rows': rows})
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 500


@app.route('/api/reports/list', methods=['GET'])
@login_required(role='doctor')
def list_reports():
    fb_meta = get_firebase_db_files_metadata('/files/reports')
    local_files = {}
    if os.path.exists(Config.REPORTS_DIR):
        for f in os.listdir(Config.REPORTS_DIR):
            if f.endswith('.pdf'):
                filepath = os.path.join(Config.REPORTS_DIR, f)
                try:
                    local_files[f] = {
                        'timestamp': os.path.getmtime(filepath),
                        'size': os.path.getsize(filepath)
                    }
                except Exception:
                    pass

    all_filenames = list(set(list(fb_meta.keys()) + list(local_files.keys())))
    all_filenames.sort(reverse=True)

    result = []
    for fname in all_filenames:
        in_local = fname in local_files
        in_firebase = fname in fb_meta
        
        timestamp = None
        size = None
        
        if in_local:
            timestamp = local_files[fname]['timestamp']
            size = local_files[fname]['size']
        elif in_firebase:
            timestamp = fb_meta[fname]['timestamp']
            size = fb_meta[fname]['size']

        result.append({
            'filename': fname,
            'timestamp': timestamp,
            'size': size,
            'in_local': in_local,
            'in_firebase': in_firebase
        })

    return jsonify(result)


@app.route('/api/reports/delete/<filename>', methods=['DELETE'])
@login_required(role='doctor')
def delete_report(filename):
    if '..' in filename or filename.startswith('/'):
        abort(400)

    # Delete from Firebase
    deleted_from_firebase = False
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        deleted_from_firebase = storage.delete_file('/files/reports', filename)
    except Exception as e:
        print(f"Error deleting report from Firebase: {e}")

    filepath = os.path.join(Config.REPORTS_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({'status': 'ok'}), 200
        except PermissionError:
            return jsonify({'error': 'File in use'}), 409
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    if deleted_from_firebase:
        return jsonify({'status': 'ok'}), 200
    abort(404)


@app.route('/api/reports/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_report(filename):
    try:
        from firebase_storage import FirebaseStorage
        storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
        file_bytes = storage.get_file('/files/reports', filename)
        if file_bytes:
            return send_file(io.BytesIO(file_bytes), download_name=filename, as_attachment=True)
    except Exception as e:
        print(f"Error fetching report from Firebase: {e}")

    return send_from_directory(Config.REPORTS_DIR, filename, as_attachment=True)


@app.route('/api/report/generate', methods=['POST'])
@login_required()
def generate_report():
    csv_file = None
    files = get_firebase_db_files('/files/csv')
    
    if files:
        try:
            from firebase_storage import FirebaseStorage
            storage = FirebaseStorage(Config.FIREBASE_STORAGE_URL)
            file_bytes = storage.get_file('/files/csv', files[0])
            if file_bytes:
                csv_file = os.path.join(Config.CSV_DIR, files[0])
                with open(csv_file, 'wb') as f:
                    f.write(file_bytes)
        except Exception as e:
            print(f"Error fetching CSV for report from Firebase: {e}")
    
    if not csv_file or not os.path.exists(csv_file):
        local_files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
        local_files.sort(reverse=True)
        if not local_files:
            return jsonify({'error': 'No data available'}), 404
        csv_file = os.path.join(Config.CSV_DIR, local_files[0])

    pdf_file = report_gen.generate(csv_file)
    return send_file(pdf_file, as_attachment=True)

# ===============================
# HELPER: GET LAN IP
# ===============================

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ===============================
# SERVER START
# ===============================

if __name__ == '__main__':
    register_socket_events(socketio)

    lan_ip = get_lan_ip()

    print("\n=======================================")
    print("   ERGO SENSOR SERVER STARTED WITH AI")
    print("=======================================")
    print(f"Local: http://127.0.0.1:5000")
    print(f"LAN:   http://{lan_ip}:5000")
    print("=======================================\n")

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )