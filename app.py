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


@app.route('/rula')
@login_required(role='doctor')
def rula_page():
    return render_template('rula.html')


@app.route('/reba')
@login_required(role='doctor')
def reba_page():
    return render_template('reba.html')


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
@login_required(role='doctor')
def calibrate():
    data_processor.calibrate()
    return jsonify({'status': 'ok'}), 200

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
        from firebase_admin import db
        ref = db.reference(path)
        data = ref.get()
        if data:
            files = [v['filename'] for v in data.values() if 'filename' in v]
            files.sort(reverse=True)
            return files
    except Exception as e:
        print(f"Error accessing Firebase RTDB: {e}")
    return None

@app.route('/api/csv/list', methods=['GET'])
@login_required(role='doctor')
def list_csv():
    files = get_firebase_db_files('/files/csv')
    if files is not None:
        return jsonify(files)
    
    # Fallback to local
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

    # Delete from Firebase
    deleted_from_firebase = False
    try:
        from firebase_admin import db
        file_key = filename.replace('.', '_')
        ref = db.reference(f'/files/csv/{file_key}')
        if ref.get():
            ref.delete()
            deleted_from_firebase = True
    except Exception:
        pass

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


@app.route('/api/csv/latest', methods=['GET'])
@login_required(role='doctor')
def get_latest_csv():
    files = get_firebase_db_files('/files/csv')
    if files:
        return download_csv(files[0])

    files = [f for f in os.listdir(Config.CSV_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)
    if not files:
        return jsonify({'error': 'No CSV files'}), 404

    return send_from_directory(Config.CSV_DIR, files[0], as_attachment=True)


@app.route('/api/csv/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_csv(filename):
    try:
        from firebase_admin import db
        file_key = filename.replace('.', '_')
        ref = db.reference(f'/files/csv/{file_key}')
        data = ref.get()
        if data and 'data' in data:
            file_bytes = base64.b64decode(data['data'])
            return send_file(io.BytesIO(file_bytes), download_name=filename, as_attachment=True)
    except Exception as e:
        print(f"Error fetching from RTDB: {e}")

    return send_from_directory(Config.CSV_DIR, filename, as_attachment=True)


@app.route('/api/reports/list', methods=['GET'])
@login_required(role='doctor')
def list_reports():
    files = get_firebase_db_files('/files/reports')
    if files is not None:
        return jsonify(files)

    files = [f for f in os.listdir(Config.REPORTS_DIR) if f.endswith('.pdf')]
    files.sort(reverse=True)
    return jsonify(files)


@app.route('/api/reports/download/<filename>', methods=['GET'])
@login_required(role='doctor')
def download_report(filename):
    try:
        from firebase_admin import db
        file_key = filename.replace('.', '_')
        ref = db.reference(f'/files/reports/{file_key}')
        data = ref.get()
        if data and 'data' in data:
            file_bytes = base64.b64decode(data['data'])
            return send_file(io.BytesIO(file_bytes), download_name=filename, as_attachment=True)
    except Exception as e:
        print(f"Error fetching from RTDB: {e}")

    return send_from_directory(Config.REPORTS_DIR, filename, as_attachment=True)


@app.route('/api/report/generate', methods=['POST'])
@login_required(role='doctor')
def generate_report():
    csv_file = None
    files = get_firebase_db_files('/files/csv')
    
    if files:
        try:
            from firebase_admin import db
            file_key = files[0].replace('.', '_')
            ref = db.reference(f'/files/csv/{file_key}')
            data = ref.get()
            if data and 'data' in data:
                file_bytes = base64.b64decode(data['data'])
                csv_file = os.path.join(Config.CSV_DIR, files[0])
                with open(csv_file, 'wb') as f:
                    f.write(file_bytes)
        except Exception as e:
            print(f"Error fetching CSV for report from RTDB: {e}")
    
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
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )