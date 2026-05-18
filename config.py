import os

class Config:
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    # Firebase
    FIREBASE_DATABASE_URL = 'https://msd-monitor-system-default-rtdb.europe-west1.firebasedatabase.app/'
    FIREBASE_CREDENTIALS_PATH = 'msd-monitor-system-firebase-adminsdk-fbsvc-57e212bc0a.json'
    # Data ingestion – exact sensor IDs as sent by ESP32
    EXPECTED_SENSORS = [
        'NECK', 'UPPER_BACK',
        'R_BICEPS', 'L_BICEPS',
        'R_FOREARM', 'L_FOREARM',
        'R_HAND', 'L_HAND',
        'R_THIGH', 'L_THIGH',      # note capital T, lowercase rest
        'R_SHANK', 'L_SHANK'        # if used
    ]
    POST_INTERVAL_MS = 100

    # Paths
    CSV_DIR = 'csv_data'
    REPORTS_DIR = 'reports'
    LOGS_DIR = 'logs'

    # Risk model parameters
    CWD = 3600
    ET = 28800
    CF = 15
    PF = 5
    MF = 25
    MRQ = 60
    MDR = 0
    E = 90
    CWL = 5
    PWL = 2
    MVC = 10
    RR = 600
    AR = 120
    RISK_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

    # RULA defaults
    RULA_LEGS_SCORE = 1

    # REBA defaults
    REBA_LEGS_SCORE = 1

    # Ensure directories exist
    for d in [CSV_DIR, REPORTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)