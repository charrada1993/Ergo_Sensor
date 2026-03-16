import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, config):
        self.config = config
        self.filename = None
        self.file = None
        self.writer = None
        self._open_file()

    def _open_file(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(self.config.CSV_DIR, f'session_{timestamp}.csv')
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Header includes RULA and REBA columns
        header = ['Timestamp', 'Neck', 'Back', 'R_Shoulder', 'L_Shoulder',
                  'R_Elbow', 'L_Elbow', 'R_Wrist', 'L_Wrist', 'Global_Risk',
                  'RULA_Right', 'RULA_Left',
                  'REBA_Right', 'REBA_Left']
        self.writer.writerow(header)
        self.file.flush()

    def log(self, angles, risk, rula=None, reba=None):
        """
        Write a row to the CSV file.
        angles: dict of joint angles (may be missing keys)
        risk: dict with 'global' key
        rula: dict with keys 'right' and 'left' (each can be None or a dict containing 'final')
        reba: dict with keys 'right' and 'left' (each can be None or a dict containing 'final')
        """
        if not self.writer:
            return

        # Extract RULA final scores
        rula_right = ''
        rula_left = ''
        if rula:
            if rula.get('right') and 'final' in rula['right']:
                rula_right = rula['right']['final']
            if rula.get('left') and 'final' in rula['left']:
                rula_left = rula['left']['final']

        # Extract REBA final scores
        reba_right = ''
        reba_left = ''
        if reba:
            if reba.get('right') and 'final' in reba['right']:
                reba_right = reba['right']['final']
            if reba.get('left') and 'final' in reba['left']:
                reba_left = reba['left']['final']

        row = [
            datetime.now().isoformat(),
            angles.get('Neck', ''),
            '',  # Back (not computed)
            angles.get('R_Shoulder', ''),
            angles.get('L_Shoulder', ''),
            angles.get('R_Elbow', ''),
            angles.get('L_Elbow', ''),
            angles.get('R_Wrist', ''),
            angles.get('L_Wrist', ''),
            risk['global'] if risk else '',
            rula_right,
            rula_left,
            reba_right,
            reba_left
        ]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()