from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, csv_file):
        # This is a simplified example. In production, parse CSV and compute statistics.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_file = os.path.join(self.config.REPORTS_DIR, f'report_{timestamp}.pdf')
        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "MSD Risk Assessment Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Placeholder statistics
        c.drawString(50, height - 120, "Maximum Angles:")
        c.drawString(70, height - 140, "Neck: 25.3 deg")
        c.drawString(70, height - 160, "R_Shoulder: 45.1 deg")
        # ... add more

        c.drawString(50, height - 200, "Average Angles:")
        c.drawString(70, height - 220, "Neck: 12.8 deg")

        c.drawString(50, height - 260, "Time in Risky Posture: 15 minutes")

        c.drawString(50, height - 300, "Risk Breakdown:")
        c.drawString(70, height - 320, "R1 (Static): 0.2")
        c.drawString(70, height - 340, "R2 (Repetition): 0.5")
        c.drawString(70, height - 360, "R3 (Angle): 0.7")
        c.drawString(70, height - 380, "R4 (Effort): 0.3")
        c.drawString(70, height - 400, "R5 (Recovery): 0.1")
        c.drawString(70, height - 420, "Global Risk: 0.36 (Moderate)")

        # Generate a simple plot (placeholder)
        plt.figure(figsize=(4, 3))
        plt.plot([1,2,3,4], [10,20,15,25])
        plt.title('Neck Angle Over Time')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 50, height - 550, width=300, height=150)

        c.save()
        return pdf_file