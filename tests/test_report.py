import os
from config import Config
from report_generator import ReportGenerator

config = Config()
generator = ReportGenerator(config)

csv_file = r"c:\MSD_System\csv_data\session_20260506_020211.csv"
pdf_path = generator.generate(csv_file)
print(f"Generated report at {pdf_path}")
