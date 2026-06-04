import os
import pandas as pd
from config import Config
from report_generator import ReportGenerator

config = Config()
generator = ReportGenerator(config)

csv_file = r"c:\MSD_System\csv_data\session_20260506_020211.csv"
df = pd.read_csv(csv_file).head(200)
temp_csv = r"c:\MSD_System\csv_data\temp_test.csv"
df.to_csv(temp_csv, index=False)

pdf_path = generator.generate(temp_csv)
print(f"Generated report at {pdf_path}")
