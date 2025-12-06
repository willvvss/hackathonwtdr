import os
import pandas as pd

# Path to your raw sensor readings CSV
csv_path = r"C:\Hackathon\sensor_readings.csv"

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
else:
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Convert Timestamp column to proper datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Export first 50 rows
    df.head(50).to_csv(r"C:\Hackathon\sensor_readings_first50.csv", index=False)
    df.head(50).to_excel(r"C:\Hackathon\sensor_readings_first50.xlsx", index=False)

    # Export ALL rows
    df.to_csv(r"C:\Hackathon\sensor_readings_all.csv", index=False)
    df.to_excel(r"C:\Hackathon\sensor_readings_all.xlsx", index=False)

    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total rows: {len(df)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    # If you want to see ALL rows in console:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("\n=== Full Dataset (all rows) ===")
    print(df)