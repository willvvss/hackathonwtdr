import pandas as pd
import sqlite3
import logging
import os
import shutil
import datetime

# Configure logging
logging.basicConfig(
    filename=r"C:\Hackathon\error_logs.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("\n=== Error Log System ===")

# Step 1: Load dataset
try:
    df = pd.read_csv(r"C:\Hackathon\cleaned_sensor_readings.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError as e:
    logging.error("Cleaned dataset not found: %s", e)
    print("⚠️ Error: Cleaned dataset missing. Check error_logs.txt for details.")
    df = None

# Step 2: Validate dataset columns
try:
    if df is not None:
        required_cols = ["timestamp", "Temperature_C", "Vibration_g", "Axis1_deg", "Axis2_deg", "Axis3_deg"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")
        print("✅ All required columns present.")
except Exception as e:
    logging.error("Data validation failed: %s", e)
    print("⚠️ Error: Data validation issue. Check error_logs.txt for details.")

# Step 3: Database integration (AFTER Step 1 and Step 2)
try:
    if df is not None:
        # Remove duplicate columns
            # Normalize column names (convert to str and strip whitespace)
            orig_cols = [str(c).strip() for c in df.columns]

            # Detect duplicates case-insensitively (SQLite treats identifiers case-insensitively)
            from collections import Counter
            norm_lower = [c.lower() for c in orig_cols]
            counts = Counter(norm_lower)
            dupes = [c for c, n in counts.items() if n > 1]
            if dupes:
                logging.warning("Found duplicate column names (case-insensitive) before DB write: %s", dupes)
                new_cols = []
                counters = {}
                for orig, nl in zip(orig_cols, norm_lower):
                    if counts[nl] > 1:
                        counters[nl] = counters.get(nl, 0) + 1
                        new_cols.append(f"{nl}_{counters[nl]}")
                    else:
                        new_cols.append(nl)
                # Log mapping for debugging
                mapping = dict(zip(orig_cols, new_cols))
                logging.error("Renamed duplicate columns for insertion (orig->new): %s", mapping)
                df.columns = new_cols
            else:
                # Use lowercased, stripped names for database columns to avoid case collisions
                df.columns = norm_lower

            # Drop common unwanted columns if present (safe — ignores missing)
            df = df.drop(columns=["Unnamed: 0", "record_id"], errors="ignore")

            # Path to the SQLite DB
            db_path = r"C:\Hackathon\robot_sensors.db"

            # If DB file exists but isn't a valid SQLite DB, back it up so we can recreate a fresh one
            try:
                if os.path.exists(db_path):
                    try:
                        with open(db_path, "rb") as f:
                            header = f.read(16)
                        if not header.startswith(b"SQLite format 3"):
                            bak = db_path + ".corrupt." + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            shutil.move(db_path, bak)
                            logging.error("Existing DB file invalid; moved to %s", bak)
                    except Exception as e:
                        logging.warning("Could not validate DB header: %s", e)

                # Connect to SQLite (with a timeout to help with locks)
                conn = sqlite3.connect(db_path, timeout=10)

                # Log DataFrame summary for debugging
                logging.info("Writing DataFrame to DB: rows=%s cols=%s", df.shape[0], list(df.columns))

                # Try to write table; on certain DatabaseErrors we attempt a single recovery
                try:
                    df.to_sql("sensor_readings", conn, if_exists="replace", index=False)
                except sqlite3.DatabaseError as db_err:
                    logging.error("First DB write attempt failed: %s", db_err)
                    # If the DB file appears corrupted, back it up and retry once
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if os.path.exists(db_path):
                        try:
                            bak = db_path + ".failed." + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            shutil.move(db_path, bak)
                            logging.error("Moved problematic DB to %s and retrying", bak)
                        except Exception as e:
                            logging.error("Could not move DB file during recovery: %s", e)
                    # Reconnect and retry
                    conn = sqlite3.connect(db_path, timeout=10)
                    df.to_sql("sensor_readings", conn, if_exists="replace", index=False)

                conn.close()
                print("✅ Data inserted into database successfully.")
            except Exception as e:
                logging.error("Database insertion failed: %s", e)
                print("⚠️ Error: Database issue. Check error_logs.txt for details.")
except Exception as e:
    logging.error("Database insertion failed: %s", e)
    print("⚠️ Error: Database issue. Check error_logs.txt for details.")

print("\n=== End of Error Log Run ===")