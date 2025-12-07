# hackathonwtdr
Trever Fuhrer | Will Vogt | Priyansh Dhiman | Arshdeep Singh

## Hackathon Robot Sensor Data Pipeline

## Overview
This project cleans and structures raw robot sensor data (temperature, vibration, axis angles)...

## How It Works
- Reads raw CSV sensor data
- Converts timestamps into standard format
- Identifies missing values
- Exports judge‑ready CSV and Excel files

## Results
Transparent dataset with 150 rows, ready for analysis and presentation.

# Executable code to run Sensor.py
C:\Hackathon\.venv\Scripts\python.exe Clean_Sensor_Reading.py

## Performance Metrics

After cleaning and validating the sensor dataset, we calculate performance metrics to evaluate both data quality and robot behavior. These metrics provide insights into how reliable the readings are and how the robot is performing under different conditions.

### 1. Data Quality
- **Row count after cleaning** → shows how many valid readings remain.
- **Missing values per column** → confirms that gaps were filled or handled.
- **Outlier removal** → ensures unrealistic values (e.g., extreme temperatures or vibration spikes) are filtered out.

### 2. Sensor Health
- **Temperature (°C)**  
  - Average, minimum, maximum, and standard deviation.  
  - Demonstrates thermal stability and whether the robot stays within safe operating ranges.

- **Vibration (g-force)**  
  - Average vibration level and peak spikes.  
  - Indicates mechanical stability and potential issues with shaking or instability.

- **Axis Orientation (degrees)**  
  - Average, minimum, and maximum values for Axis1, Axis2, and Axis3.  
  - Shows how the robot’s orientation changes over time and whether it remains stable.

### 3. Robot Performance Indicators
- **Thermal stability** → Is the robot maintaining safe temperature ranges?  
- **Mechanical stability** → Are vibration spikes rare or frequent?  
- **Orientation control** → Are axis angles consistent or erratic?

## System Alerts

The system alerts module automatically scans the cleaned sensor dataset and raises warnings when performance metrics exceed safe thresholds. This ensures the robot can self‑diagnose issues and notify operators in real time.

### Alert Rules
- **Temperature**
  - ⚠️ Triggered if `Temperature_C > 50 °C` (overheating).
  - ⚠️ Triggered if `Temperature_C < 10 °C` (abnormally cold or sensor error).
- **Vibration**
  - ⚠️ Triggered if `Vibration_g > 0.3 g` (mechanical instability).
- **Orientation**
  - ⚠️ Triggered if any axis (`Axis1_deg`, `Axis2_deg`, `Axis3_deg`) exceeds 60° (unsafe tilt).
- **Data Quality**
  - ⚠️ Triggered if more than 5% of values are missing in the dataset.

### Example Output
If thresholds are breached: === System Alerts === ⚠️ Axis1_deg exceeded safe orientation range (>60°) ⚠️ More than 5% missing values in dataset
If all readings are safe: === System Alerts === ✅ All systems stable

### Purpose
System alerts provide immediate feedback on robot health and data integrity. They make the pipeline production‑ready by ensuring unsafe conditions are flagged before further analysis or deployment.


### Output
The script prints a summary report in the console and can export results to CSV/Excel. This makes it easy to compare raw vs. cleaned data and demonstrate improvements in reliability.

## Maintenance Notes

The maintenance notes module converts system alerts into actionable recommendations for operators. Instead of only flagging unsafe conditions, it provides human‑readable guidance that can be logged for long‑term reliability.

### Example Rules
- **Overheating** → "Inspect cooling system; possible overheating."
- **Abnormal cold readings** → "Check sensor calibration; abnormal low reading."
- **Excessive vibration** → "Inspect mechanical joints; possible instability."
- **Unsafe orientation** → "Check axis control; unsafe tilt detected."
- **Data integrity issues** → "Review sensor wiring/logging; data integrity issue."

### Example Output
If issues are detected:

Documentation:

Document of Data Cleaning Decisions: 
[Hackathon 2025, Document your data cleaning decisions .pdf](https://github.com/user-attachments/files/24017416/Hackathon.2025.Document.your.data.cleaning.decisions.pdf)

Document of Written Documentation:
[Hackathon Written Documentation 2025.pdf](https://github.com/user-attachments/files/24017420/Hackathon.Written.Documentation.2025.pdf)

