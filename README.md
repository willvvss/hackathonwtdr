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

### Output
The script prints a summary report in the console and can export results to CSV/Excel. This makes it easy to compare raw vs. cleaned data and demonstrate improvements in reliability.
