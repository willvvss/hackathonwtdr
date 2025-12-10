# CSI Hackathon – Robot Collision Diagnostic Tool

**Team:** Trever Fuhrer | Will Vogt | Priyansh Dhiman | Arshdeep Singh

## Overview

This project is an intelligent **Robot Diagnostic Dashboard** designed to reduce downtime in manufacturing environments. It ingests raw, unstructured robot logs and telemetry data, structures them into a unified "Event History," and uses **Azure OpenAI (5.1)** to automatically generate actionable maintenance plans for critical collision events.

## Tech Stack

**Frontend:** Streamlit (Python)  
**Data Processing:** Pandas, Regex  
**AI Model:** Azure OpenAI (GPT-5.1)  
**Environment:** Python 3.9+

## See it in Action

![Program Demo](show.gif)

## How It Works

1. **Data Ingestion:** Operators upload raw files (Error Logs, System Alerts, Torque Timeseries, Sensor Readings).  
2. **ETL Pipeline:** A Python script parses disparate text files and CSVs, merging them into a single, structured `events.csv`.  
3. **Interactive Dashboard:** A Streamlit web app visualizes the data, highlighting **Critical** and **High** severity events.  
4. **Generative AI Analysis:** The system sends high-severity event context to Azure OpenAI, which returns a structured JSON maintenance plan (Diagnosis, Inspection Steps, Maintenance Actions).

## Workflow

To use the tool effectively:

1. **Upload Data:** Drag and drop raw log files into the sidebar upload widget.  
2. **Run Pipeline:** Click the **"Run Full Pipeline (ETL + AI)"** button. This parses the files and immediately queries the AI model for solutions.  
3. **Triage:** Use the **Critical Alert Buttons** at the top of the dashboard to jump instantly to the most severe crashes.  
4. **Resolve:** Read the AI-generated "Inspection Steps" and "Return to Service" plan to fix the robot.

## Key Features

### 1. Automated ETL & Parsing
Instead of manually reading through thousands of lines of `error_logs.txt`, our pipeline automatically extracts:

- **Timestamps & Locations:** When and where the crash happened (Axis/Joint).  
- **Severity Classification:** Tags events as Critical, High, or Warning.  
- **Contextual Data:** Correlates collision events with Peak Torque % and Sensor Alerts.

### 2. Intelligent System Alerts
The dashboard provides an immediate "triage" view for operators:

- **Red "Critical" Buttons:** One-click access to the most severe crashes, organized in a control panel view.  
- **Metric Cards:** Real-time counters for Total Events, Critical Errors, and AI-Generated Plans.

### 3. AI-Driven Maintenance Plans
We utilize Large Language Models (LLMs) to turn raw error codes into human-readable instructions.

**Input:** Raw error message + Peak Torque + Last Maintenance Date  
**AI Output:**

- **Diagnosis:** Technical explanation of the fault (e.g., "Axis 3 collision detected with 145% rated torque").  
- **Inspection Steps:** Bulleted list of hardware to check (e.g., "Inspect harmonic drive for backlash").  
- **Return to Service:** Safety clearance steps to restart production.

## Performance Metrics

- **Data Structured:** Successfully parsed unstructured text logs into a 100% structured dataframe.  
- **Response Time:** AI Maintenance Plans are generated in <5 seconds per event.  
- **Reliability:** Visualizes "Days Since Last Maintenance" alongside crash data to identify neglect-related failures.

## Usage

**To run the application:**

```bash
# 1. Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Run Streamlit App
streamlit run streamlit_app.py
```
