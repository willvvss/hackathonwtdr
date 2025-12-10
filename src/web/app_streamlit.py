import sys
from pathlib import Path
import json
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# messy path fix to make imports work on everyone's laptop
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_DIR,
    EVENTS_FILE,
    AI_RECOMMENDATIONS_FILE,
)
from src.run_pipeline import main as run_pipeline_main

load_dotenv()

# Hopefully this doesn't break on the demo machine
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("WARNING: openai not installed, AI features won't work")

# ------------------------------
# Mappings & Helpers
# ------------------------------

# Greg: mapped these based on the sample dataset filenames. 
# If they change the filenames, this breaks.
EXPECTED_FILES = {
    "error_logs": "error_logs.txt",
    "system_alerts": "system_alerts.txt",
    "maintenance_notes": "maintenance_notes.txt",
    "sensor_readings": "sensor_readings.csv",
    "torque_timeseries": "Torque Timeseries.csv",
    "torque_cycles": "Torque Events by Cycle.csv",
    "performance_metrics": "performance_metrics.csv",
}

def save_uploaded_files(uploaded_files):
    # Dumps files into data_raw/ so the pipeline can find them
    if not uploaded_files:
        return []

    saved = []
    for uf in uploaded_files:
        name_lower = uf.name.lower()
        mapped_name = None
        
        # super basic string matching bc regex was acting up
        if "error" in name_lower and "log" in name_lower:
            mapped_name = EXPECTED_FILES["error_logs"]
        elif "alert" in name_lower:
            mapped_name = EXPECTED_FILES["system_alerts"]
        elif "maint" in name_lower or "note" in name_lower:
            mapped_name = EXPECTED_FILES["maintenance_notes"]
        elif "sensor" in name_lower:
            mapped_name = EXPECTED_FILES["sensor_readings"]
        elif "timeseries" in name_lower or "time series" in name_lower:
            mapped_name = EXPECTED_FILES["torque_timeseries"]
        elif "event" in name_lower and "cycle" in name_lower:
            mapped_name = EXPECTED_FILES["torque_cycles"]
        elif "perf" in name_lower:
            mapped_name = EXPECTED_FILES["performance_metrics"]

        # Default to original name if we can't guess it
        final_name = mapped_name if mapped_name else uf.name
        out_path = RAW_DIR / final_name
        
        with out_path.open("wb") as f:
            f.write(uf.getbuffer())
        saved.append(out_path)

    return saved

def load_data():
    # just a wrapper to load the csvs safely
    try:
        events = pd.read_csv(EVENTS_FILE)
    except FileNotFoundError:
        events = pd.DataFrame()

    try:
        recs = pd.read_csv(AI_RECOMMENDATIONS_FILE)
        # make sure event_id matches the other dataframe
        if not recs.empty and "event_id" in recs.columns:
            recs["event_id"] = pd.to_numeric(recs["event_id"], errors="coerce").astype("Int64")
    except FileNotFoundError:
        recs = pd.DataFrame()

    return events, recs

def build_prompt(row):
    # This prompt seems to work best with GPT-4o, don't change the JSON structure part
    fields = {
        "event_id": int(row.get("event_id", -1)),
        "timestamp": str(row.get("timestamp", "")),
        "axis": int(row.get("axis", 0)),
        "location": str(row.get("location", "")),
        "collision_type": str(row.get("collision_type", "")),
        "severity": str(row.get("severity", "")),
        "peak_torque": float(row.get("peak_torque_pct", 0.0)),
        "alert_msg": str(row.get("alert_message", "")),
        "last_maint": str(row.get("last_maintenance_task", "")),
        "raw_msg": str(row.get("message_raw", "")),
    }

    return f"""
You are a senior robotics reliability engineer. 
Analyze this collision event and provide a maintenance plan in JSON ONLY.

Context:
- ID: {fields['event_id']}
- Time: {fields['timestamp']}
- Joint: {fields['axis']} ({fields['location']})
- Type: {fields['collision_type']} (Severity: {fields['severity']})
- Peak Torque: {fields['peak_torque']}% of rated
- Alert: {fields['alert_msg']}
- Last Maintenance: {fields['last_maint']}
- Raw Log: {fields['raw_msg']}

Response Format (JSON):
{{
    "event_id": {fields['event_id']},
    "diagnosis": "brief technical explanation",
    "inspection_steps": "bullet points with line breaks",
    "maintenance_actions": "bullet points with line breaks",
    "safety_clearance": "bullet points",
    "return_to_service": "steps to restart"
}}
Keep it concise.
"""

def run_ai_analysis(events, endpoint, api_key, deployment):
    if OpenAI is None:
        st.error("OpenAI lib missing! pip install openai")
        return pd.DataFrame()

    client = OpenAI(base_url=endpoint, api_key=api_key)

    # Filter: Only send HIGH or CRITICAL severity to save API credits during testing
    # Check for both "High" and "Critical" (case insensitive)
    if "severity" in events.columns:
        subset = events[events["severity"].str.lower().isin(["high", "critical"])].copy()
        if subset.empty:
            # Fallback if no high severity events exist
            subset = events.head(5) 
    else:
        subset = events.head(5)

    rec_rows = []
    progress_bar = st.progress(0)
    total = len(subset)

    for i, (_, row) in enumerate(subset.iterrows()):
        ev_id = int(row["event_id"])
        prompt = build_prompt(row)

        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a robotics engineer. JSON output only."},
                    {"role": "user", "content": prompt},
                ]
            )
            content = resp.choices[0].message.content.strip()
            
            # sometimes the model wraps json in ```json ... ```
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            data = json.loads(content)
        except Exception as e:
            # If it fails, just fill empty so the UI doesn't crash
            data = {
                "event_id": ev_id,
                "diagnosis": f"AI Error: {str(e)}",
                "inspection_steps": "N/A", 
                "maintenance_actions": "N/A",
                "safety_clearance": "N/A",
                "return_to_service": "N/A"
            }

        # Merge original fields for display
        output = {
            "event_id": int(data.get("event_id", ev_id)),
            "axis": row.get("axis"),
            "severity": row.get("severity"),
            "collision_type": row.get("collision_type"),
            "diagnosis": data.get("diagnosis", ""),
            "inspection_steps": data.get("inspection_steps", ""),
            "maintenance_actions": data.get("maintenance_actions", ""),
            "safety_clearance": data.get("safety_clearance", ""),
            "return_to_service": data.get("return_to_service", ""),
        }
        rec_rows.append(output)
        progress_bar.progress((i + 1) / total)

    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(AI_RECOMMENDATIONS_FILE, index=False)
    return rec_df

# ------------------------------
# Streamlit UI
# ------------------------------

def main():
    st.set_page_config(page_title="CSI Bot Diagnostic", layout="wide")
    st.title("CSI Hackathon: Robot Diagnostic Tool")

    # Sidebar: File IO
    st.sidebar.header("1. Data Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Logs/CSVs",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_paths = save_uploaded_files(uploaded_files)
        st.sidebar.success(f"Loaded {len(saved_paths)} files.")

    if st.sidebar.button("Run ETL Pipeline"):
        with st.spinner("Parsing logs..."):
            run_pipeline_main()
        st.sidebar.success("Done! Events rebuilt.")

    # Main Area
    events, recs = load_data()

    if events.empty:
        st.warning("⚠️ No event data found. Please upload files and run the pipeline.")
        return

    # Top level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(events))
    
    # Count both High and Critical so this isn't 0
    if "severity" in events.columns:
        critical_count = len(events[events['severity'].str.lower().isin(['critical', 'high'])])
    else:
        critical_count = 0
    col2.metric("Critical / High Errors", critical_count)
    
    col3.metric("AI Recommendations", len(recs) if not recs.empty else 0)

    st.subheader("Event Log")
    
    # FIX: Updated per error logs (use_container_width -> width="stretch")
    st.dataframe(events, width="stretch")

    # Event Viewer
    st.divider()
    st.subheader("Deep Dive & AI Analysis")
    
    # Select event ID
    if not events.empty:
        all_ids = sorted(events["event_id"].unique())
        selected_id = st.selectbox("Select Event ID", all_ids)
        
        col_left, col_right = st.columns([1, 1])

        with col_left:
            ev_subset = events[events["event_id"] == selected_id]
            if not ev_subset.empty:
                ev = ev_subset.iloc[0]
                st.markdown(f"#### 📊 Raw Data (ID: {selected_id})")
                st.write(f"**Timestamp:** {ev.get('timestamp')}")
                st.write(f"**Axis:** {ev.get('axis')} ({ev.get('location')})")
                st.write(f"**Severity:** {ev.get('severity')}")
                st.write(f"**Alert:** {ev.get('alert_message')}")
                st.code(ev.get('message_raw'), language="text")

        with col_right:
            st.markdown(f"#### 🧠 AI Maintenance Plan")
            
            # Check if we already have a rec for this ID
            current_rec = pd.DataFrame()
            if not recs.empty:
                current_rec = recs[recs["event_id"] == selected_id]

            if not current_rec.empty:
                r = current_rec.iloc[0]
                with st.expander("Diagnosis", expanded=True):
                    st.info(r.get("diagnosis"))
                with st.expander("Action Plan"):
                    st.write("**Inspection:**")
                    st.text(r.get("inspection_steps"))
                    st.write("**Fix:**")
                    st.text(r.get("maintenance_actions"))
            else:
                st.info("No AI analysis generated for this event yet.")
    
    # Azure Config Section (Bottom)
    st.divider()
    with st.expander("⚙️ Admin / API Settings"):
        c1, c2, c3 = st.columns(3)
        # Defaults from .env so we don't have to type it every time
        endpoint = c1.text_input("Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        api_key = c2.text_input("Key", type="password", value=os.getenv("AZURE_OPENAI_API_KEY", ""))
        deployment = c3.text_input("Deployment", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))

        if st.button("Generate New Recommendations (High Severity Only)"):
            if not api_key:
                st.error("Need API Key!")
            else:
                with st.spinner("Querying Azure OpenAI..."):
                    run_ai_analysis(events, endpoint, api_key, deployment)
                st.success("Analysis complete! Reloading...")
                st.rerun()

if __name__ == "__main__":
    main()