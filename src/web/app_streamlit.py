import sys
from pathlib import Path
import json
import os

# whats this
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.config import (
    RAW_DIR,
    EVENTS_FILE,
    AI_RECOMMENDATIONS_FILE,
)
from src.run_pipeline import main as run_pipeline_main

# Load .env at startup
load_dotenv()

# Try to import OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ------------------------------
# Helpers
# ------------------------------

EXPECTED_FILES = {
    "error_logs": "error_logs.txt",
    "system_alerts": "system_alerts.txt",
    "maintenance_notes": "maintenance_notes.txt",
    "sensor_readings": "sensor_readings.csv",
    "torque_timeseries": "Torque Timeseries.csv",
    "torque_cycles": "Torque Events by Cycle.csv",
    "performance_metrics": "performance_metrics.csv",
}


def save_uploaded_files(uploaded_files: list):
    """
    Save uploaded raw files into data_raw/ using the names expected
    by the pipeline. We match by simple substrings in the filename.
    """
    if not uploaded_files:
        return []

    saved = []
    for uf in uploaded_files:
        name_lower = uf.name.lower()

        mapped_name = None
        # crude but effective matching by substring
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
        elif "perf" in name_lower or "performance" in name_lower:
            mapped_name = EXPECTED_FILES["performance_metrics"]

        # If we couldn't guess, just keep original name
        if mapped_name is None:
            mapped_name = uf.name

        out_path = RAW_DIR / mapped_name
        with out_path.open("wb") as f:
            f.write(uf.getbuffer())
        saved.append(out_path)

    return saved


def load_events_and_recs():
    # Load structured events
    try:
        events = pd.read_csv(EVENTS_FILE)
    except FileNotFoundError:
        events = pd.DataFrame()

    # Load AI recommendations
    try:
        recs = pd.read_csv(AI_RECOMMENDATIONS_FILE)
        if not recs.empty and "event_id" in recs.columns:
            # Clean and force event_id to plain int for reliable matching
            recs["event_id"] = pd.to_numeric(
                recs["event_id"], errors="coerce"
            ).astype("Int64")
    except FileNotFoundError:
        recs = pd.DataFrame()

    return events, recs


def build_prompt_for_event(row: pd.Series) -> str:
    """
    Build a text prompt describing one event for the LLM.
    """
    fields = {
        "event_id": int(row.get("event_id", -1)),
        "timestamp": str(row.get("timestamp", "")),
        "axis": int(row.get("axis", 0)) if pd.notna(row.get("axis", None)) else 0,
        "location": str(row.get("location", "")),
        "collision_type": str(row.get("collision_type", "")),
        "severity": str(row.get("severity", "")),
        "peak_torque_pct": float(row.get("peak_torque_pct", 0.0))
        if pd.notna(row.get("peak_torque_pct", None))
        else 0.0,
        "force_value": float(row.get("force_value", 0.0)),
        "repeats_24h": int(row.get("repeats_24h", 0))
        if pd.notna(row.get("repeats_24h", None))
        else 0,
        "alert_level": str(row.get("alert_level", "")),
        "alert_type": str(row.get("alert_type", "")),
        "alert_message": str(row.get("alert_message", "")),
        "last_maintenance_task": str(row.get("last_maintenance_task", "")),
        "days_since_last_maintenance": int(row.get("days_since_last_maintenance", 0))
        if pd.notna(row.get("days_since_last_maintenance", None))
        else None,
        "message_raw": str(row.get("message_raw", "")),
    }

    prompt = f"""
You are an expert robotics reliability engineer. You will receive a single robot collision or near-miss event
with context. Produce a structured maintenance recommendation in JSON ONLY, no extra text.

Event details:
- event_id: {fields['event_id']}
- timestamp: {fields['timestamp']}
- axis / joint: {fields['axis']} (location: {fields['location']})
- collision_type: {fields['collision_type']}
- severity: {fields['severity']}
- peak_torque_pct_of_rated: {fields['peak_torque_pct']}
- force_value_proxy: {fields['force_value']}
- repeats_24h_same_code_and_axis: {fields['repeats_24h']}
- alert_level: {fields['alert_level']}
- alert_type: {fields['alert_type']}
- alert_message: {fields['alert_message']}
- last_maintenance_task: {fields['last_maintenance_task']}
- days_since_last_maintenance: {fields['days_since_last_maintenance']}
- raw_error_message: {fields['message_raw']}

Output a JSON object with exactly these keys:
- "event_id" (int)
- "diagnosis" (string)
- "inspection_steps" (string, bullet-style with line breaks)
- "maintenance_actions" (string, bullet-style with line breaks)
- "safety_clearance" (string, bullet-style with line breaks)
- "return_to_service" (string, bullet-style with line breaks)

Keep it concise but specific to this joint and event history.
"""
    return prompt.strip()


def generate_ai_recommendations(
    events: pd.DataFrame,
    endpoint: str,
    api_key: str,
    deployment: str,
) -> pd.DataFrame:
    """
    Call Azure OpenAI (using OpenAI client with base_url) for each relevant event
    and save ai_recommendations.csv.
    """

    if OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not installed. Run `pip install openai` inside your venv."
        )

    client = OpenAI(
        base_url=endpoint,  # e.g. "https://<resource>.openai.azure.com/openai/v1"
        api_key=api_key,
    )

    # focus on the most important events
    if "severity" in events.columns:
        subset = events[events["severity"].isin(["high", "critical"])].copy()
        if subset.empty:
            subset = events.copy()
    else:
        subset = events.copy()

    rec_rows = []

    for _, row in subset.iterrows():
        ev_id = int(row["event_id"])
        prompt = build_prompt_for_event(row)

        try:
            resp = client.chat.completions.create(
                model=deployment,  # deployment name, e.g. "gpt-5.1-chat"
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior robotics maintenance engineer. Respond in JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                # no temperature override for this deployment
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            rec_rows.append(
                {
                    "event_id": ev_id,
                    "axis": row.get("axis"),
                    "severity": row.get("severity"),
                    "collision_type": row.get("collision_type"),
                    "diagnosis": f"Error calling Azure OpenAI: {e}",
                    "inspection_steps": "",
                    "maintenance_actions": "",
                    "safety_clearance": "",
                    "return_to_service": "",
                }
            )
            continue

        try:
            data = json.loads(content)
        except Exception:
            data = {
                "event_id": ev_id,
                "diagnosis": content,
                "inspection_steps": "",
                "maintenance_actions": "",
                "safety_clearance": "",
                "return_to_service": "",
            }

        rec_rows.append(
            {
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
        )

    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(AI_RECOMMENDATIONS_FILE, index=False)
    return rec_df


# ------------------------------
# Streamlit App
# ------------------------------

def main():
    st.title("CSI Hackathon – Robot Collision Diagnostic Tool")

    # --- Sidebar: File upload + pipeline ---
    st.sidebar.header("1. Upload raw data files")

    uploaded_files = st.sidebar.file_uploader(
        "Upload robot logs / alerts / maintenance / torque CSVs",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_paths = save_uploaded_files(uploaded_files)
        st.sidebar.success(
            f"Saved {len(saved_paths)} files into data_raw/. "
            "You can now run the data pipeline."
        )

    st.sidebar.header("2. Run data pipeline")
    if st.sidebar.button("Run data pipeline"):
        with st.spinner("Running parsing + event builder pipeline..."):
            run_pipeline_main()
        st.sidebar.success("Pipeline complete. Events have been rebuilt.")

    # --- Main data display ---
    events, recs = load_events_and_recs()

    st.subheader("Structured Events")
    if events.empty:
        st.warning("No events found. Upload files and run the data pipeline.")
        return

    st.dataframe(events)

    # Show which events have AI plans (nice for judges too)
    if not recs.empty and "event_id" in recs.columns:
        ai_ids = sorted(set(int(x) for x in recs["event_id"].dropna().tolist()))
        st.markdown(f"**Events with AI maintenance plans:** {ai_ids}")
    else:
        st.markdown("**Events with AI maintenance plans:** none yet")

    # Event selection
    selected_id = st.number_input(
        "View event by ID",
        min_value=1,
        max_value=int(events["event_id"].max()),
        step=1,
    )
    selected_id_int = int(selected_id)

    # show raw event info
    ev = events[events["event_id"] == selected_id_int]
    if not ev.empty:
        eraw = ev.iloc[0]
        st.markdown(f"### Event {int(eraw['event_id'])} – raw summary")
        st.write(
            f"Timestamp: {eraw.get('timestamp', 'N/A')}  \n"
            f"Axis / location: {eraw.get('axis', 'N/A')} / {eraw.get('location', 'N/A')}  \n"
            f"Collision type: {eraw.get('collision_type', 'N/A')}  \n"
            f"Severity: {eraw.get('severity', 'N/A')}  \n"
            f"Peak torque %: {eraw.get('peak_torque_pct', 'N/A')}  \n"
            f"Alert: {eraw.get('alert_level', 'N/A')} / {eraw.get('alert_type', 'N/A')} – {eraw.get('alert_message', '')}  \n"
            f"Last maintenance: {eraw.get('last_maintenance_task', 'N/A')} "
            f"({eraw.get('days_since_last_maintenance', 'N/A')} days ago)"
        )

    # --- Azure OpenAI config ---
    st.subheader("AI Maintenance Recommendations (Azure GPT)")

    default_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    default_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    default_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    with st.expander("Azure OpenAI configuration", expanded=False):
        endpoint = st.text_input(
            "Azure OpenAI endpoint",
            value=default_endpoint,
            placeholder="https://<resource>.openai.azure.com/openai/v1",
        )
        api_key = st.text_input(
            "Azure OpenAI API key",
            type="password",
            value=default_api_key,
            placeholder="Enter your Azure OpenAI key",
        )
        deployment = st.text_input(
            "Deployment name (e.g., gpt-5.1-chat)",
            value=default_deployment,
            placeholder="gpt-5.1-chat",
        )

    if st.button("Generate AI recommendations for events"):
        if not endpoint or not api_key or not deployment:
            st.error("Please fill in endpoint, API key, and deployment name.")
        else:
            try:
                with st.spinner("Calling Azure OpenAI to generate recommendations..."):
                    recs = generate_ai_recommendations(
                        events,
                        endpoint=endpoint,
                        api_key=api_key,
                        deployment=deployment,
                    )
                st.success(
                    f"Generated {len(recs)} recommendations and saved to ai_recommendations.csv."
                )
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    # --- Show AI recs for selected event ---
    if not recs.empty and "event_id" in recs.columns:
        er = recs[recs["event_id"] == selected_id_int]
        if not er.empty:
            er = er.iloc[0]
            st.markdown(f"## AI Plan for Event {int(er['event_id'])}")
            st.write(
                f"Axis: {er.get('axis', 'N/A')} | Severity: {er.get('severity', 'N/A')} | "
                f"Collision type: {er.get('collision_type', 'N/A')}"
            )

            st.markdown("### Diagnosis")
            st.write(er.get("diagnosis", ""))

            st.markdown("### Inspection Steps")
            st.text(er.get("inspection_steps", ""))

            st.markdown("### Maintenance Actions")
            st.text(er.get("maintenance_actions", ""))

            st.markdown("### Safety Clearance")
            st.text(er.get("safety_clearance", ""))

            st.markdown("### Return to Service")
            st.text(er.get("return_to_service", ""))
        else:
            st.info(
                "No AI plan found for this event. "
                "Either it is not high/critical severity or recommendations "
                "have not been generated for this ID yet."
            )
    else:
        st.info(
            "No AI recommendations found. Configure Azure and click "
            "'Generate AI recommendations for events'."
        )


if __name__ == "__main__":
    main()