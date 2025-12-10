from datetime import timedelta
import json
import os
from typing import Union  # Added this import

import pandas as pd

from src.config import (
    ERROR_LOGS_PARSED,
    SYSTEM_ALERTS_PARSED,
    MAINT_NOTES_PARSED,
    TORQUE_CYCLES_CLEAN,
    EVENTS_FILE,
    TORQUE_MEDIUM_THRESHOLD,
    TORQUE_CRITICAL_THRESHOLD,
    REPEAT_WINDOW_HOURS,
    VALIDATION_DIR,
)

# Simple mapping from torque % to Newtons for scoring (document this in your write-up)
MAX_FORCE_N = 10_000.0  # acceptable range per spec; used as rated equivalent


# FIXED: Changed type hint to use Union for compatibility
def _load_csv(path: Union[str, bytes, "os.PathLike"]):
    try:
        # Check if file exists first to avoid pandas errors on missing files
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


def _attach_torque_cycles(events: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    if events.empty or cycles.empty:
        events["cycle_id"] = pd.NA
        events["peak_torque_pct"] = pd.NA
        return events

    # Ensure datetime
    for col in ("cycle_start", "cycle_end"):
        if col in cycles.columns:
            cycles[col] = pd.to_datetime(cycles[col], errors="coerce", utc=True)

    out_rows = []
    for _, ev in events.iterrows():
        ts = ev["timestamp"]
        if pd.isna(ts):
            out_rows.append(ev)
            continue

        matches = cycles[
            (cycles["cycle_start"] <= ts) & (cycles["cycle_end"] >= ts)
        ]

        if matches.empty:
            out_rows.append(ev)
            continue

        match = matches.iloc[0]
        ev = ev.copy()
        ev["cycle_id"] = match.get("cycle_id")
        ev["peak_torque_pct"] = match.get("peak_torque_pct")

        # If axis was unknown (0) but cycle axis exists, infer it
        if ev.get("axis", 0) in (0, None, pd.NA) and not pd.isna(match.get("axis")):
            ev["axis"] = int(match["axis"])
            ev["axis_source"] = "from_torque_cycle"
        out_rows.append(ev)

    return pd.DataFrame(out_rows)


def _attach_nearest_alert(events: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    if events.empty or alerts.empty:
        for col in ("alert_level", "alert_type", "alert_message"):
            if col not in events.columns:
                events[col] = pd.NA
        return events

    alerts = alerts.copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], errors="coerce", utc=True)

    severity_order = {"CRITICAL": 4, "ALERT": 3, "WARN": 2, "NOTICE": 1, "INFO": 0}

    out_rows = []
    window = timedelta(seconds=30)

    for _, ev in events.iterrows():
        ts = ev["timestamp"]
        if pd.isna(ts):
            out_rows.append(ev)
            continue

        window_mask = (alerts["timestamp"] >= ts - window) & (
            alerts["timestamp"] <= ts + window
        )
        nearby = alerts[window_mask]
        if nearby.empty:
            out_rows.append(ev)
            continue

        nearby = nearby.assign(
            _sev_score=nearby["alert_level"]
            .fillna("")
            .str.upper()
            .map(severity_order)
            .fillna(-1)
        ).sort_values(["_sev_score", "timestamp"], ascending=[False, True])

        best = nearby.iloc[0]
        ev = ev.copy()
        ev["alert_level"] = best.get("alert_level")
        ev["alert_type"] = best.get("alert_type")
        ev["alert_message"] = best.get("alert_message")
        out_rows.append(ev)

    return pd.DataFrame(out_rows)


def _attach_last_maintenance(events: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    if events.empty or maint.empty:
        events["last_maintenance_date"] = pd.NaT
        events["last_maintenance_task"] = pd.NA
        events["days_since_last_maintenance"] = pd.NA
        return events

    maint = maint.copy()
    if "date" in maint.columns:
        maint["date"] = pd.to_datetime(maint["date"], errors="coerce").dt.date

    out_rows = []

    for _, ev in events.iterrows():
        axis = ev.get("axis", None)
        ts = ev["timestamp"]
        if axis is None or pd.isna(ts):
            out_rows.append(ev)
            continue

        ev_date = ts.date()
        subset = maint[(maint["axis"] == axis) & (maint["date"] <= ev_date)]
        if subset.empty:
            out_rows.append(ev)
            continue

        last = subset.sort_values("date").iloc[-1]
        ev = ev.copy()
        ev["last_maintenance_date"] = last["date"]
        ev["last_maintenance_task"] = last["task_type"]
        ev["days_since_last_maintenance"] = (ev_date - last["date"]).days
        out_rows.append(ev)

    return pd.DataFrame(out_rows)


def _compute_severity(row: pd.Series) -> str:
    msg = str(row.get("message_raw", "") or "").lower()
    p = row.get("peak_torque_pct", None)
    alert_level = str(row.get("alert_level", "") or "").upper()

    # Direct collision / e-stop take priority
    if "collision" in msg or "e-stop" in msg or "estop" in msg:
        if p is not None and not pd.isna(p) and p >= TORQUE_CRITICAL_THRESHOLD:
            return "critical"
        return "high"

    # Use torque percentage thresholds when available
    if p is not None and not pd.isna(p):
        if p >= TORQUE_CRITICAL_THRESHOLD:
            return "critical"
        if p >= TORQUE_MEDIUM_THRESHOLD:
            return "medium"

    # Fallback to alert level
    if alert_level == "CRITICAL":
        return "critical"
    if alert_level in ("ALERT", "WARN"):
        return "medium"
    if alert_level in ("NOTICE", "INFO"):
        return "low"

    return "low"


def _classify_collision_type(row: pd.Series) -> str:
    msg = str(row.get("message_raw", "") or "").lower()
    code = str(row.get("error_code", "") or "")

    if "collision" in msg:
        return "hard_impact"
    if "torque limit" in msg:
        return "torque_limit"
    if "overtravel" in msg:
        return "overtravel"
    if "singularity" in msg:
        return "path_singularity"
    if "fence open" in msg:
        return "safety_fence"
    if "e-stop" in msg or "estop" in msg:
        return "emergency_stop"
    if code.startswith("SRVO"):
        return "servo_fault"
    if code.startswith("MOTN"):
        return "motion_fault"
    return "other"


def _compute_location(axis: Union[int, float, None]) -> str:
    try:
        ax = int(axis)
    except (TypeError, ValueError):
        ax = 0
    if ax <= 0:
        return "J0"
    return f"J{ax}"


def _compute_confidence_and_notes(row: pd.Series) -> tuple[str, str]:
    """
    Map the upstream metadata into a single confidence_flag + notes for each event.
    """
    reasons: list[str] = []
    flag = "high"

    ts_source = str(row.get("timestamp_source", "") or "")
    axis_source = str(row.get("axis_source", "") or "log")

    if ts_source == "time_only_default_date":
        reasons.append("Timestamp date inferred from DEFAULT_LOG_DATE")
        flag = "medium"
    elif ts_source == "missing":
        reasons.append("Timestamp missing in source logs")
        flag = "low"

    if axis_source == "from_torque_cycle":
        reasons.append("Axis inferred from torque cycle match")
        if flag == "high":
            flag = "inferred"
    elif axis_source == "unknown":
        reasons.append("Axis unknown; set to 0 (J0)")
        if flag == "high":
            flag = "medium"

    if pd.isna(row.get("peak_torque_pct")):
        reasons.append("No peak_torque_pct available for matching cycle")
        if flag == "high":
            flag = "medium"

    if not reasons:
        return "high", "Fully observed; no imputation applied"

    return flag, "; ".join(reasons)


def build_events() -> pd.DataFrame:
    # 1) Load all inputs
    errors = pd.read_csv(ERROR_LOGS_PARSED, parse_dates=["timestamp"])
    alerts = pd.read_csv(SYSTEM_ALERTS_PARSED, parse_dates=["timestamp"])
    maint = pd.read_csv(MAINT_NOTES_PARSED)
    cycles = pd.read_csv(TORQUE_CYCLES_CLEAN)

    # Ensure timestamp column exists in errors (for older parsed files)
    if "timestamp" not in errors.columns:
        raise SystemExit("ERROR_LOGS_PARSED must contain a 'timestamp' column.")

    # 2) Filter "interesting" error events
    interesting_mask = errors["message_raw"].str.contains(
        "collision|torque limit|overtravel|singularity|e-stop|fence open",
        case=False,
        na=False,
    )
    events = errors[interesting_mask].copy()

    total_interesting = len(events)
    missing_ts_mask = events["timestamp"].isna()
    dropped_missing_ts = int(missing_ts_mask.sum())

    # Track discard stats for documentation
    stats = {
        "total_error_rows": int(len(errors)),
        "interesting_error_rows": int(total_interesting),
        "dropped_missing_timestamp": dropped_missing_ts,
    }
    
    # Ensure directory exists before writing stats
    if not VALIDATION_DIR.exists():
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        
    (VALIDATION_DIR / "event_build_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    # 3) Drop rows without timestamps (but we just logged how many)
    events = events[~missing_ts_mask].reset_index(drop=True)

    # 4) Axis handling – always have an axis column, default 0 for unknown
    if "axis" not in events.columns:
        events["axis"] = 0

    events["axis"] = pd.to_numeric(events["axis"], errors="coerce").fillna(0).astype(int)
    events["axis_source"] = "log"
    events.loc[events["axis"] <= 0, "axis_source"] = "unknown"

    # 5) Attach torque cycle context (cycle_id, peak_torque_pct, axis inference)
    events = _attach_torque_cycles(events, cycles)

    # 6) Attach nearest system alert within 30 seconds
    events = _attach_nearest_alert(events, alerts)

    # 7) Attach last maintenance for that axis
    events = _attach_last_maintenance(events, maint)

    # 8) Compute severity
    events["severity"] = events.apply(_compute_severity, axis=1)

    # 9) Compute repeats within REPEAT_WINDOW_HOURS
    events = events.sort_values("timestamp").reset_index(drop=True)

    repeats: list[int] = []
    window = timedelta(hours=REPEAT_WINDOW_HOURS)
    for i, ev in events.iterrows():
        ts = ev["timestamp"]
        axis = ev.get("axis", None)
        code = ev.get("error_code", None)

        if pd.isna(ts) or axis is None or code is None:
            repeats.append(0)
            continue

        window_start = ts - window
        prior = events.iloc[:i]
        mask = (
            (prior["timestamp"] >= window_start)
            & (prior["timestamp"] <= ts)
            & (prior["axis"] == axis)
            & (prior["error_code"] == code)
        )
        repeats.append(int(mask.sum()))

    events["repeats_24h"] = repeats

    # 10) Collision type classification
    events["collision_type"] = events.apply(_classify_collision_type, axis=1)

    # 11) Location from axis
    events["location"] = events["axis"].apply(_compute_location)

    # 12) Force value in Newtons, derived from peak_torque_pct proxy
    if "peak_torque_pct" in events.columns:
        pct = pd.to_numeric(events["peak_torque_pct"], errors="coerce")
        force = (pct / 100.0) * MAX_FORCE_N
        events["force_value"] = force.clip(lower=0.0, upper=MAX_FORCE_N)
    else:
        events["force_value"] = pd.NA

    # 13) Status column for downstream user (all start pending)
    events["status"] = "pending_inspection"

    # 14) Confidence + notes
    conf_flags = []
    notes = []
    for _, row in events.iterrows():
        flag, note = _compute_confidence_and_notes(row)
        conf_flags.append(flag)
        notes.append(note)

    events["confidence_flag"] = conf_flags
    events["notes"] = notes

    # 15) Add event_id as simple index
    events = events.reset_index(drop=True)
    events["event_id"] = events.index + 1

    # Reorder columns: event_id first and main fields up front
    first_cols = [
        "event_id",
        "timestamp",
        "location",
        "axis",
        "collision_type",
        "error_code",
        "error_group",
        "severity",
        "force_value",
        "repeats_24h",
        "cycle_id",
        "peak_torque_pct",
        "alert_level",
        "alert_type",
        "last_maintenance_date",
        "last_maintenance_task",
        "days_since_last_maintenance",
        "status",
        "confidence_flag",
        "notes",
        "message_raw",
        "alert_message",
    ]
    cols = [c for c in first_cols if c in events.columns] + [
        c for c in events.columns if c not in first_cols
    ]
    events = events[cols]

    # Debug print of key fields
    debug_cols = [
        "event_id",
        "timestamp",
        "axis",
        "location",
        "error_code",
        "severity",
        "collision_type",
        "peak_torque_pct",
        "force_value",
        "repeats_24h",
        "confidence_flag",
    ]
    existing_debug_cols = [c for c in debug_cols if c in events.columns]
    print(events[existing_debug_cols].head(10))

    events.to_csv(EVENTS_FILE, index=False)
    return events


if __name__ == "__main__":
    df = build_events()
    print(f"Built {len(df)} events -> {EVENTS_FILE}")