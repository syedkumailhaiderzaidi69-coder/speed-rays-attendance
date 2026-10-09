"""Speed Rays attendance.

Run locally:
    python -m streamlit run app.py
"""

from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import os
import tempfile
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st
from filelock import FileLock

BASE = Path(__file__).resolve().parent
ATTENDANCE = BASE / "attendance.csv"

COLUMNS = ["date", "worker_id", "hours", "site"]
CODES = ["", "8", "9", "10", "11", "AB", "P", "-"]

LOCK = FileLock(str(BASE / "attendance.lock"), timeout=10)

CODE_LABELS = {
    "": "Select attendance",
    "8": "🟢 8 hours",
    "9": "🟠 9 hours · 1 OT",
    "10": "🟠 10 hours · 2 OT",
    "11": "🟠 11 hours · 3 OT",
    "AB": "🔴 Absent",
    "P": "🟢 Present · hours unspecified",
    "-": "— Not applicable",
}


def load_workers():
    workers = pd.read_csv(BASE / "workers.csv", dtype=str, keep_default_na=False)
    required = {"worker_id", "name", "nationality", "trade", "site"}
    if not required.issubset(workers.columns):
        raise ValueError("workers.csv must include: " + ", ".join(sorted(required)))
    if workers.worker_id.duplicated().any():
        raise ValueError("Worker IDs must be unique.")
    if workers[list(required)].eq("").any().any():
        raise ValueError("Worker details cannot be blank.")
    return workers


def validate_attendance(frame, workers):
    if set(frame.columns) != set(COLUMNS):
        raise ValueError("Attendance CSV needs exactly: date,worker_id,hours,site")
    frame = frame[COLUMNS].copy().fillna("").astype(str)
    if frame.empty:
        return frame
    parsed = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or not parsed.dt.strftime("%Y-%m-%d").equals(frame.date):
        raise ValueError("Dates must use YYYY-MM-DD.")
    if not frame.hours.isin(CODES[1:]).all():
        raise ValueError("Hours must be 8, 9, 10, 11, AB, P or -.")
    if not frame.worker_id.isin(workers.worker_id).all():
        raise ValueError("Attendance contains an unknown worker ID.")
    if not frame.site.isin(workers.site.unique()).all():
        raise ValueError("Attendance contains an unknown site.")
    if frame.duplicated(["date", "worker_id"]).any():
        raise ValueError("Only one attendance record per worker per date is allowed.")
    return frame.sort_values(["date", "site", "worker_id"]).reset_index(drop=True)


def load_attendance(workers):
    if not ATTENDANCE.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(ATTENDANCE, dtype=str, keep_default_na=False)
    return validate_attendance(frame, workers)


def atomic_write(frame):
    handle, temporary_name = tempfile.mkstemp(dir=BASE, suffix=".csv")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            frame[COLUMNS].to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, ATTENDANCE)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fingerprint(frame):
    content = frame[COLUMNS].sort_values(["date", "worker_id"]).to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def day_scope(frame, selected_date, worker_ids):
    return frame[(frame.date == selected_date) & frame.worker_id.isin(worker_ids)]


def save_day(workers, selected_date, site_workers, codes, expected_fingerprint):
    with LOCK:
        current = load_attendance(workers)
        scope = day_scope(current, selected_date, site_workers.worker_id)
        if fingerprint(scope) != expected_fingerprint:
            raise ValueError("Another manager changed these entries. Click Reload saved entries, review them, and try again.")
        keep = current.drop(scope.index)
        rows = []
        for worker_id, site, code in zip(site_workers.worker_id, site_workers.site, codes):
            if code:
                rows.append({"date": selected_date, "worker_id": worker_id, "hours": code, "site": site})
        new_records = pd.DataFrame(rows, columns=COLUMNS)
        combined = pd.concat([keep, new_records], ignore_index=True)
        atomic_write(validate_attendance(combined, workers))


def calculated(frame):
    frame = frame.copy()
    frame["total_hours"] = pd.to_numeric(frame.hours, errors="coerce").fillna(0).astype(int)
    frame["ot_hours"] = (frame.total_hours - 8).clip(lower=0)
    frame["regular_hours"] = frame.total_hours - frame.ot_hours
    frame["present_days"] = frame.hours.isin(["8", "9", "10", "11", "P"]).astype(int)
    frame["absent_days"] = frame.hours.eq("AB").astype(int)
    frame["unspecified_hours_days"] = frame.hours.eq("P").astype(int)
    frame["not_applicable_days"] = frame.hours.eq("-").astype(int)
    return frame


def period_bounds(year, month):
    start = date(year, month, 21)
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month.replace(day=20)
    return start, end


def current_period(today):
    if today.day >= 21:
        reference = today
    else:
        reference = today.replace(day=1) - timedelta(days=1)
    return reference.year, reference.month


def pay_report(workers, records, start, end):
    period_records = records[(records.date >= start.isoformat()) & (records.date <= end.isoformat())]
    rows = calculated(period_records)
    fields = ["total_hours", "regular_hours", "ot_hours", "present_days", "absent_days", "unspecified_hours_days", "not_applicable_days"]
    totals = rows.groupby("worker_id")[fields].sum().reset_index()
    report = workers.merge(totals, on="worker_id", how="left").fillna(0)
    report[fields] = report[fields].astype(int)
    recorded_days = rows.groupby("worker_id").size()
    period_days = (end - start).days + 1
    report["unrecorded_days"] = report.worker_id.map(recorded_days).fillna(0).rsub(period_days).astype(int)
    return report


def daily_table(workers, records, selected_date):
    rows = records[records.date == selected_date]
    table = workers.merge(rows[["worker_id", "hours", "site"]], on="worker_id", how="left", suffixes=("", "_recorded"))
    table["site"] = table.site_recorded.fillna(table.site)
    table["hours"] = table.hours.fillna("")
    return calculated(table.drop(columns="site_recorded"))


def whatsapp_text(table, selected_date):
    lines = ["*SPEED RAYS — DAILY ATTENDANCE*", f"Date: {selected_date}"]
    for site in sorted(table.site.unique()):
        group = table[table.site == site]
        present = group[group.present_days == 1]
        absent = group[group.absent_days == 1]
        lines.extend(["", f"*{site}*", f"Present ({len(present)}):"])
        if present.empty:
            lines.append("None")
        else:
            for _, worker in present.iterrows():
                if worker.hours.isdigit():
                    hours_label = f"{worker.hours}h"
                else:
                    hours_label = "P (hours unspecified)"
                lines.append(f"• {worker.worker_id} {worker['name']} — {hours_label}")
        lines.append(f"Absent ({len(absent)}):")
        if absent.empty:
            lines.append("None")
        else:
            for _, worker in absent.iterrows():
                lines.append(f"• {worker.worker_id} {worker['name']}")
        lines.extend([
            f"Total recorded hours: {group.total_hours.sum()}",
            f"OT hours: {group.ot_hours.sum()}",
            f"Not applicable: {group.not_applicable_days.sum()}",
            f"Not recorded: {group.hours.eq('').sum()}",
        ])
        unspecified = group.unspecified_hours_days.sum()
        if unspecified:
            lines.append(f"P with unspecified hours: {unspecified} (excluded from hours totals)")
    return "\n".join(lines)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def main():
    st.set_page_config(page_title="Speed Rays Attendance", page_icon="📋", layout="wide")

    st.markdown("""
        <style>
        .stButton button, .stDownloadButton button { min-height: 48px; font-weight: 600; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📋 Speed Rays Attendance")
    st.caption("Tile work & manpower supply · Dubai · Pay period: 21st–20th")

    today = datetime.now(ZoneInfo("Asia/Dubai")).date()

    try:
        workers = load_workers()
        records = load_attendance(workers)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"Cannot load CSV files: {error}")
        st.stop()

    page = st.sidebar.radio("Open", ["Daily Attendance", "Dashboard", "Pay Period Report", "WhatsApp Export", "Worker Master", "Backup & Restore"])

    st.sidebar.caption(f"Dubai date: {today:%d %b %Y} · {len(workers)} workers")
    st.sidebar.warning("Cloud CSV storage is temporary. Download a backup after every entry session.")

    if page == "Daily Attendance":
        selected_date = st.date_input("Attendance date", value=today, max_value=today).isoformat()
        site = st.radio("Site", sorted(workers.site.unique()), horizontal=True)
        team = workers[workers.site == site]
        scope = day_scope(records, selected_date, team.worker_id)
        prefix = f"entry_{selected_date}_{site}"
        snapshot_key = prefix + "_snapshot"

        if st.button("Reload saved entries", use_container_width=True):
            for key in list(st.session_state):
                if key.startswith(prefix):
                    del st.session_state[key]
            st.rerun()

        if snapshot_key not in st.session_state:
            st.session_state[snapshot_key] = fingerprint(scope)

        saved = scope.set_index("worker_id").hours.to_dict()

        st.caption("🟢 Present · 🔴 Absent · 🟠 OT. Blank = not recorded. P adds no hours. - = not applicable.")

        with st.form(prefix):
            codes = []
            for _, worker in team.iterrows():
                saved_code = saved.get(worker.worker_id, "")
                code = st.selectbox(
                    f"{worker.worker_id} · {worker['name']} ({worker.trade})",
                    options=CODES,
                    index=CODES.index(saved_code),
                    key=f"{prefix}_{worker.worker_id}",
                    format_func=lambda value: CODE_LABELS[value],
                )
                codes.append(code)
            submitted = st.form_submit_button("💾 Save attendance", type="primary", use_container_width=True)

        if submitted:
            try:
                save_day(workers, selected_date, team, codes, st.session_state[snapshot_key])
                records = load_attendance(workers)
                st.session_state[snapshot_key] = fingerprint(day_scope(records, selected_date, team.worker_id))
                st.success(f"Saved {site} attendance for {selected_date}. Download your backup below.")
            except (ValueError, OSError, TimeoutError) as error:
                st.error(str(error))

        st.download_button("⬇ Download attendance backup", data=csv_bytes(records), file_name="attendance.csv", mime="text/csv", use_container_width=True)

    elif page == "Dashboard":
        selected_date = st.date_input("Summary date", value=today).isoformat()
        table = daily_table(workers, records, selected_date)
        left, right = st.columns(2)
        left.metric("🟢 Present", int(table.present_days.sum()))
        right.metric("🔴 Absent", int(table.absent_days.sum()))
        left, right = st.columns(2)
        left.metric("Recorded hours", int(table.total_hours.sum()))
        right.metric("🟠 OT hours", int(table.ot_hours.sum()))
        st.caption(f"Not recorded: {table.hours.eq('').sum()} · Not applicable: {table.not_applicable_days.sum()} · P without hours: {table.unspecified_hours_days.sum()}")
        summary = table.groupby("site")[["present_days", "absent_days", "total_hours", "ot_hours", "regular_hours"]].sum().reset_index()
        st.subheader("Summary by site")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        chart_data = summary.melt(id_vars="site", value_vars=["regular_hours", "ot_hours"], var_name="Type", value_name="Hours")
        figure = px.bar(chart_data, x="site", y="Hours", color="Type", barmode="stack", color_discrete_map={"regular_hours": "#22c55e", "ot_hours": "#f59e0b"}, template="plotly_dark", title="Regular hours and overtime by site")
        st.plotly_chart(figure, use_container_width=True)
        st.subheader("Worker details")
        st.dataframe(table[["worker_id", "name", "site", "hours", "total_hours", "ot_hours"]], hide_index=True, use_container_width=True)

    elif page == "Pay Period Report":
        default_year, default_month = current_period(today)
        left, right = st.columns(2)
        year = int(left.number_input("Start year", min_value=2020, max_value=2100, value=default_year, step=1))
        month = right.selectbox("Start month", options=list(range(1, 13)), index=default_month - 1, format_func=lambda value: date(2000, value, 1).strftime("%B"))
        start, end = period_bounds(year, month)
        st.subheader(f"Period: {start:%d %b %Y} → {end:%d %b %Y}")
        report = pay_report(workers, records, start, end)
        st.dataframe(report[["worker_id", "name", "site", "total_hours", "regular_hours", "ot_hours", "present_days", "absent_days", "unrecorded_days"]], hide_index=True, use_container_width=True)
        st.download_button("⬇ Download pay period CSV", data=csv_bytes(report), file_name=f"pay_period_{start}_{end}.csv", mime="text/csv", use_container_width=True)

    elif page == "WhatsApp Export":
        selected_date = st.date_input("Report date", value=today).isoformat()
        table = daily_table(workers, records, selected_date)
        text = whatsapp_text(table, selected_date)
        st.text_area("Copy this text", value=text, height=400)

    elif page == "Worker Master":
        site_filter = st.multiselect("Filter by site", options=sorted(workers.site.unique()), default=sorted(workers.site.unique()))
        trade_filter = st.multiselect("Filter by trade", options=sorted(workers.trade.unique()), default=sorted(workers.trade.unique()))
        filtered = workers[workers.site.isin(site_filter) & workers.trade.isin(trade_filter)]
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        st.caption(f"Showing {len(filtered)} of {len(workers)} workers")

    elif page == "Backup & Restore":
        st.subheader("Download backups")
        st.download_button("⬇ Download attendance.csv", data=csv_bytes(records), file_name="attendance.csv", mime="text/csv", use_container_width=True)
        st.download_button("⬇ Download workers.csv", data=csv_bytes(workers), file_name="workers.csv", mime="text/csv", use_container_width=True)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("attendance.csv", csv_bytes(records))
            zf.writestr("workers.csv", csv_bytes(workers))
        st.download_button("⬇ Download both CSVs (ZIP)", data=zip_buffer.getvalue(), file_name="speed_rays_backup.zip", mime="application/zip", use_container_width=True)

        st.subheader("Restore from backup")
        uploaded = st.file_uploader("Upload attendance.csv backup", type=["csv"])
        if uploaded is not None:
            backup = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            try:
                backup = validate_attendance(backup, workers)
                if st.button("Restore / merge this backup"):
                    with LOCK:
                        current = load_attendance(workers)
                        combined = pd.concat([current, backup], ignore_index=True)
                        combined = combined.drop_duplicates(["date", "worker_id"], keep="last")
                        atomic_write(validate_attendance(combined, workers))
                    st.success("Backup restored.")
                    st.rerun()
            except ValueError as error:
                st.error(str(error))

    st.sidebar.caption("Speed Rays Attendance · v1.0")


if __name__ == "__main__":
    main()
