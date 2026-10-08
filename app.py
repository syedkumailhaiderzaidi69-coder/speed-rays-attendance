"""Speed Rays attendance — run with: python -m streamlit run app.py"""
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


def load_workers():
    workers = pd.read_csv(BASE / "workers.csv", dtype=str, keep_default_na=False)
    required = {"worker_id", "name", "nationality", "trade", "site"}
    if not required.issubset(workers.columns):
        raise ValueError("workers.csv must include: " + ", ".join(sorted(required)))
    if workers.worker_id.duplicated().any() or workers[list(required)].eq("").any().any():
        raise ValueError("Worker IDs must be unique and worker details cannot be blank.")
    return workers


def validate_attendance(frame, workers):
    """The hours field stores the original code, including AB, P and -."""
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
    return validate_attendance(
        pd.read_csv(ATTENDANCE, dtype=str, keep_default_na=False), workers
    )


def atomic_write(frame):
    """A replace avoids partial CSV files if a save is interrupted."""
    handle, name = tempfile.mkstemp(dir=BASE, suffix=".csv")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            frame[COLUMNS].to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, ATTENDANCE)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def fingerprint(frame):
    content = frame[COLUMNS].sort_values(["date", "worker_id"]).to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def day_scope(frame, selected_date, ids):
    return frame[(frame.date == selected_date) & frame.worker_id.isin(ids)]


def save_day(workers, selected_date, site_workers, codes, expected):
    """Upsert this site's workers; preserve other dates/sites and detect conflicts."""
    with LOCK:
        current = load_attendance(workers)
        scope = day_scope(current, selected_date, site_workers.worker_id)
        if fingerprint(scope) != expected:
            raise ValueError("Another manager changed these entries. Reload saved entries and try again.")
        keep = current.drop(scope.index)
        rows = [dict(date=selected_date, worker_id=wid, hours=code, site=site)
                for wid, site, code in zip(site_workers.worker_id, site_workers.site, codes)
                if code]  # Blank deletes a previous record, rather than marking absent.
        combined = pd.concat([keep, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
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
    return start, next_month.replace(day=20)


def current_period(today):
    reference = today if today.day >= 21 else today.replace(day=1) - timedelta(days=1)
    return reference.year, reference.month


def pay_report(workers, records, start, end):
    rows = calculated(records[(records.date >= start.isoformat()) & (records.date <= end.isoformat())])
    fields = ["total_hours", "regular_hours", "ot_hours", "present_days", "absent_days",
              "unspecified_hours_days", "not_applicable_days"]
    totals = rows.groupby("worker_id")[fields].sum().reset_index()
    report = workers.merge(totals, on="worker_id", how="left").fillna(0)
    report[fields] = report[fields].astype(int)
    counts = rows.groupby("worker_id").size()
    report["unrecorded_days"] = report.worker_id.map(counts).fillna(0).rsub((end-start).days+1).astype(int)
    return report


def daily_table(workers, records, selected_date):
    rows = records[records.date == selected_date]
    table = workers.merge(rows[["worker_id", "hours", "site"]], on="worker_id", how="left", suffixes=("", "_recorded"))
    # Historical records keep the site saved on that date.
    table["site"] = table.site_recorded.fillna(table.site)
    table["hours"] = table.hours.fillna("")
    return calculated(table.drop(columns="site_recorded"))


def whatsapp_text(table, selected_date):
    lines = ["*SPEED RAYS — DAILY ATTENDANCE*", f"Date: {selected_date}"]
    for site in sorted(table.site.unique()):
        group = table[table.site == site]
        present = group[group.present_days == 1]
        absent = group[group.absent_days == 1]
        lines += ["", f"*{site}*", f"Present ({len(present)}):"]
        lines += [f"• {r.worker_id} {r['name']} — {r.hours + 'h' if r.hours.isdigit() else 'P (hours unspecified)'}"
                  for _, r in present.iterrows()] or ["None"]
        lines += [f"Absent ({len(absent)}):"]
        lines += [f"• {r.worker_id} {r['name']}" for _, r in absent.iterrows()] or ["None"]
        lines += [f"Total recorded hours: {group.total_hours.sum()}", f"OT hours: {group.ot_hours.sum()}",
                  f"Not applicable: {group.not_applicable_days.sum()}", f"Not recorded: {group.hours.eq('').sum()}"]
        if group.unspecified_hours_days.sum():
            lines += [f"P with unspecified hours: {group.unspecified_hours_days.sum()} (excluded from hours totals)"]
    return "\n".join(lines)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def main():
    st.set_page_config(page_title="Speed Rays Attendance", page_icon="📋", layout="wide")
    st.markdown("""<style>
    .stButton button, .stDownloadButton button {min-height:48px; font-weight:600;}
    [data-testid="stMetricValue"] {font-size:1.8rem;}
    </style>""", unsafe_allow_html=True)
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
        selected = st.date_input("Attendance date", today, max_value=today).isoformat()
        site = st.radio("Site", sorted(workers.site.unique()), horizontal=True)
        team = workers[workers.site == site]
        scope = day_scope(records, selected, team.worker_id)
        prefix = f"entry_{selected}_{site}"
        token = prefix + "_snapshot"
        if st.button("Reload saved entries", width="stretch"):
            for key in list(st.session_state):
                if key.startswith(prefix):
                    del st.session_state[key]
            st.rerun()
        if token not in st.session_state:
            st.session_state[token] = fingerprint(scope)
        saved = scope.set_index("worker_id").hours.to_dict()
        st.caption("🟢 Present · 🔴 Absent · 🟠 OT. Blank = not recorded. P adds no hours. - = not applicable.")
        with st.form(prefix):
            codes = []
            for _, worker in team.iterrows():
                code = saved.get(worker.worker_id, "")
                codes.append(st.selectbox(
                    f"{worker.worker_id} · {worker['name']} ({worker.trade})", CODES,
                    index=CODES.index(code), key=f"{prefix}_{worker.worker_id}",
                    format_func=lambda c: {"":"Select attendance", "8":"🟢 8 hours", "9":"🟠 9 hours · 1 OT",
                        "10":"🟠 10 hours · 2 OT", "11":"🟠 11 hours · 3 OT", "AB":"🔴 Absent",
                        "P":"🟢 Present · hours unspecified", "-":"— Not applicable"}[c]))
            submit = st.form_submit_button("💾 Save attendance", type="primary", width="stretch")
        if submit:
            try:
                save_day(workers, selected, team, codes, st.session_state[token])
                records = load_attendance(workers)
                st.session_state[token] = fingerprint(day_scope(records, selected, team.worker_id))
                st.success(f"Saved {site} attendance for {selected}. Download your backup below.")
            except (ValueError, OSError, TimeoutError) as error:
                st.error(str(error))
        st.download_button("⬇ Download attendance backup", csv_bytes(records), "attendance.csv", "text/csv", width="stretch")

    elif page == "Dashboard":
        selected = st.date_input("Summary date", today).isoformat()
        table = daily_table(workers, records, selected)
        a, b = st.columns(2)
        a.metric("🟢 Present", int(table.present_days.sum()))
        b.metric("🔴 Absent", int(table.absent_days.sum()))
        a, b = st.columns(2)
        a.metric("Recorded hours", int(table.total_hours.sum()))
        b.metric("🟠 OT hours", int(table.ot_hours.sum()))
        st.caption(f"Not recorded: {table.hours.eq('').sum()} · Not applicable: {table.not_applicable_days.sum()} · P without hours: {table.unspecified_hours_days.sum()}")
        summary = table.groupby("site")[["present_days", "absent_days", "total_hours", "ot_hours", "regular_hours"]].sum().reset_index()
        st.dataframe(summary, hide_index=True, width="stretch")
        chart = summary.melt(id_vars="site", value_vars=["regular_hours", "ot_hours"], var_name="Type", value_name="Hours")
        st.plotly_chart(px.bar(chart, x="site", y="Hours", color="Type", barmode="stack",
                              color_discrete_map={"regular_hours":"#22c55e", "ot_hours":"#f59e0b"}, template="plotly_dark"), width="stretch")
        st.dataframe(table[["worker_id", "name", "site", "hours", "total_hours", "ot_hours"]], hide_index=True, width="stretch")

    elif page == "Pay Period Report":
        year, month = current_period(today)
        a, b = st.columns(2)
        year = int(a.number_input("Start year", min_value=2020, max_value=2100, value=year, step=1))
        month = b.selectbox("Start month", list(range(1,13)), index=month-1, format_func=lambda m: date(2000,m,1).strftime("%B"))
        start, end = period_bounds(year, month)
        st.info(f"{start:%d %b %Y} to {end:%d %b %Y} (inclusive)")
        site = st.selectbox("Site filter", ["All"] + sorted(workers.site.unique()))
        report = pay_report(workers, records, start, end)
        if site != "All":
            report = report[report.site == site]
        st.caption("P counts as present with zero known hours. Missing dates are unrecorded, not absent. Unrecorded days include future dates in an unfinished period. Site filter uses the worker master site.")
        st.dataframe(report, hide_index=True, width="stretch")
        export = report.assign(period_start=start.isoformat(), period_end=end.isoformat())
        st.download_button("⬇ Export pay period CSV", csv_bytes(export), f"pay_period_{start}_{end}.csv", "text/csv", width="stretch")

    elif page == "WhatsApp Export":
        selected = st.date_input("Report date", today).isoformat()
        text = whatsapp_text(daily_table(workers, records, selected), selected)
        st.caption("Select and copy the text below, then paste it into WhatsApp.")
        st.text_area("Daily WhatsApp report", text, height=500)
        st.download_button("⬇ Download report text", text.encode(), f"daily_report_{selected}.txt", "text/plain", width="stretch")

    elif page == "Worker Master":
        site = st.selectbox("Site", ["All"] + sorted(workers.site.unique()))
        trade = st.selectbox("Trade", ["All"] + sorted(workers.trade.unique()))
        filtered = workers.copy()
        if site != "All":
            filtered = filtered[filtered.site == site]
        if trade != "All":
            filtered = filtered[filtered.trade == trade]
        st.metric("Workers", len(filtered))
        st.dataframe(filtered, hide_index=True, width="stretch")
        st.download_button("⬇ Download worker list", csv_bytes(filtered), "workers.csv", "text/csv")

    else:
        st.warning("Streamlit Cloud does not provide durable local CSV storage. Save backups to your computer after each entry session. Saving here does not update GitHub.")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("workers.csv", csv_bytes(workers))
            archive.writestr("attendance.csv", csv_bytes(records))
        st.download_button("⬇ Download both CSVs (ZIP)", buffer.getvalue(), f"attendance_backup_{today}.zip", "application/zip", width="stretch")
        st.subheader("Restore attendance from a backup")
        st.caption("Upload attendance.csv. Rows are merged by date + worker ID; uploaded rows replace matching saved rows. Other saved rows stay. Restore worker master changes manually in workers.csv.")
        uploaded = st.file_uploader("Attendance CSV", type="csv")
        if uploaded:
            try:
                incoming = validate_attendance(pd.read_csv(uploaded, dtype=str, keep_default_na=False), workers)
                st.dataframe(incoming, hide_index=True, width="stretch")
                if st.button("Restore / merge this backup", type="primary", width="stretch"):
                    with LOCK:
                        latest = load_attendance(workers)
                        merged = pd.concat([latest, incoming], ignore_index=True).drop_duplicates(["date", "worker_id"], keep="last")
                        atomic_write(validate_attendance(merged, workers))
                    st.success("Backup restored. Open Dashboard to review the records.")
            except (ValueError, OSError, TimeoutError, pd.errors.ParserError) as error:
                st.error(f"Restore failed: {error}")


if __name__ == "__main__":
    main()
