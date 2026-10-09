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


# All data files live beside app.py.
BASE = Path(__file__).resolve().parent
ATTENDANCE = BASE / "attendance.csv"

COLUMNS = ["date", "worker_id", "hours", "site"]
CODES = ["", "8", "9", "10", "11", "AB", "P", "-"]

# Protect CSV saves when multiple managers use the same app instance.
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


# ------------------------------------------------------------
# CSV loading and validation
# ------------------------------------------------------------

def load_workers():
    workers = pd.read_csv(
        BASE / "workers.csv",
        dtype=str,
        keep_default_na=False,
    )

    required = {"worker_id", "name", "nationality", "trade", "site"}

    if not required.issubset(workers.columns):
        raise ValueError(
            "workers.csv must include: "
            + ", ".join(sorted(required))
        )

    if workers.worker_id.duplicated().any():
        raise ValueError("Worker IDs must be unique.")

    if workers[list(required)].eq("").any().any():
        raise ValueError("Worker details cannot be blank.")

    return workers


def validate_attendance(frame, workers):
    """Validate records without losing attendance codes.

    The hours column stores text: 8, 9, 10, 11, AB, P or -.
    """

    if set(frame.columns) != set(COLUMNS):
        raise ValueError(
            "Attendance CSV needs exactly: date,worker_id,hours,site"
        )

    frame = frame[COLUMNS].copy().fillna("").astype(str)

    if frame.empty:
        return frame

    parsed = pd.to_datetime(
        frame.date,
        format="%Y-%m-%d",
        errors="coerce",
    )

    if (
        parsed.isna().any()
        or not parsed.dt.strftime("%Y-%m-%d").equals(frame.date)
    ):
        raise ValueError("Dates must use YYYY-MM-DD.")

    if not frame.hours.isin(CODES[1:]).all():
        raise ValueError(
            "Hours must be 8, 9, 10, 11, AB, P or -."
        )

    if not frame.worker_id.isin(workers.worker_id).all():
        raise ValueError(
            "Attendance contains an unknown worker ID."
        )

    if not frame.site.isin(workers.site.unique()).all():
        raise ValueError("Attendance contains an unknown site.")

    if frame.duplicated(["date", "worker_id"]).any():
        raise ValueError(
            "Only one attendance record per worker per date is allowed."
        )

    return (
        frame.sort_values(["date", "site", "worker_id"])
        .reset_index(drop=True)
    )


def load_attendance(workers):
    if not ATTENDANCE.exists():
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.read_csv(
        ATTENDANCE,
        dtype=str,
        keep_default_na=False,
    )

    return validate_attendance(frame, workers)


def atomic_write(frame):
    """Replace the CSV safely instead of writing a partial file."""

    handle, temporary_name = tempfile.mkstemp(
        dir=BASE,
        suffix=".csv",
    )

    try:
        with os.fdopen(
            handle,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            frame[COLUMNS].to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_name, ATTENDANCE)

    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fingerprint(frame):
    """Identify whether saved entries changed during editing."""

    content = (
        frame[COLUMNS]
        .sort_values(["date", "worker_id"])
        .to_csv(index=False)
    )

    return hashlib.sha256(content.encode()).hexdigest()


def day_scope(frame, selected_date, worker_ids):
    return frame[
        (frame.date == selected_date)
        & frame.worker_id.isin(worker_ids)
    ]


def save_day(
    workers,
    selected_date,
    site_workers,
    codes,
    expected_fingerprint,
):
    """Update this site's entries, preserving other dates and sites."""

    with LOCK:
        current = load_attendance(workers)

        scope = day_scope(
            current,
            selected_date,
            site_workers.worker_id,
        )

        if fingerprint(scope) != expected_fingerprint:
            raise ValueError(
                "Another manager changed these entries. "
                "Click Reload saved entries, review them, "
                "and try again."
            )

        keep = current.drop(scope.index)

        rows = []

        for worker_id, site, code in zip(
            site_workers.worker_id,
            site_workers.site,
            codes,
        ):
            # A blank selection means no saved record.
            # It also clears a previously saved entry.
            if code:
                rows.append(
                    {
                        "date": selected_date,
                        "worker_id": worker_id,
                        "hours": code,
                        "site": site,
                    }
                )

        new_records = pd.DataFrame(rows, columns=COLUMNS)

        combined = pd.concat(
            [keep, new_records],
            ignore_index=True,
        )

        atomic_write(
            validate_attendance(combined, workers)
        )


# ------------------------------------------------------------
# Attendance calculations and reports
# ------------------------------------------------------------

def calculated(frame):
    frame = frame.copy()

    # AB, P, -, and blank have no known worked hours.
    frame["total_hours"] = (
        pd.to_numeric(frame.hours, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Calculate overtime per day before summing reports.
    frame["ot_hours"] = (
        frame.total_hours - 8
    ).clip(lower=0)

    frame["regular_hours"] = (
        frame.total_hours - frame.ot_hours
    )

    frame["present_days"] = frame.hours.isin(
        ["8", "9", "10", "11", "P"]
    ).astype(int)

    frame["absent_days"] = (
        frame.hours.eq("AB").astype(int)
    )

    frame["unspecified_hours_days"] = (
        frame.hours.eq("P").astype(int)
    )

    frame["not_applicable_days"] = (
        frame.hours.eq("-").astype(int)
    )

    return frame


def period_bounds(year, month):
    """Return the 21st of this month through the next month's 20th."""

    start = date(year, month, 21)

    next_month = (
        start.replace(day=28) + timedelta(days=4)
    ).replace(day=1)

    end = next_month.replace(day=20)

    return start, end


def current_period(today):
    if today.day >= 21:
        reference = today
    else:
        reference = (
            today.replace(day=1) - timedelta(days=1)
        )

    return reference.year, reference.month


def pay_report(workers, records, start, end):
    period_records = records[
        (records.date >= start.isoformat())
        & (records.date <= end.isoformat())
    ]

    rows = calculated(period_records)

    fields = [
        "total_hours",
        "regular_hours",
        "ot_hours",
        "present_days",
        "absent_days",
        "unspecified_hours_days",
        "not_applicable_days",
    ]

    totals = (
        rows.groupby("worker_id")[fields]
        .sum()
        .reset_index()
    )

    # Include workers who have no attendance records.
    report = workers.merge(
        totals,
        on="worker_id",
        how="left",
    ).fillna(0)

    report[fields] = report[fields].astype(int)

    recorded_days = rows.groupby("worker_id").size()
    period_days = (end - start).days + 1

    report["unrecorded_days"] = (
        report.worker_id.map(recorded_days)
        .fillna(0)
        .rsub(period_days)
        .astype(int)
    )

    return report


def daily_table(workers, records, selected_date):
    rows = records[records.date == selected_date]

    table = workers.merge(
        rows[["worker_id", "hours", "site"]],
        on="worker_id",
        how="left",
        suffixes=("", "_recorded"),
    )

    # Use the historical site when a record exists.
    table["site"] = table.site_recorded.fillna(
        table.site
    )

    table["hours"] = table.hours.fillna("")

    return calculated(
        table.drop(columns="site_recorded")
    )


def whatsapp_text(table, selected_date):
    lines = [
        "*SPEED RAYS — DAILY ATTENDANCE*",
        f"Date: {selected_date}",
    ]

    for site in sorted(table.site.unique()):
        group = table[table.site == site]

        present = group[group.present_days == 1]
        absent = group[group.absent_days == 1]

        lines.extend(
            [
                "",
                f"*{site}*",
                f"Present ({len(present)}):",
            ]
        )

        if present.empty:
            lines.append("None")
        else:
            for _, worker in present.iterrows():
                if worker.hours.isdigit():
                    hours_label = f"{worker.hours}h"
                else:
                    hours_label = "P (hours unspecified)"

                lines.append(
                    f"• {worker.worker_id} "
                    f"{worker['name']} — {hours_label}"
                )

        lines.append(f"Absent ({len(absent)}):")

        if absent.empty:
            lines.append("None")
        else:
            for _, worker in absent.iterrows():
                lines.append(
                    f"• {worker.worker_id} "
                    f"{worker['name']}"
                )

        lines.extend(
            [
                (
                    "Total recorded hours: "
                    f"{group.total_hours.sum()}"
                ),
                f"OT hours: {group.ot_hours.sum()}",
                (
                    "Not applicable: "
                    f"{group.not_applicable_days.sum()}"
                ),
                (
                    "Not recorded: "
                    f"{group.hours.eq('').sum()}"
                ),
            ]
        )

        unspecified = group.unspecified_hours_days.sum()

        if unspecified:
            lines.append(
                f"P with unspecified hours: {unspecified} "
                "(excluded from hours totals)"
            )

    return "\n".join(lines)


def csv_bytes(frame):
    # UTF-8 BOM helps Excel display names correctly.
    return frame.to_csv(index=False).encode("utf-8-sig")


# ------------------------------------------------------------
# Streamlit application
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Speed Rays Attendance",
        page_icon="📋",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stButton button, .stDownloadButton button {
            min-height: 48px;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📋 Speed Rays Attendance")
    st.caption(
        "Tile work & manpower supply · Dubai · "
        "Pay period: 21st–20th"
    )

    today = datetime.now(
        ZoneInfo("Asia/Dubai")
    ).date()

    try:
        workers = load_workers()
        records = load_attendance(workers)

    except (
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Cannot load CSV files: {error}")
        st.stop()

    page = st.sidebar.radio(
        "Open",
        [
            "Daily Attendance",
            "Dashboard",
            "Pay Period Report",
            "WhatsApp Export",
            "Worker Master",
            "Backup & Restore",
        ],
    )

    st.sidebar.caption(
        f"Dubai date: {today:%d %b %Y} · "
        f"{len(workers)} workers"
    )

    st.sidebar.warning(
        "Cloud CSV storage is temporary. "
        "Download a backup after every entry session."
    )

    # --------------------------------------------------------
    # Daily attendance
    # --------------------------------------------------------

    if page == "Daily Attendance":
        selected_date = st.date_input(
            "Attendance date",
            value=today,
            max_value=today,
        ).isoformat()

        site = st.radio(
            "Site",
            sorted(workers.site.unique()),
            horizontal=True,
        )

        team = workers[workers.site == site]

        scope = day_scope(
            records,
            selected_date,
            team.worker_id,
        )

        prefix = f"entry_{selected_date}_{site}"
        snapshot_key = prefix + "_snapshot"

        if st.button(
            "Reload saved entries",
            width="stretch",
        ):
            for key in list(st.session_state):
                if key.startswith(prefix):
                    del st.session_state[key]

            st.rerun()

        if snapshot_key not in st.session_state:
            st.session_state[snapshot_key] = (
                fingerprint(scope)
            )

        saved = (
            scope.set_index("worker_id")
            .hours.to_dict()
        )

        st.caption(
            "🟢 Present · 🔴 Absent · 🟠 OT. "
            "Blank = not recorded. "
            "P adds no hours. - = not applicable."
        )

        with st.form(prefix):
            codes = []

            for _, worker in team.iterrows():
                saved_code = saved.get(
                    worker.worker_id,
                    "",
                )

                code = st.selectbox(
                    (
                        f"{worker.worker_id} · "
                        f"{worker['name']} "
                        f"({worker.trade})"
                    ),
                    options=CODES,
                    index=CODES.index(saved_code),
                    key=(
                        f"{prefix}_{worker.worker_id}"
                    ),
                    format_func=lambda value: (
                        CODE_LABELS[value]
                    ),
                )

                codes.append(code)

            submitted = st.form_submit_button(
                "💾 Save attendance",
                type="primary",
                width="stretch",
            )

        if submitted:
            try:
                save_day(
                    workers,
                    selected_date,
                    team,
                    codes,
                    st.session_state[snapshot_key],
                )

                records = load_attendance(workers)

                st.session_state[snapshot_key] = (
                    fingerprint(
                        day_scope(
                            records,
                            selected_date,
                            team.worker_id,
                        )
                    )
                )

                st.success(
                    f"Saved {site} attendance for "
                    f"{selected_date}. "
                    "Download your backup below."
                )

            except (
                ValueError,
                OSError,
                TimeoutError,
            ) as error:
                st.error(str(error))

        st.download_button(
            "⬇ Download attendance backup",
            data=csv_bytes(records),
            file_name="attendance.csv",
            mime="text/csv",
            width="stretch",
        )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    elif page == "Dashboard":
        selected_date = st.date_input(
            "Summary date",
            value=today,
        ).isoformat()

        table = daily_table(
            workers,
            records,
            selected_date,
        )

        left, right = st.columns(2)

        left.metric(
            "🟢 Present",
            int(table.present_days.sum()),
        )

        right.metric(
            "🔴 Absent",
            int(table.absent_days.sum()),
        )

        left, right = st.columns(2)

        left.metric(
            "Recorded hours",
            int(table.total_hours.sum()),
        )

        right.metric(
            "🟠 OT hours",
            int(table.ot_hours.sum()),
        )

        st.caption(
            f"Not recorded: {table.hours.eq('').sum()} · "
            "Not applicable: "
            f"{table.not_applicable_days.sum()} · "
            "P without hours: "
            f"{table.unspecified_hours_days.sum()}"
        )

        summary = (
            table.groupby("site")[
                [
                    "present_days",
                    "absent_days",
                    "total_hours",
                    "ot_hours",
                    "regular_hours",
                ]
            ]
            .sum()
            .reset_index()
        )

        st.subheader("Summary by site")

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )

        chart_data = summary.melt(
            id_vars="site",
            value_vars=[
                "regular_hours",
                "ot_hours",
            ],
            var_name="Type",
            value_name="Hours",
        )

        figure = px.bar(
            chart_data,
            x="site",
            y="Hours",
            color="Type",
            barmode="stack",
            color_discrete_map={
                "regular_hours": "#22c55e",
                "ot_hours": "#f59e0b",
            },
            template="plotly_dark",
            title="Regular hours and overtime by site",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

        st.subheader("Worker details")

        st.dataframe(
            table[
                [
                    "worker_id",
                    "name",
                    "site",
                    "hours",
                    "total_hours",
                    "ot_hours",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # Pay period report
    # --------------------------------------------------------

    elif page == "Pay Period Report":
        default_year, default_month = (
            current_period(today)
        )

        left, right = st.columns(2)

        year = int(
            left.number_input(
                "Start year",
                min_value=2020,
                max_value=2100,
                value=default_year,
                step=1,
            )
        )

        month = right.selectbox(
            "Start month",
            options=list(range(1, 13)),
            index=default_month - 1,
            format_func=lambda value: (
                date(2000, value, 1).strftime("%B")
)
