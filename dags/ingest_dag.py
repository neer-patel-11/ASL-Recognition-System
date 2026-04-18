# File: dags/ingest_dag.py
import os
import sys
import json
import hashlib
import logging
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email

sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "recipient@gmail.com")

default_args = {
    "owner": "asl-project",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="asl_data_ingestion",
    description="Download → Validate → Split → Baseline → DVC track → Email",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["asl", "data-engineering", "ingestion"],
)


# ─── TASK 0: Guard — should we even run? ─────────────────────────────────────
def task_check_should_run(**context):
    """
    Skip entire pipeline if:
      - data/raw already exists AND
      - params.yaml has not changed since last run AND
      - force_rerun is false in params.yaml
    """
    from data.config_loader import load_params, load_pipeline_config

    params = load_params()
    cfg    = load_pipeline_config()

    force = params.get("pipeline", {}).get("force_rerun", False)
    marker_file = cfg["paths"]["dvc_marker_file"]
    raw_dir     = cfg["paths"]["raw_dir"]
    params_file = cfg["paths"]["params_file"]

    # Always run if forced
    if force:
        logger.info("force_rerun=true → running pipeline")
        context["ti"].xcom_push(key="should_run", value=True)
        return

    # Always run if raw data doesn't exist yet
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        logger.info("data/raw is empty → running pipeline (first time)")
        context["ti"].xcom_push(key="should_run", value=True)
        return

    # Check if params.yaml changed since last run
    if not os.path.exists(marker_file):
        logger.info("No previous run marker found → running pipeline")
        context["ti"].xcom_push(key="should_run", value=True)
        return

    with open(marker_file) as f:
        last_run = json.load(f)

    # Hash current params.yaml
    with open(params_file, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    if current_hash != last_run.get("params_hash"):
        logger.info("params.yaml changed → running pipeline")
        context["ti"].xcom_push(key="should_run", value=True)
        return

    logger.info(
        "Data exists and params unchanged → skipping pipeline. "
        "Set force_rerun: true in params.yaml to override."
    )
    context["ti"].xcom_push(key="should_run", value=False)


# ─── TASK 1: Download ─────────────────────────────────────────────────────────
def task_download(**context):
    should_run = context["ti"].xcom_pull(key="should_run", task_ids="check_should_run")
    if not should_run:
        logger.info("Skipping download — nothing changed")
        return None

    from data.download import download_kaggle_dataset, load_train_csv

    stats = download_kaggle_dataset()
    df, csv_path = load_train_csv(stats["dataset_path"])

    context["ti"].xcom_push(key="download_stats",     value=stats)
    context["ti"].xcom_push(key="dataset_path",       value=stats["dataset_path"])
    context["ti"].xcom_push(key="csv_path",           value=csv_path)
    context["ti"].xcom_push(key="total_rows",         value=len(df))
    context["ti"].xcom_push(
        key="label_distribution",
        value=df["label"].value_counts().to_dict()
    )
    return stats


# ─── TASK 2: Validate ────────────────────────────────────────────────────────
def task_validate(**context):
    should_run = context["ti"].xcom_pull(key="should_run", task_ids="check_should_run")
    if not should_run:
        logger.info("Skipping validate")
        return None

    from data.validate import validate_dataset

    ti           = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="download_dataset")
    csv_path     = ti.xcom_pull(key="csv_path",     task_ids="download_dataset")

    results = validate_dataset(dataset_path, csv_path)
    ti.xcom_push(key="validation_results", value=results)

    fail_rate = results["failed"] / results["total"] if results["total"] > 0 else 0
    from data.config_loader import load_params
    max_fail = load_params()["validation"]["max_fail_rate"]

    if fail_rate > max_fail:
        raise ValueError(
            f"Validation failure rate {fail_rate:.1%} exceeds threshold {max_fail:.1%}"
        )
    return results


# ─── TASK 3: Split ───────────────────────────────────────────────────────────
def task_split(**context):
    should_run = context["ti"].xcom_pull(key="should_run", task_ids="check_should_run")
    if not should_run:
        logger.info("Skipping split")
        return None

    from data.split import split_dataset

    ti           = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="download_dataset")
    csv_path     = ti.xcom_pull(key="csv_path",     task_ids="download_dataset")

    split_stats = split_dataset(dataset_path=dataset_path, train_csv_path=csv_path)
    ti.xcom_push(key="split_stats", value=split_stats)
    return split_stats


# ─── TASK 4: Baseline stats ──────────────────────────────────────────────────
def task_baseline(**context):
    should_run = context["ti"].xcom_pull(key="should_run", task_ids="check_should_run")
    if not should_run:
        logger.info("Skipping baseline")
        return None

    from data.baseline_stats import compute_baseline
    from data.config_loader import load_pipeline_config

    cfg = load_pipeline_config()
    stats = compute_baseline(
        processed_dir=os.path.join(cfg["paths"]["processed_dir"], "train")
    )
    context["ti"].xcom_push(key="baseline_stats_global", value=stats.get("_global", {}))
    return stats


# File: dags/ingest_dag.py
# In task_dvc_track() — replace the subprocess section

import shutil

def task_dvc_track(**context):
    should_run = context["ti"].xcom_pull(key="should_run", task_ids="check_should_run")
    if not should_run:
        logger.info("Skipping DVC tracking — nothing changed")
        return None

    from data.config_loader import load_params, load_pipeline_config

    params = load_params()
    cfg    = load_pipeline_config()

    raw_dir       = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    params_file   = cfg["paths"]["params_file"]
    marker_file   = cfg["paths"]["dvc_marker_file"]

    # ── Find dvc binary explicitly ────────────────────────────────────────
    dvc_bin = shutil.which("dvc")
    if dvc_bin is None:
        # fallback: common pip install locations inside container
        for candidate in [
            "/home/airflow/.local/bin/dvc",
            "/usr/local/bin/dvc",
            "/usr/bin/dvc",
        ]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                dvc_bin = candidate
                break

    if dvc_bin is None:
        raise RuntimeError(
            "dvc binary not found in container. "
            "Make sure 'dvc' is in the pip install line of docker-compose."
        )

    logger.info(f"Using dvc binary: {dvc_bin}")

    # ── dvc add ───────────────────────────────────────────────────────────
    for path in [raw_dir, processed_dir]:
        if os.path.exists(path):
            result = subprocess.run(
                [dvc_bin, "add", path],
                capture_output=True, text=True, cwd="/opt/airflow"
            )
            if result.returncode != 0:
                logger.warning(f"dvc add {path} stderr: {result.stderr}")
            else:
                logger.info(f"dvc add {path} → OK\n{result.stdout.strip()}")
        else:
            logger.warning(f"Path not found, skipping dvc add: {path}")

    # ── dvc push ──────────────────────────────────────────────────────────
    push_result = subprocess.run(
        [dvc_bin, "push"],
        capture_output=True, text=True, cwd="/opt/airflow"
    )
    logger.info(f"dvc push stdout: {push_result.stdout.strip()}")
    if push_result.returncode != 0:
        logger.warning(f"dvc push stderr: {push_result.stderr}")

    # ── Write marker file ─────────────────────────────────────────────────
    with open(params_file, "rb") as f:
        params_hash = hashlib.md5(f.read()).hexdigest()

    marker = {
        "params_hash":    params_hash,
        "last_run_utc":   datetime.utcnow().isoformat(),
        "dataset_handle": params["data"]["dataset_handle"],
        "dvc_bin_used":   dvc_bin,
        "split_ratios": {
            "train": params["split"]["train_ratio"],
            "val":   params["split"]["val_ratio"],
        },
    }
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    with open(marker_file, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"DVC marker written: {marker_file}")
    context["ti"].xcom_push(key="dvc_marker", value=marker)
    return marker

# ─── TASK 6: Email report ────────────────────────────────────────────────────
def task_send_report(**context):
    ti = context["ti"]

    should_run     = ti.xcom_pull(key="should_run",          task_ids="check_should_run")
    download_stats = ti.xcom_pull(key="download_stats",      task_ids="download_dataset")
    validation     = ti.xcom_pull(key="validation_results",  task_ids="validate_data")
    split_stats    = ti.xcom_pull(key="split_stats",         task_ids="split_dataset")
    label_dist     = ti.xcom_pull(key="label_distribution",  task_ids="download_dataset")
    baseline       = ti.xcom_pull(key="baseline_stats_global", task_ids="compute_baseline")
    dvc_marker     = ti.xcom_pull(key="dvc_marker",          task_ids="dvc_track")

    exec_date = context["execution_date"]
    run_id    = context["run_id"]

    if not should_run:
        send_email(
            to=[ALERT_EMAIL],
            subject=f"[ASL Pipeline] ⏭ Skipped — No Changes Detected ({exec_date.strftime('%Y-%m-%d %H:%M')})",
            html_content=f"""
            <html><body style="font-family:Arial,sans-serif">
              <h2 style="color:#888">⏭ Pipeline Skipped</h2>
              <p>Data already exists and <code>params.yaml</code> has not changed since last run.</p>
              <p>To force a rerun, set <code>force_rerun: true</code> in <code>params.yaml</code>.</p>
              <p style="color:#aaa;font-size:12px">Run ID: {run_id}</p>
            </body></html>
            """,
        )
        return

    # Build label table
    label_rows = "".join(
        f"<tr><td style='padding:4px 12px'>{l}</td><td style='padding:4px 12px'>{c}</td></tr>"
        for l, c in sorted((label_dist or {}).items())
    )

    # Build split table
    split_rows = ""
    if split_stats and "splits" in split_stats:
        for name, info in split_stats["splits"].items():
            split_rows += (
                f"<tr><td style='padding:4px 12px'>{name}</td>"
                f"<td style='padding:4px 12px'>{info['count']}</td>"
                f"<td style='padding:4px 12px'>{info['copied']}</td></tr>"
            )

    dvc_hash    = (dvc_marker or {}).get("params_hash", "N/A")
    dvc_time    = (dvc_marker or {}).get("last_run_utc", "N/A")
    b_mean      = (baseline   or {}).get("mean",     "N/A")
    b_var       = (baseline   or {}).get("variance", "N/A")
    dl_images   = (download_stats or {}).get("image_count",       "N/A")
    dl_time     = (download_stats or {}).get("download_time_sec", "N/A")
    val_total   = (validation or {}).get("total",  "N/A")
    val_passed  = (validation or {}).get("passed", "N/A")
    val_failed  = (validation or {}).get("failed", "N/A")

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#333;max-width:700px">
      <h2 style="color:#1a6eb5">✅ ASL Data Pipeline — Run Complete</h2>
      <p><b>Run ID:</b> {run_id}<br><b>Date:</b> {exec_date}</p>

      <h3 style="color:#178c5a">📥 Download</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Images downloaded</td><td style="padding:4px 12px">{dl_images}</td></tr>
        <tr><td style="padding:4px 12px">Download time (s)</td><td style="padding:4px 12px">{dl_time}</td></tr>
      </table>

      <h3 style="color:#178c5a">✔️ Validation</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Total checked</td><td style="padding:4px 12px">{val_total}</td></tr>
        <tr><td style="padding:4px 12px">Passed</td><td style="padding:4px 12px">{val_passed}</td></tr>
        <tr><td style="padding:4px 12px">Failed</td><td style="padding:4px 12px">{val_failed}</td></tr>
      </table>

      <h3 style="color:#178c5a">✂️ Split (70/15/15)</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff">
          <th style="padding:4px 12px">Split</th>
          <th style="padding:4px 12px">Count</th>
          <th style="padding:4px 12px">Copied</th>
        </tr>
        {split_rows}
      </table>

      <h3 style="color:#178c5a">📊 Label Distribution</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Label</th><th style="padding:4px 12px">Count</th></tr>
        {label_rows}
      </table>

      <h3 style="color:#178c5a">📈 Baseline Stats</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Global pixel mean</td><td style="padding:4px 12px">{b_mean}</td></tr>
        <tr><td style="padding:4px 12px">Global pixel variance</td><td style="padding:4px 12px">{b_var}</td></tr>
      </table>

      <h3 style="color:#178c5a">🗃 DVC Tracking</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">params.yaml hash</td><td style="padding:4px 12px"><code>{dvc_hash}</code></td></tr>
        <tr><td style="padding:4px 12px">Tracked at (UTC)</td><td style="padding:4px 12px">{dvc_time}</td></tr>
      </table>

      <br>
      <p style="color:#aaa;font-size:12px">
        Airflow DAG: <b>asl_data_ingestion</b> | {exec_date}
      </p>
    </body></html>
    """

    send_email(
        to=[ALERT_EMAIL],
        subject=f"[ASL Pipeline] ✅ Data Ingestion Complete — {exec_date.strftime('%Y-%m-%d %H:%M')}",
        html_content=html,
    )
    logger.info(f"Report sent to {ALERT_EMAIL}")


# ─── Wire up tasks ────────────────────────────────────────────────────────────

t0_check    = PythonOperator(task_id="check_should_run",  python_callable=task_check_should_run, dag=dag)
t1_download = PythonOperator(task_id="download_dataset",  python_callable=task_download,          dag=dag)
t2_validate = PythonOperator(task_id="validate_data",     python_callable=task_validate,          dag=dag)
t3_split    = PythonOperator(task_id="split_dataset",     python_callable=task_split,             dag=dag)
t4_baseline = PythonOperator(task_id="compute_baseline",  python_callable=task_baseline,          dag=dag)
t5_dvc      = PythonOperator(task_id="dvc_track",         python_callable=task_dvc_track,         dag=dag)
t6_email    = PythonOperator(task_id="send_email_report", python_callable=task_send_report,       dag=dag)

t0_check >> t1_download >> t2_validate >> t3_split >> t4_baseline >> t5_dvc >> t6_email