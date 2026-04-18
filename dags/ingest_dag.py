# File: dags/ingest_dag.py
import os
import sys
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email

# Make src/ importable inside Airflow container
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "patelneer403@gmail.com")

default_args = {
    "owner": "asl-project",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": True,   # Airflow sends auto email on task failure
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="asl_data_ingestion",
    description="Download, validate, split ASL dataset and send email report",
    default_args=default_args,
    schedule_interval=None,     # Trigger manually or via API
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["asl", "data-engineering", "ingestion"],
)


# ─── Task 1: Download ────────────────────────────────────────────────────────

def task_download(**context):
    from data.download import download_kaggle_dataset, load_train_csv

    stats = download_kaggle_dataset(dest_dir="/opt/airflow/data/raw")
    df, csv_path = load_train_csv(stats["dataset_path"])

    # Push to XCom so later tasks can use it
    context["ti"].xcom_push(key="download_stats", value=stats)
    context["ti"].xcom_push(key="dataset_path", value=stats["dataset_path"])
    context["ti"].xcom_push(key="csv_path", value=csv_path)
    context["ti"].xcom_push(key="total_rows", value=len(df))
    context["ti"].xcom_push(
        key="label_distribution",
        value=df["label"].value_counts().to_dict()
    )

    logger.info(f"Download complete: {stats}")
    return stats


# ─── Task 2: Validate ────────────────────────────────────────────────────────

def task_validate(**context):
    from data.validate import validate_dataset

    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="download_dataset")
    csv_path = ti.xcom_pull(key="csv_path", task_ids="download_dataset")

    results = validate_dataset(dataset_path, csv_path)

    ti.xcom_push(key="validation_results", value=results)

    # Fail the task if too many errors (> 5%)
    total = results["total"]
    failed = results["failed"]
    fail_rate = failed / total if total > 0 else 0

    if fail_rate > 0.05:
        raise ValueError(
            f"Validation failure rate too high: {fail_rate:.1%} "
            f"({failed}/{total} images failed)"
        )

    logger.info(f"Validation passed: {results['passed']}/{total}")
    return results


# ─── Task 3: Split ───────────────────────────────────────────────────────────

def task_split(**context):
    from data.split import split_dataset

    ti = context["ti"]
    dataset_path = ti.xcom_pull(key="dataset_path", task_ids="download_dataset")
    csv_path = ti.xcom_pull(key="csv_path", task_ids="download_dataset")

    split_stats = split_dataset(
        dataset_path=dataset_path,
        train_csv_path=csv_path,
        output_dir="/opt/airflow/data/processed",
    )

    ti.xcom_push(key="split_stats", value=split_stats)
    logger.info(f"Split complete: {split_stats}")
    return split_stats


# ─── Task 4: Compute Baseline Stats ──────────────────────────────────────────

def task_baseline(**context):
    from data.baseline_stats import compute_baseline
    import sys
    sys.path.insert(0, "/opt/airflow")

    stats = compute_baseline(processed_dir="/opt/airflow/data/processed/train")
    context["ti"].xcom_push(key="baseline_stats_global", value=stats.get("_global", {}))
    logger.info("Baseline stats computed.")
    return stats


# ─── Task 5: Send Email Report ───────────────────────────────────────────────

def task_send_report(**context):
    ti = context["ti"]

    download_stats = ti.xcom_pull(key="download_stats",       task_ids="download_dataset")
    validation     = ti.xcom_pull(key="validation_results",   task_ids="validate_data")
    split_stats    = ti.xcom_pull(key="split_stats",           task_ids="split_dataset")
    label_dist     = ti.xcom_pull(key="label_distribution",   task_ids="download_dataset")
    baseline       = ti.xcom_pull(key="baseline_stats_global", task_ids="compute_baseline")

    run_id = context["run_id"]
    exec_date = context["execution_date"]

    # Build label distribution table rows
    label_rows = ""
    if label_dist:
        for label, count in sorted(label_dist.items()):
            label_rows += f"<tr><td style='padding:4px 12px'>{label}</td><td style='padding:4px 12px'>{count}</td></tr>"

    # Build split summary rows
    split_rows = ""
    if split_stats and "splits" in split_stats:
        for split_name, info in split_stats["splits"].items():
            split_rows += (
                f"<tr>"
                f"<td style='padding:4px 12px'>{split_name}</td>"
                f"<td style='padding:4px 12px'>{info['count']}</td>"
                f"<td style='padding:4px 12px'>{info['copied']}</td>"
                f"</tr>"
            )

    val_passed = validation.get("passed", "N/A") if validation else "N/A"
    val_failed = validation.get("failed", "N/A") if validation else "N/A"
    val_total  = validation.get("total",  "N/A") if validation else "N/A"
    dl_time    = download_stats.get("download_time_sec", "N/A") if download_stats else "N/A"
    dl_images  = download_stats.get("image_count", "N/A") if download_stats else "N/A"
    b_mean     = baseline.get("mean", "N/A") if baseline else "N/A"
    b_var      = baseline.get("variance", "N/A") if baseline else "N/A"

    html_content = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333; max-width: 700px;">
      <h2 style="color: #1a6eb5;">✅ ASL Data Ingestion Pipeline — Run Report</h2>
      <p><b>Run ID:</b> {run_id}<br>
         <b>Execution date:</b> {exec_date}</p>

      <h3 style="color:#178c5a;">📥 Download Summary</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Total images downloaded</td><td style="padding:4px 12px">{dl_images}</td></tr>
        <tr><td style="padding:4px 12px">Download time (sec)</td><td style="padding:4px 12px">{dl_time}</td></tr>
      </table>

      <h3 style="color:#178c5a;">✔️ Validation Summary</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Total checked</td><td style="padding:4px 12px">{val_total}</td></tr>
        <tr><td style="padding:4px 12px">Passed</td><td style="padding:4px 12px" style="color:green">{val_passed}</td></tr>
        <tr><td style="padding:4px 12px">Failed</td><td style="padding:4px 12px">{val_failed}</td></tr>
      </table>

      <h3 style="color:#178c5a;">✂️ Dataset Split (70/15/15)</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff">
          <th style="padding:4px 12px">Split</th>
          <th style="padding:4px 12px">Count</th>
          <th style="padding:4px 12px">Files Copied</th>
        </tr>
        {split_rows}
      </table>

      <h3 style="color:#178c5a;">📊 Label Distribution (train.csv)</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Label</th><th style="padding:4px 12px">Count</th></tr>
        {label_rows}
      </table>

      <h3 style="color:#178c5a;">📈 Baseline Stats (global)</h3>
      <table border="1" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f4ff"><th style="padding:4px 12px">Metric</th><th style="padding:4px 12px">Value</th></tr>
        <tr><td style="padding:4px 12px">Global pixel mean</td><td style="padding:4px 12px">{b_mean}</td></tr>
        <tr><td style="padding:4px 12px">Global pixel variance</td><td style="padding:4px 12px">{b_var}</td></tr>
      </table>

      <br>
      <p style="color:#888; font-size:12px;">
        Sent by Airflow DAG: <b>asl_data_ingestion</b> | {exec_date}
      </p>
    </body></html>
    """

    send_email(
        to=[ALERT_EMAIL],
        subject=f"[ASL Pipeline] s Data Ingestion Complete — {exec_date.strftime('%Y-%m-%d %H:%M')}",
        html_content=html_content,
    )
    logger.info(f"Report email sent to {ALERT_EMAIL}")



t1_download = PythonOperator(
    task_id="download_dataset",
    python_callable=task_download,
    dag=dag,
)

t2_validate = PythonOperator(
    task_id="validate_data",
    python_callable=task_validate,
    dag=dag,
)

t3_split = PythonOperator(
    task_id="split_dataset",
    python_callable=task_split,
    dag=dag,
)

t4_baseline = PythonOperator(
    task_id="compute_baseline",
    python_callable=task_baseline,
    dag=dag,
)

t5_email = PythonOperator(
    task_id="send_email_report",
    python_callable=task_send_report,
    dag=dag,
)

t1_download >> t2_validate >> t3_split >> t4_baseline >> t5_email