from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Set default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'cfa_data_scrape_dag',
    default_args=default_args,
    description='DAG to scrape data from CFA Institute Publications and upload to S3 and Snowflake',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Define the BashOperator task
scrape_task = BashOperator(
    task_id='scrape_data',
    bash_command="python /opt/airflow/dags/cfa_pdfs_images_s3_sf.py",
    dag=dag,
)