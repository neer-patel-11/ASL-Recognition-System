

https://www.kaggle.com/datasets/aishikai/asl-alphabets


asl-recognition/
├── dags/
│   └── ingest_dag.py
├── src/data/
│   ├── download.py
│   ├── validate.py
│   ├── preprocess.py
│   └── split.py
├── logs/
├── data/
│   ├── raw/
│   └── processed/
├── docker-compose.airflow.yml
├── .env
└── requirements-airflow.txt




# 1. Make sure your .env is filled in, then:
sudo docker compose -f docker-compose.airflow.yml up airflow-init

# 2. Start all services
docker compose -f docker-compose.airflow.yml up -d

# 3. Open Airflow UI
# → http://localhost:8080
# → username: admin  password: admin

# 4. Trigger the DAG manually from UI
# 5. Check task logs in UI:
# → DAGs → asl_data_ingestion → Graph → click any task → Logs

# 6. To stop:
docker compose -f docker-compose.airflow.yml down


docker compose down -v  # old command (if still exists)
docker rm -f $(docker ps -aq)
docker volume prune -f