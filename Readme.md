

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





## DVC

# Run from your project root  e.g. ASL-Recognition-System/
cd /media/neer/1E9C598B9C595DF92/Mtech/sem_2/MLops/project/ASL-Recognition-System

# If git not initialized yet:
git init
git add .
git commit -m "initial commit"

# Initialize DVC
dvc init

# This creates: .dvc/  .dvcignore
# Commit these immediately
git add .dvc .dvcignore
git commit -m "dvc: initialize"



# This is where DVC stores the actual data files (like a local S3)
mkdir -p /media/neer/1E9C598B9C595DF92/Mtech/sem_2/MLops/dvc-store

# Add it as the default remote
dvc remote add -d localstore /media/neer/1E9C598B9C595DF92/Mtech/sem_2/MLops/dvc-store

# Verify
dvc remote list
# Should print:  localstore    /media/...

git add .dvc/config
git commit -m "dvc: add local remote storage"