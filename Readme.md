

<!-- https://www.kaggle.com/datasets/aishikai/asl-alphabets -->


https://www.kaggle.com/datasets/mayank0bhardwaj/alphabets

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

mkdir -p data/raw data/processed config logs dags src/data



# From project root on your HOST machine:

# Add the generated data files to DVC tracking
dvc add data/raw
dvc add data/processed

# Push data to local DVC store
dvc push

# Commit the .dvc pointer files to Git
git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "dvc: track raw and processed data after first pipeline run"




Docker up and runing 

sudo chown -R 50000:50000 logs
chmod -R 755 logs

sudo chown -R 50000:50000 data
chmod -R 755 data


docker compose -f docker-compose.airflow.yml up airflow-init

# 5. Start services (after init exits cleanly)
docker compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler postgres

# 6. Wait ~20 seconds, then check everything is healthy
docker compose -f docker-compose.airflow.yml ps
# All three should show "running" or "healthy"

# 7. Now trigger the DAG
docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow dags trigger asl_data_ingestion



## Every time after (e.g. next day, after reboot)

# 1. Start services (skip init — already done)
docker compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler postgres

# 2. Wait ~15 seconds for scheduler to come up, then trigger
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags trigger asl_data_ingestion


If you changed params.yaml and want to retrigger

# Services already running — just trigger directly
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags trigger asl_data_ingestion
# Guard task will detect the params hash changed and run automatically


Stopping everything cleanly
# Stop but keep data (postgres volume preserved)
docker compose -f docker-compose.airflow.yml down

# Stop AND wipe everything including DB (use only if you want a clean slate)
docker compose -f docker-compose.airflow.yml down -v
# Note: if you run down -v you must redo step 4 (airflow-init) next time




## re run after config change for extracting dataset 
# Trigger the DAG
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags trigger asl_data_ingestion
