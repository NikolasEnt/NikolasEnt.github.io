---
layout: post
title: "Robust MLflow Deployment: From Localhost to Production"
title_img: /assets/post26/logo.webp
abstract: >
    A practical guide to upgrading MLOps infrastructure by dockerizing MLflow, decoupling storage with PostgreSQL and MinIO, and performing a migration from local SQLite to a production-ready setup.
date: 2025-10-26 23:00:00 +0000
categories: [MLOps, Infrastructure, Docker]
article: true
sitemap:
    lastmod: 2025-10-26
---

Experiments tracking is one of the keys to success in ML tasks as it allows structuring experimentation and keeping logs of experiments for future analysis. As proposed in [AI-project-template](https://github.com/NikolasEnt/AI-project-template), MLflow is a convenient way to maintain a structured experiment log. In addition, [MLflow](https://mlflow.org/) offers many useful MLOps functionalities, including models' lifecycle management and automations. This post covers various aspects of MLflow server deployment and its infrastructure, from simple local setup for a solo developer to robust cloud deployment.

Managing machine learning experiments with MLflow typically starts with a local deployment and a default `./mlruns` folder. This works for solo prototyping, but once parallel training or team collaboration are introduced, the file-based backend becomes a bottleneck. SQLite locks up concurrent writes and local artifact storage makes network sharing of models and artifacts challenging.

In this post, we will build a production-grade MLflow infrastructure from a simple local setup for a solo developer to a robust cloud deployment with migration between stages. We will cover:
1.  **Dockerization**: Building a flexible server image.
2.  **Orchestration**: A `docker compose` stack with PostgreSQL and MinIO.
3.  **Migration**: Moving from SQLite to PostgreSQL without losing performance.
4.  **Security**: Nginx reverse proxy with large file support.

## 1. The Containerized Setup

To ensure reproducibility and DevOps simplicity, we avoid installing MLflow directly on the host system. Instead, we bundle the server and its dependencies into a lightweight container.
## Dockerfile

This image is designed to be backend-agnostic. It supports both local file storage (for quick tests) and remote backends (Postgres/S3) via environment variables.

```docker
FROM python:3.11-slim

# Use the version defined in .env or default to a specific tag
ARG MLFLOW_VERSION=3.7.0
ARG PORT=5000

WORKDIR /mlflow/

RUN pip install --no-cache-dir \
    mlflow==$MLFLOW_VERSION \
    psycopg2-binary==2.9.11 \
    boto3==1.40.59 \
    cryptography==46.0.3

EXPOSE ${PORT}

# Defaults
ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server \
    --backend-store-uri ${BACKEND_URI} \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination ${ARTIFACT_ROOT} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Makefile

We can use a `Makefile` to standardize run commands and avoid copying long Docker commands strings.

```makefile
PROJECT_NAME=mlflow
IMAGE_NAME=$(PROJECT_NAME)
CONTAINER_NAME=--name=$(PROJECT_NAME)
port=5001

.PHONY: all build stop run logs

all: build stop run logs

build:
	docker build --build-arg PORT=$(port) -t $(IMAGE_NAME) -f Dockerfile .

stop:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

run:
	docker run -d --restart unless-stopped -it \
		-p $(port):5000 \
		-v $(shell pwd)/data:/mlflow \
		$(CONTAINER_NAME) $(IMAGE_NAME)

logs:
	docker logs -f $(PROJECT_NAME)
```

## 2. The Production Architecture (Docker Compose)

We need to decouple the components:
1.  **Tracking Server**: Stateless API handler - MLflow server.
2.  **Metadata Store**: PostgreSQL (replaces SQLite local database).
3.  **Artifact Store**: MinIO (S3-compatible).

Best practices include:

1.  **Secret Management**: Use a `.env` file data with secrets management in docker compose.
2.  **Health Checks**: Use `healthcheck` blocks in Postgres and MinIO. The `mlflow` and `createbuckets` services wait until their dependencies are actually ready to accept connections, preventing 'Connection Refused' crash loops at startup.
4.  **Network Isolation**: Kept the custom bridge network.

---

#### The Secrets (`.env`)
Create a `.env` file in the same directory. Do not commit this file to Git, add it to `.gitignore`.

```bash
# Postgres
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=ChangeMe_PgPassword_123
POSTGRES_DB=mlflow

# MinIO (S3)
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=ChangeMe_MinioPassword_123

# MLflow
MLFLOW_PORT=5001
MLFLOW_VERSION=3.7.0
```

#### docker-compose.yml
The following `docker-compose.yml` file uses health checks and environment variables with secrets.

```yaml
name: mlflow
services:
  db:
    image: postgres:18-alpine
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      PGDATA: /var/lib/postgresql/data/pgdata
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - mlflow-net

  minio:
    image: minio/minio:RELEASE.2025-09-07T16-13-09Z
    restart: always
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD_FILE: /run/secrets/minio_password
    secrets:
      - minio_password
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - mlflow-net

  # One-off job to create the bucket automatically
  createbuckets:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    environment:
      MINIO_ROOT_USER: minio_admin
    secrets:
      - minio_password

    entrypoint:
      - /bin/sh
      - -c
      - |
        MINIO_PASS=$$(cat /run/secrets/minio_password)
        /usr/bin/mc alias set myminio http://minio:9000 ${MINIO_ROOT_USER} $$MINIO_PASS
        /usr/bin/mc mb --ignore-existing myminio/mlflow
        /usr/bin/mc anonymous set download myminio/mlflow
    networks:
      - mlflow-net

  mlflow:
    build:
      context: .
      args:
        MLFLOW_VERSION: ${MLFLOW_VERSION}
    restart: always
    depends_on:
      db:
        condition: service_healthy
      minio:
        condition: service_healthy
      createbuckets:
        condition: service_completed_successfully
    ports:
      - "${MLFLOW_PORT:-5000}:5000"
    secrets:
      - db_password
      - minio_password
    environment:
      ARTIFACT_ROOT: s3://mlflow/
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_DB: ${POSTGRES_DB}
      MLFLOW_DISABLE_TELEMETRY: true
      MLFLOW_SQLALCHEMYSTORE_POOL_SIZE: 5
      MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW: 5
      MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE: 1800
      MLFLOW_SQLALCHEMYSTORE_ECHO: false
      MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE: true
    command:
      - /bin/sh
      - -c
      - |
        export BACKEND_URI="postgresql://${POSTGRES_USER}:$$(cat /run/secrets/db_password)@db:5432/${POSTGRES_DB}"
        echo "$$BACKEND_URI"
        export AWS_SECRET_ACCESS_KEY="$$(cat /run/secrets/minio_password)"
        exec mlflow server \
          --backend-store-uri "$$BACKEND_URI" \
          --default-artifact-root "mlflow-artifacts:/" \
          --artifacts-destination "$$ARTIFACT_ROOT" \
          --serve-artifacts \
          --host 0.0.0.0 \
          --port 5000
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s
    networks:
      - mlflow-net

networks:
  mlflow-net:
    driver: bridge

volumes:
  postgres_data:
  minio_data:

secrets:
  db_password:
    environment: "POSTGRES_PASSWORD"
  minio_password:
    environment: "MINIO_ROOT_PASSWORD"
```

The `MLFLOW_S3_ENDPOINT_URL` in the `docker-compose.yml` file is used by the MLflow server (to display artifacts in the UI). The same environment variable must also be set on the client for direct upload only, while the config above proxies artifacts via MLflow server.

We use `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true` (equivalent to `--disable-security-middleware` in the MLflow server start command) because the Nginx reverse proxy manages connection security, however, in more complex cases it is worth reading the official MLflow security guide: [Protect Your Tracking Server from Network Exposure](https://mlflow.org/docs/latest/self-hosting/security/network/). Particularly, it is important to set `--allowed-hosts` properly.

## 3. Migration: SQLite to PostgreSQL

A common issue when migrating from SQLite is using tools like [`pgloader`](https://github.com/dimitri/pgloader) in "auto-magic" mode. While this transfers the data, it often fails to create the correct indexes and foreign key constraints that MLflow expects. This leads to extremely slow query performance on the MLflow frontend, so we will use Schema-first migration.

Instead of asking `pgloader` to do everything by itself, we can let MLflow create the perfect schema first, then fill it with data.

* Step 1: Initialize the Schema:

Start the `docker-compose` stack once. The MLflow container will connect to an empty Postgres DB and apply the latest tables and indexes.
Check logs to confirm: `docker logs -f mlflow`.

* Step 2: Migrate Data Only

Create a `pgloader` script file named `migration.load`:

```lisp
LOAD DATABASE
     FROM sqlite:///data/mlflow.db
     INTO postgresql://mlflow:ChangeMe_PgPassword_123@db:5432/mlflow

WITH data only, truncate, reset no sequences, downcase identifiers
SET PostgreSQL PARAMETERS session_replication_role = 'replica', work_mem = '16MB', search_path = 'public';
```

* Step 3: Run the Migration

Run `pgloader` using a temporary container (connecting to the docker compose network to reach Postgres):

```bash
docker compose down mlflow
docker compose exec db psql -U mlflow -d mlflow -c "GRANT ALL ON SCHEMA public TO mlflow;"
docker run --rm -it --network mlflow_mlflow-net \
    -v $(pwd)/data:/data \
    -v $(pwd)/migration.load:/migration.load \
    ghcr.io/dimitri/pgloader:latest \
    pgloader /migration.load
docker compose up -d mlflow
```

This ensures your new Postgres database has the exact indexes MLflow requires for performance.


## 4. Security & Nginx Configuration

Original MLflow lacks built-in authentication. We can use Nginx as a reverse proxy to handle Basic Auth. The following command can be used to create a password:

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd mlflow_username  # Then type password when prompted
```

**Important**: It is important to increase the `client_max_body_size` value. The default is usually not sufficient, which causes uploads of large artifacts (like PyTorch checkpoints) to fail with `413 Request Entity Too Large` exception.

**Minimal Nginx Config:**

```nginx
server {
    listen 80;
    server_name mlops.internal;

    # Allow large artifacts uploads
    client_max_body_size 10G;

    location /mlflow/ {
        proxy_pass http://localhost:15554/;
        auth_basic "Restricted Access";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
    }
}
```

*Tip:* If serving under a path like `/mlflow/`, you may need to add `--static-prefix /mlflow` to the MLflow start command in Dockerfile to ensure assets load correctly.

Using Nginx enables more complex setups as the service scales, including load balancing and more advanced security features.

*Note:* Given the current situation with Nginx, it can be beneficial to consider using Nginx alternatives, many of which use the same config structure and offer extra features. MinIO alternatives, such us RustFS and Garage are worth considering too.


## 5. Client-Side Usage

To interact with this stack, Python client needs to know MLflow endpoint only as we proxy artifacts via MLflow server.
Set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` environmental variables with Mlflow server username and password as set in Nginx config.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")  # Update to the server address

# model = ... 
with mlflow.start_run():
    mlflow.log_param("epochs", 10)
    mlflow.pytorch.log_model(model, "model")
```

While proxying artifacts is highly secure (since clients never access the storage backend directly), it can become a bottleneck. If multiple artifacts are uploaded simultaneously, the MLflow server's memory and network become the bottleneck. In such cases, it is highly recommended to use direct uploads. In this setup, the MLflow server only handles lightweight metadata. For artifacts, it hands the client a storage URI, and the client uploads directly to MinIO/S3. For this:

1. Update the MLflow server command in `docker-compose.yml`:

```bash
        exec mlflow server \
          --backend-store-uri "$$BACKEND_URI" \
          --default-artifact-root "$$ARTIFACT_ROOT" \
          --host 0.0.0.0 \
          --port 5000
```

2. Configure the client to upload directly. To do this, specify environment variables with the storage URI and credentials:

* `MLFLOW_S3_ENDPOINT_URL` - URI of the MinIO/S3 server
* `AWS_ACCESS_KEY_ID` - Value from `MINIO_ROOT_USER` in this example
* `AWS_SECRET_ACCESS_KEY` - Value from `MINIO_ROOT_PASSWORD` in this example


### Transitioning to Cloud

This architecture is cloud-ready. For example, to use services such as:
1.  **AWS S3**: Remove the MinIO service and update `MLFLOW_S3_ENDPOINT_URL`. Set `ARTIFACT_ROOT` to `s3://my-prod-bucket/`.
2.  **Azure**: Use `wasbs://` scheme. Install `azure-storage-blob` in the Dockerfile.
3.  **Managed Postgres**: Simply update the `BACKEND_URI` to point to RDS or Azure Database for PostgreSQL.






