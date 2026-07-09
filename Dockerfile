# Cognitive Transport Index — Python runtime for MQTT pipeline demos.
# Mosquitto runs as a separate Compose service (see docker-compose.yml).
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for scientific Python wheels / MQTT tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY pipeline/ pipeline/
COPY scripts/ scripts/
COPY analysis/ analysis/
COPY benchmarks/ benchmarks/
COPY tests/ tests/
COPY run_session.py preprocess.py pytest.ini ./

ENV PYTHONUNBUFFERED=1 \
    MQTT_BROKER=mosquitto \
    MQTT_PORT=1883 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/data

# Default: headless live demo against the Compose mosquitto service
CMD ["python", "scripts/run_live_demo.py", "--broker", "mosquitto", "--seconds", "90"]
