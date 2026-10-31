FROM python:3.11-slim

WORKDIR /app

# nodejs + npm: QA for Express / Next.js apps (node --check, npm test, next build).
RUN apt-get update && apt-get install -y --no-install-recommends curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY workspace.py .
COPY server.py .
COPY tools ./tools

EXPOSE 9000

CMD ["python3", "server.py"]
