FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY scrc ./scrc
COPY configs ./configs
COPY artifacts/production ./artifacts/production
COPY experiments/causal_diagnostics.json ./experiments/causal_diagnostics.json
COPY web ./web
RUN useradd --uid 10001 --create-home appuser && mkdir -p /app/var && chown -R appuser:appuser /app/var
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=5)"
CMD ["python", "-m", "scrc.api.serve"]
