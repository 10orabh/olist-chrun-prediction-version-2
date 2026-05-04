
FROM python:3.12-slim


WORKDIR /app


COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt && \
    rm -rf /root/.cache/pip


COPY connectors/ ./connectors/
COPY src/ ./src/
COPY utils/ ./utils/
COPY app.py .
COPY params.yaml .


RUN mkdir -p artifacts/models

EXPOSE 8000

#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]