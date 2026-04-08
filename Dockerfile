# 1. Sabse halki base image
FROM python:3.12-slim

# 2. Container ke andar workspace
WORKDIR /app

# 3. Pehle requirements copy karke install karein
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt && \
    rm -rf /root/.cache/pip

# 4. Zaroori code folders copy karein
# (Aapke app.py ke imports ke liye ye teeno zaroori hain)
COPY connectors/ ./connectors/
COPY src/ ./src/
COPY utils/ ./utils/
COPY app.py .
COPY params.yaml .

# 5. Model download hone ke liye folder structure taiyar karein
RUN mkdir -p artifacts/models

# 6. FastAPI ka port
EXPOSE 8000

# 7. Server start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]