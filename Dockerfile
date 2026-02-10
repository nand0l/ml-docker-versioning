FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py MODEL_VERSION ./

ARG MODEL_C=1.0
ARG MODEL_MAX_ITER=300

ENV MODEL_C=${MODEL_C}
ENV MODEL_MAX_ITER=${MODEL_MAX_ITER}

RUN python train.py

EXPOSE 8080

CMD ["sh", "-c", "export APP_VERSION=$(cat MODEL_VERSION) && uvicorn app:app --host 0.0.0.0 --port 8080"]
