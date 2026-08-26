FROM python:3.12-slim
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120
COPY Novelty/controller/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY Paper ./Paper
COPY Novelty ./Novelty
ENV PYTHONPATH=/app:/app/Paper
CMD ["uvicorn","Novelty.controller.main:app","--host","0.0.0.0","--port","8080"]
