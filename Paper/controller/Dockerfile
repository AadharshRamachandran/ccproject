FROM python:3.12-slim
WORKDIR /app
COPY Paper/controller/requirements.txt ./requirements.txt
# Explicitly select the official CPU-only wheel. The controller does not train
# models, and therefore does not need the CUDA packages pulled by PyPI torch.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu \
    && pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt
COPY Paper ./Paper
COPY Novelty ./Novelty
RUN ln -s /app/Paper/src /app/src && ln -s /app/Paper/controller /app/controller && mkdir -p /app/Paper/models
CMD ["uvicorn","controller.main:app","--host","0.0.0.0","--port","8080"]
