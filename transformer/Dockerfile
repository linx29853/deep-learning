FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY encoder.py decoder.py minimal_transformer.py ./
ENTRYPOINT ["python", "minimal_transformer.py"]
CMD ["--steps", "1500", "--device", "cpu"]
