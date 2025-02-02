FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy model and code
COPY onnx_model/ /app/onnx_model
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
