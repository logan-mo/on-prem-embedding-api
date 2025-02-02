# Use NVIDIA PyTorch container with TensorRT pre-installed
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set working directory
WORKDIR /workspace

# Copy project files
COPY convert_to_trt.py .
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    onnx \
    protobuf-compiler \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

# Expose TensorRT build result path
ENV TRT_ENGINE_PATH=/workspace/stella_model.trt

# Command to execute the script
CMD ["python", "convert_to_trt.py"]
