# on-prem-embedding-api

# To build the model:
docker build -t trt-builder .

On Linux
docker run --gpus all -v $(pwd):/workspace trt-builder

On Windows(cmd):
docker run --gpus all -v "%cd%":/workspace trt-builder

On Windows(powershell):
docker run --gpus all -v "${PWD}:/workspace" trt-builder

# To build the API:

docker build -t stella_api .
docker run --gpus all -p 8000:8000 stella_api


# Infinity

ssh ubuntu@10.51.100.111

import torch
torch.cuda.is_available()

ubuntu2204/arm64
sudo apt-get -y install cudnn9-cuda-1
pip install nvidia_cudnn_frontend

export LD_LIBRARY_PATH=/home/ubuntu/stella_api/TensorRT-10.8.0.43/lib

Update names
Fix documents that are on preview
Add OpenAI to document generation process



git clone https://huggingface.co/NovaSearch/stella_en_1.5B_v5


To https://github.com/logan-mo/on-prem-embedding-api.git

