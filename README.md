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
