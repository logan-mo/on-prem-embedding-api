import asyncio
from fastapi import FastAPI
from transformers import AutoTokenizer
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List
from pydantic import BaseModel
from threading import Lock
import torch

app = FastAPI()

# TensorRT Logger and Model Paths
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_ENGINE_PATH = "./stella_model.trt"
TOKENIZER_PATH = "./onnx_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Batch processing settings
MAX_BATCH_SIZE = 32
BATCH_TIMEOUT = 0.05  # 50ms timeout for batch processing

batch_queue = []
batch_results = {}
batch_lock = Lock()


class TextBatchRequest(BaseModel):
    texts: List[str]


# Load TensorRT Engine
def load_trt_engine(engine_path: str):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    """Allocate input/output device buffers for TensorRT execution."""
    h_input = cuda.pagelocked_empty(
        trt.volume(engine.get_binding_shape(0)), dtype=np.float32
    )
    h_output = cuda.pagelocked_empty(
        trt.volume(engine.get_binding_shape(1)), dtype=np.float32
    )

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


# TensorRT model context
engine = load_trt_engine(TRT_ENGINE_PATH)
context = engine.create_execution_context()
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)


def preprocess(texts: List[str]):
    """Tokenize input texts and return padded input tensors."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return (
        inputs["input_ids"].detach().cpu().numpy(),
        inputs["attention_mask"].detach().cpu().numpy(),
    )


def run_inference(input_ids: np.ndarray) -> List[np.ndarray]:
    """Run inference on the model with TensorRT."""
    cuda.memcpy_htod_async(d_input, input_ids, stream)
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # Return embeddings as list
    return h_output.tolist()


async def process_batch():
    """Process queued requests in batches asynchronously."""
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)

        with batch_lock:
            if not batch_queue:
                continue

            # Extract queued requests
            current_batch = batch_queue[:MAX_BATCH_SIZE]
            del batch_queue[:MAX_BATCH_SIZE]

        texts, request_ids = zip(*current_batch)

        # Preprocess and run inference
        input_ids, _ = preprocess(list(texts))
        embeddings = run_inference(input_ids)

        # Return embeddings to individual requests
        for req_id, embedding in zip(request_ids, embeddings):
            batch_results[req_id] = embedding


@app.post("/embed")
async def get_embeddings(request: TextBatchRequest):
    """Handle embedding requests by queuing them for batch processing."""
    request_id = id(request)

    with batch_lock:
        batch_queue.append((request.texts, request_id))

    # Wait for the result
    while request_id not in batch_results:
        await asyncio.sleep(0.01)

    result = batch_results.pop(request_id)
    return {"embeddings": result}


# Start the batch processing background task
asyncio.create_task(process_batch())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
