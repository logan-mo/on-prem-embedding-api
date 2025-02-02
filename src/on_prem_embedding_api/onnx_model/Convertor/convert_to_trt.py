import torch
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime as ort
import tensorrt as trt
import numpy as np
import os
from onnxsim import simplify


# Paths
MODEL_NAME = "NovaSearch/stella_en_1.5B_v5"  # Hugging Face model
ONNX_PATH = "./stella_model.onnx"
SIMPLIFIED_ONNX_PATH = "./stella_model_simplified.onnx"
TRT_ENGINE_PATH = "./stella_model.trt"
MAX_SEQ_LENGTH = 256  # Adjust based on your input requirements

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def export_onnx_model(model_name: str):
    """Export the Hugging Face model to ONNX."""
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Dummy input for tracing
    inputs = tokenizer(
        "Dummy text for export.",
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    input_ids = inputs["input_ids"]

    # Export to ONNX
    model.eval()
    torch.onnx.export(
        model,
        (input_ids,),
        ONNX_PATH,
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "embeddings": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"Model exported to {ONNX_PATH}")


def simplify_onnx_model(onnx_path: str, output_path: str):
    """Simplify ONNX model for better inference performance."""
    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated!")
    onnx.save(model_simplified, output_path)
    print(f"Simplified ONNX model saved to {output_path}")


def build_trt_engine(onnx_path: str, trt_engine_path: str):
    """Convert ONNX model to TensorRT engine."""
    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network,
        trt.OnnxParser(network, TRT_LOGGER) as parser,
    ):

        builder.max_batch_size = 32
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1 GB workspace
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision

        # Load the ONNX model
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Build and serialize the engine
        engine = builder.build_engine(network, config)
        if not engine:
            raise RuntimeError("Failed to build the TensorRT engine")

        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to {trt_engine_path}")


if __name__ == "__main__":
    # Step 1: Export the Hugging Face model to ONNX
    export_onnx_model(MODEL_NAME)

    # Step 2: Simplify the ONNX model
    simplify_onnx_model(ONNX_PATH, SIMPLIFIED_ONNX_PATH)

    # Step 3: Convert the simplified ONNX to TensorRT
    build_trt_engine(SIMPLIFIED_ONNX_PATH, TRT_ENGINE_PATH)
