from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModel
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.intel.neural_compressor import INCModel

model_name = "NovaSearch/stella_en_1.5B_v5"
onnx_output_path = "./onnx_model"

# Load Model for embedding extraction
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save in ONNX format for optimized inference
model.save_pretrained(onnx_output_path)
tokenizer.save_pretrained(onnx_output_path)

# Optimize with ONNX Runtime
optimized_model = ORTModel.from_pretrained(onnx_output_path)
optimization_config = OptimizationConfig(optimization_level=2)
optimized_model.optimize(optimization_config)
print("Embedding model optimized successfully.")


quantized_model = INCModel.from_pretrained(onnx_model_path, export=True)
quantized_model.save_pretrained("./onnx_quantized_model")
