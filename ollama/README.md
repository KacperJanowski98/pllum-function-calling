# Ollama Integration

- Conversion of fine-tuned models to GGUF format
- Creation of Ollama Modelfiles with appropriate configuration
- Function calling server for API access

## Conversion to Ollama

The process of converting the fine-tuned model to Ollama format consists of four main steps:

1. **Merge LoRA Adapters with Base Model**
2. **Convert Merged Model to GGUF Format**
3. **Create Ollama Modelfile**
4. **Deploy and Test with Ollama**

Two options are available for running the conversion:

### Option 1: Local Conversion

Run individual scripts in sequence:

```bash
# Merge LoRA adapters with base model
python merge_lora_simple.py --input-dir models/pllum-function-calling-20250330_071532 --output-dir models/pllum-function-calling-merged

# Convert to GGUF format
python convert_to_gguf.py --input-dir models/pllum-function-calling-merged --output-file models/pllum-function-calling.gguf --quantization Q8_0

# Create Ollama Modelfile
python create_modelfile_fixed.py --model-path models/pllum-function-calling.gguf --output-dir models/ollama --model-name pllum-fc
```

### Option 2: Google Colab Conversion

For systems with limited resources, a Colab notebook is provided:

1. Upload `pllum_to_ollama_colab.ipynb` to Google Colab
2. Set the runtime type to GPU
3. Mount Google Drive and specify your model path
4. Run the notebook cells in sequence
5. Download the resulting ZIP file containing all necessary files

## Installation and Usage

Once the model has been converted to GGUF format:

1. **Install Ollama** if not already installed:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Install the model in Ollama**:
   ```bash
   cd models/ollama
   ./install_model.sh
   ```

3. **Test the model**:
   ```bash
   # For regular chat
   ollama run pllum-fc

   # For function calling
   ./test_model.py --prompt "Jaka jest pogoda w Warszawie?" --function-call
   ```

4. **Run the function calling server** (optional):
   ```bash
   pip install flask requests
   python function_calling_server.py
   ```

## API Usage

The function calling server provides a REST API for making function calls:

```bash
curl -X POST http://localhost:5000/function_call \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Jaka jest pogoda w Warszawie?",
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "location": {
            "type": "string",
            "description": "The city and state or country",
            "required": true
          },
          "unit": {
            "type": "string",
            "description": "Temperature unit (celsius/fahrenheit)",
            "required": false
          }
        }
      }
    ]
  }'
```

## Troubleshooting

- **Memory issues during conversion**: Try using the Google Colab notebook instead of local conversion
- **Quantization errors**: Start with Q8_0 quantization which is more reliable, then try Q4_K_M if needed
- **CUDA errors**: Ensure you have the latest NVIDIA drivers installed
- **Ollama errors**: Make sure the Modelfile syntax is correct (use create_modelfile_fixed.py)
- **Function calling issues**: Verify the prompt format in your requests matches the expected format

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- transformers, peft, bitsandbytes
- llama.cpp dependencies (cmake, make)
- Ollama (for deployment)
