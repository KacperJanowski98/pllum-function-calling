# Function Calling Dataset Analysis and PLLuM Fine-tuning

This project provides tools for analyzing the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset, a collection of 60,000 function calling examples created with the APIGen pipeline. It includes functionality to translate queries to Polish for fine-tuning the PLLuM model, and implements fine-tuning of the PLLuM 8B model for function calling tasks using QLoRA and the Unsloth framework.

## Overview

The dataset contains 60,000 data points collected by APIGen, an automated data generation pipeline designed to produce verifiable high-quality datasets for function-calling applications. Each data point is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring reliability and correctness.

According to human evaluation of 600 sampled data points, the dataset has a correctness rate above 95% (with the remaining 5% having minor issues like inaccurate arguments).

## Project Structure

```
.
├── .env                  # Environment variables (add your HF token here)
├── .gitignore            # Git ignore file
├── data/                 # Directory for generated datasets
│   └── .gitkeep          # Placeholder to ensure directory is tracked
├── docs/                 # Documentation directory
│   └── instruction.md    # Project instructions
├── models/               # Directory for storing fine-tuned models
│   └── .gitkeep          # Placeholder to ensure directory is tracked
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # Project documentation
├── src/                  # Source code
│   ├── __init__.py       # Makes src a package
│   ├── auth.py           # Hugging Face authentication utilities
│   ├── dataset.py        # Dataset loading and processing utilities
│   ├── fine_tuning.py    # Utilities for fine-tuning the PLLuM model
│   ├── translator.py     # Utilities for translating text
│   └── translation_dataset.py  # Create translated datasets
└── notebooks/            # Jupyter notebooks
    ├── create_translated_dataset.ipynb  # Create a dataset with Polish translations
    ├── dataset_exploration.ipynb        # Notebook for exploring the dataset
    ├── fine_tuning.ipynb                # Notebook for fine-tuning PLLuM model
    ├── pllum_fine_tune_colab.ipynb      # Notebook for fine-tuning PLLuM model on google colab
    ├── test_model.ipynb                 # Test notebook for the fine-tuned model
    └── translation_test.ipynb           # Test notebook for translation
```

## Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Hugging Face account with access to the dataset
- NVIDIA GPU (RTX 4060 or better) with CUDA support
- CUDA Toolkit (11.8 or compatible version)

### Installation

1. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

2. Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Install the rest of the dependencies:
```bash
uv pip install -e .
```

4. Install CUDA-specific optimizations (optional but recommended):
```bash
uv pip install -e ".[cuda]"
```

5. Create a `.env` file with your Hugging Face token:
```bash
cp .env.example .env
# Edit .env to add your Hugging Face token
```

6. Verify CUDA availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Usage

### Loading the Dataset

```python
from src.dataset import load_function_calling_dataset

# Load the dataset
dataset = load_function_calling_dataset()

# Access a sample
sample = dataset['train'][0]
print(sample)
```

### Translating Queries to Polish

```python
from src.translator import translate_text, translate_query_in_sample

# Translate a single text
english_text = "Find the nearest restaurant to my location"
polish_text = translate_text(english_text, src='en', dest='pl')
print(polish_text)

# Translate a query in a dataset sample
from src.dataset import load_function_calling_dataset, parse_json_entry

dataset = load_function_calling_dataset()
sample = dataset['train'][0]
translated_sample = translate_query_in_sample(sample, src='en', dest='pl')
parsed = parse_json_entry(translated_sample)
print(f"Translated query: {parsed['query']}")
```

### Creating a Translated Dataset

```python
from src.translation_dataset import create_translated_dataset

# Create a dataset with 40% queries translated to Polish
output_path = "data/translated_dataset.json"
translated_dataset_path = create_translated_dataset(
    output_path=output_path,
    sample_size=1000,  # Optional: limit dataset size
    translation_percentage=0.4,
    random_seed=42
)
print(f"Dataset created at: {translated_dataset_path}")
```

### Fine-tuning PLLuM for Function Calling

The project includes functionality to fine-tune the [CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct) model for function calling:

```python
from src.fine_tuning import (
    PLLuMFineTuningConfig,
    setup_model_and_tokenizer,
    prepare_dataset,
    train_model
)

# Configure fine-tuning parameters
config = PLLuMFineTuningConfig(
    model_name_or_path="CYFRAGOVPL/Llama-PLLuM-8B-instruct",
    output_dir="models/pllum-function-calling",
    dataset_path="data/translated_dataset.json",
    # QLoRA settings for efficient training on consumer GPUs
    use_4bit=True,
    lora_r=16,
    lora_alpha=32
)

# Load model and tokenizer with QLoRA and Unsloth optimizations
model, tokenizer = setup_model_and_tokenizer(config)

# Prepare dataset for training
train_dataset = prepare_dataset(
    dataset_path=config.dataset_path,
    tokenizer=tokenizer,
    max_length=config.max_seq_length
)

# Train the model
trained_model = train_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    config=config
)
```

### Using the Fine-tuned Model

```python
from src.fine_tuning import (
    load_fine_tuned_model,
    generate_function_call
)

# Load a fine-tuned model
model, tokenizer = load_fine_tuned_model("models/pllum-function-calling")

# Define tools
weather_tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state or country",
                "required": True
            },
            "unit": {
                "type": "string",
                "description": "Unit of temperature: 'celsius' or 'fahrenheit'",
                "required": False
            }
        }
    }
]

# Generate function call for a query
query = "Jaka jest pogoda w Warszawie?"  # Polish query: "What's the weather in Warsaw?"
function_call = generate_function_call(
    model=model,
    tokenizer=tokenizer,
    query=query,
    tools=weather_tools,
    temperature=0.1
)

print(function_call)
```

### Exploring with Notebooks

Start Jupyter to explore the notebooks:

```bash
jupyter notebook
```

Available notebooks:
- `notebooks/dataset_exploration.ipynb` - Examples of working with the dataset
- `notebooks/translation_test.ipynb` - Testing the translation functionality
- `notebooks/create_translated_dataset.ipynb` - Creating a dataset with Polish translations
- `notebooks/fine_tuning.ipynb` - Fine-tuning PLLuM 8B with QLoRA and Unsloth
- `notebooks/test_model.ipynb` - Testing the fine-tuned model for function calling

## Fine-tuning Features

The project includes functionality to fine-tune the PLLuM 8B model for function calling:

- `src/fine_tuning.py` - Utilities for fine-tuning the PLLuM model with QLoRA and Unsloth
- Optimized for consumer GPUs (tested on NVIDIA RTX 4060)
- Using 4-bit quantization for memory efficiency
- Support for both English and Polish queries
- Customizable hyperparameters via the `PLLuMFineTuningConfig` class
- Generation utilities for using the fine-tuned model

### Fine-tuning Hardware Environment

The fine-tuning process was performed using Google Colab with an H100 GPU, as the local hardware proved insufficient for training the PLLuM model efficiently. This enabled faster training times and allowed handling larger batch sizes than would be possible with consumer-grade GPUs.

### Fine-tuning Experiments

Two fine-tuning runs were conducted:
1. A smaller run with 1000 examples in the dataset
2. A larger run with 5000 examples in the dataset

For the larger run with 5000 examples, Unsloth reported the following training details:
```
trainer = Trainer(
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 5,000 | Num Epochs = 4 | Total steps = 624
O^O/ \_/ \    Batch size per device = 8 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (8 x 4 x 1) = 32
 "-____-"     Trainable parameters = 41,943,040/8,000,000,000 (0.52% trained)
Unsloth: Will smartly offload gradients to save VRAM!
```

This shows that only 0.52% of the model's parameters (~42M out of 8B) were being trained thanks to the LoRA approach, with Unsloth providing additional VRAM optimization through smart gradient offloading.

5k runs used the following hyperparameters:

```json
{
  "model_name_or_path": "CYFRAGOVPL/Llama-PLLuM-8B-instruct",
  "output_dir": "/content/pllum-function-calling-output/models/pllum-function-calling-20250330_071532",
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4",
  "use_nested_quant": false,
  "num_train_epochs": 4,
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "max_grad_norm": 0.3,
  "max_steps": -1,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine",
  "logging_steps": 25,
  "save_steps": 100,
  "save_total_limit": 1,
  "max_seq_length": 1024,
  "dataset_path": "/content/pllum-function-calling-output/data/xlam_function_calling_pl.json",
  "padding": "max_length",
  "pad_to_multiple_of": 8,
  "use_cuda": true,
  "device_map": "auto"
}
```

1k runs used the following hyperparameters:
```json
{
  "model_name_or_path": "CYFRAGOVPL/Llama-PLLuM-8B-instruct",
  "output_dir": "/content/pllum-function-calling-output/models/pllum-function-calling-20250328_204133",
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4",
  "use_nested_quant": false,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.0002,
  "weight_decay": 0.01,
  "max_grad_norm": 0.3,
  "max_steps": -1,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine",
  "logging_steps": 10,
  "save_steps": 200,
  "save_total_limit": 1,
  "max_seq_length": 1024,
  "dataset_path": "/content/pllum-function-calling-output/data/xlam_function_calling_pl.json",
  "padding": "max_length",
  "pad_to_multiple_of": 8,
  "use_cuda": true,
  "device_map": "auto"
}
```

These settings were optimized for the H100 GPU environment in Google Colab, allowing for larger batch sizes (8) than would be possible on consumer hardware like the RTX 4060.

## CUDA Configuration

This project relies heavily on CUDA for GPU acceleration. The implementation:

- Uses PyTorch with CUDA support for tensor operations
- Implements QLoRA 4-bit quantization to reduce GPU memory requirements
- Leverages the Unsloth framework for optimized training on consumer GPUs
- Includes CUDA-specific optimizations (triton, flash-attention) as optional dependencies

For optimal performance on an RTX 4060:
- Set `per_device_train_batch_size=4` (or lower if you encounter OOM errors)
- Enable gradient accumulation (recommended: `gradient_accumulation_steps=2`)
- Use 4-bit quantization (`use_4bit=True`)
- Adjust LoRA ranks based on available memory (`lora_r=16` is a good starting point)

## Translation Features

The project includes functionality to translate dataset queries from English to Polish:

- `src/translator.py` - Utilities for translating text using the Googletrans library
- `src/translation_dataset.py` - Functions for creating datasets with translated queries
- Ability to create a dataset with a specified percentage (default 40%) of queries translated to Polish
- Preservation of original dataset structure, with only the query field translated

## Data Format

Each entry in the dataset follows this JSON format:

- `query` (string): The query or problem statement
- `tools` (array): Available tools to solve the query
  - Each tool has `name`, `description`, and `parameters`
- `answers` (array): Corresponding answers showing which tools were used with what arguments

## Subsequent Tasks

The following tasks are planned for further development of this project:

### Deploying with Ollama

The fine-tuned PLLuM model will be deployed using Ollama, which provides an easy-to-use interface for running large language models locally. This will allow for testing the model's function calling capabilities in a production-like environment.

### Integration with MCP Servers

The model will be integrated with MCP servers using the [mcp-cli repository](https://github.com/KacperJanowski98/mcp-cli/tree/main). This integration will allow for:
- Scalable deployment of the fine-tuned model
- Testing the model's performance in a distributed environment
- Evaluating real-world latency and throughput metrics
- Comparing performance between local and server-based deployments

### Model Optimization for Edge Devices
To enable deployment on resource-constrained edge devices, the fine-tuned model will undergo further optimization:
- Quantization to lower precision formats (INT8, INT4) to reduce model size and memory footprint
- Pruning to remove unnecessary weights while maintaining accuracy
- Knowledge distillation to create smaller, faster models that retain function calling capabilities
- Exploration of model compilation techniques to optimize inference speed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset
- [APIGen pipeline](https://apigen-pipeline.github.io/) for dataset generation
- [CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct) model
- [Unsloth](https://github.com/unslothai/unsloth) framework for optimized LLM fine-tuning
