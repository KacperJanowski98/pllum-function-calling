"""""
This script creates an Ollama Modelfile for a GGUF model.

Usage:
    python create_modelfile_fixed.py --model-path models/pllum-function-calling.gguf --output-dir models/ollama --model-name pllum-fc
"""

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_modelfile(model_path, output_dir, model_name, system_prompt=None, template=None):
    """
    Create an Ollama Modelfile.
    
    Args:
        model_path: Path to the GGUF model file
        output_dir: Directory to save the Modelfile
        model_name: Name of the Ollama model
        system_prompt: Custom system prompt (optional)
        template: Custom template (optional)
    """
    # Make sure model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} does not exist")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model filename (without path)
    model_filename = os.path.basename(model_path)
    
    # Create Modelfile path
    modelfile_path = os.path.join(output_dir, "Modelfile")
    
    # Use custom system prompt if provided
    if not system_prompt:
        system_prompt = "Jesteś modelem językowym PLLuM, wyspecjalizowanym w przetwarzaniu języka polskiego oraz innych języków słowiańskich i bałtyckich. Twoje umiejętności obejmują generowanie spójnych tekstów, odpowiadanie na pytania, podsumowywanie treści oraz wspieranie aplikacji specjalistycznych, takich jak inteligentni asystenci. Zostałeś wytrenowany na wysokiej jakości korpusach tekstowych i dostosowany do precyzyjnego dopasowania odpowiedzi, uwzględniając specyfikę polskiego języka i kultury. Jeśli nie posiadasz pełnych informacji lub pytanie jest niejasne, zawsze poproś użytkownika o doprecyzowanie."
    
    # Template for chat format (default or custom)
    if not template:
        template = '''{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}
# Narzędzia
Możesz wywołać jedną lub więcej funkcji, aby pomóc z zapytaniem użytkownika.
Dostępne narzędzia:
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}

Gdy chcesz wywołać funkcję, odpowiedz używając formatu JSON:
[{"name": "nazwa_funkcji", "arguments": {"parametr1": "wartość1", "parametr2": "wartość2"}}]
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
Wynik funkcji:
{{ .Content }}<|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}'''
    
    # Create Modelfile content
    modelfile_content = f'''FROM {model_filename}
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

# System message
SYSTEM "{system_prompt}"

# Template for chat format
TEMPLATE "{template}"
'''
    
    # Write Modelfile
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    logger.info(f"Modelfile created at {modelfile_path}")
    
    # Create a helper script to install the model
    install_script_path = os.path.join(output_dir, "install_model.sh")
    install_script_content = f'''#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first:"
    echo "curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Copy the model file to the current directory if needed
if [[ "{model_path}" != "$SCRIPT_DIR/{model_filename}" ]]; then
    echo "Copying model file to $SCRIPT_DIR..."
    cp "{model_path}" "$SCRIPT_DIR/{model_filename}"
fi

# Create the model in Ollama
echo "Creating model {model_name} in Ollama..."
cd "$SCRIPT_DIR"
ollama create {model_name} -f ./Modelfile

echo "Model {model_name} has been created in Ollama!"
echo "You can now run it with: ollama run {model_name}"
'''
    
    # Write install script and make it executable
    with open(install_script_path, "w", encoding="utf-8") as f:
        f.write(install_script_content)
    
    os.chmod(install_script_path, 0o755)
    logger.info(f"Install script created at {install_script_path}")
    
    # Create a Python test script
    test_script_path = os.path.join(output_dir, "test_model.py")
    test_script_content = '''#!/usr/bin/env python3
import requests
import json
import argparse

def query_ollama(model_name, prompt, temperature=0.1, tools=None):
    # Query the Ollama model
    # Format the prompt with tools if provided
    formatted_prompt = prompt
    if tools:
        formatted_prompt = f"""
Poniżej znajduje się zapytanie i lista dostępnych narzędzi.
Proszę wywołać odpowiednie narzędzie, aby odpowiedzieć na zapytanie użytkownika.

Zapytanie: {prompt}

Dostępne narzędzia:
{json.dumps(tools, indent=2, ensure_ascii=False)}
"""

    # Call Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": formatted_prompt,
            "temperature": temperature,
            "top_p": 0.9
        }
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    # Parse the response
    result = response.json()
    response_text = result.get("response", "")
    
    # Try to parse as JSON if it looks like JSON
    if response_text.strip().startswith("[") and response_text.strip().endswith("]"):
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
    
    return response_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ollama model")
    parser.add_argument("--model", default="pllum-fc", help="Ollama model name")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the model")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature (0.0-1.0)")
    parser.add_argument("--function-call", action="store_true", help="Format as function calling request")
    
    args = parser.parse_args()
    
    if args.function_call:
        # Example weather tool
        tools = [
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
        result = query_ollama(args.model, args.prompt, args.temperature, tools)
    else:
        result = query_ollama(args.model, args.prompt, args.temperature)
    
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)
'''
    
    # Write test script and make it executable
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(test_script_content)
    
    os.chmod(test_script_path, 0o755)
    logger.info(f"Test script created at {test_script_path}")
    
    return {
        "modelfile_path": modelfile_path,
        "install_script_path": install_script_path,
        "test_script_path": test_script_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Ollama Modelfile")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--output-dir", required=True, help="Directory to save Modelfile")
    parser.add_argument("--model-name", required=True, help="Name of Ollama model")
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument("--template-file", help="Path to a file containing a custom template")
    
    args = parser.parse_args()
    
    # Load template from file if provided
    template = None
    if args.template_file and os.path.exists(args.template_file):
        with open(args.template_file, 'r', encoding='utf-8') as f:
            template = f.read()
    
    create_modelfile(args.model_path, args.output_dir, args.model_name, args.system_prompt, template)
