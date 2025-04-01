"""
A Flask server for handling function calling with Ollama.

This server provides a REST API for sending queries to an Ollama model
and getting function calls in response.

Usage:
  1. Start the server: python function_calling_server.py
  2. Send requests to: http://localhost:5000/function_call
  
Example:
  curl -X POST http://localhost:5000/function_call \
    -H "Content-Type: application/json" \
    -d '{"query":"What is the weather in Warsaw?","tools":[{"name":"get_weather","description":"Get weather information","parameters":{"location":{"type":"string","required":true}}}]}'
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Union
from flask import Flask, request, jsonify, Response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = Flask(__name__)

# Configuration
OLLAMA_API_HOST = os.environ.get("OLLAMA_API_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "pllum-fc")
DEFAULT_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))

def query_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False
) -> Union[Dict[str, Any], Response]:
    """
    Query the Ollama API.
    
    Args:
        prompt: Prompt to send to the model
        model: Ollama model name
        temperature: Sampling temperature
        stream: Whether to stream the response
        
    Returns:
        API response or Flask streaming response
    """
    logger.info(f"Querying Ollama model: {model}")
    
    api_url = f"{OLLAMA_API_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": stream
    }
    
    try:
        if stream:
            def generate():
                with requests.post(api_url, json=payload, stream=True) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama API error: {response.status_code}")
                        yield json.dumps({"error": f"Ollama API error: {response.status_code}"})
                        return
                    
                    for line in response.iter_lines():
                        if line:
                            yield line.decode('utf-8') + '\n'
            
            return Response(generate(), content_type='application/json')
        else:
            response = requests.post(api_url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return {"error": f"Ollama API error: {response.status_code}"}
            
            return response.json()
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": f"Request error: {str(e)}"}

def format_function_call_prompt(query: str, tools: List[Dict[str, Any]]) -> str:
    """
    Format a function calling prompt.
    
    Args:
        query: User query
        tools: List of available tools
        
    Returns:
        Formatted prompt
    """
    tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
    
    # Determine if query is Polish or English (very simple detection)
    polish_chars = set('ąćęłńóśźż')
    is_polish = any(char.lower() in polish_chars for char in query)
    
    if is_polish:
        return f"""Poniżej znajduje się zapytanie i lista dostępnych narzędzi.
Proszę wywołać odpowiednie narzędzie, aby odpowiedzieć na zapytanie użytkownika.

Zapytanie: {query}

Dostępne narzędzia:
{tools_str}"""
    else:
        return f"""Below is a query and a list of available tools.
Please call the appropriate tool to respond to the user's query.

Query: {query}

Available tools:
{tools_str}"""

@app.route('/function_call', methods=['POST'])
def function_call() -> Response:
    """Handle function calling requests."""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        query = data.get('query')
        tools = data.get('tools', [])
        model = data.get('model', DEFAULT_MODEL)
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        stream = data.get('stream', False)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Format the prompt for function calling
        prompt = format_function_call_prompt(query, tools)
        
        # Query Ollama
        result = query_ollama(prompt, model, temperature, stream)
        
        if stream:
            return result
        
        if "error" in result:
            return jsonify(result), 500
        
        response_text = result.get("response", "")
        
        # Parse the JSON response
        try:
            if response_text.strip().startswith("[") and response_text.strip().endswith("]"):
                function_call = json.loads(response_text)
                return jsonify(function_call)
            else:
                return jsonify({"raw_response": response_text})
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse response as JSON: {response_text}")
            return jsonify({"raw_response": response_text})
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """Check if the server is running and Ollama is available."""
    try:
        # Check if Ollama is available
        response = requests.get(f"{OLLAMA_API_HOST}/api/tags")
        
        if response.status_code == 200:
            return jsonify({
                "status": "ok",
                "ollama": "available",
                "models": response.json()
            })
        else:
            return jsonify({
                "status": "degraded",
                "ollama": "unavailable",
                "error": f"Ollama API returned status code {response.status_code}"
            }), 503
    
    except requests.RequestException as e:
        return jsonify({
            "status": "degraded",
            "ollama": "unavailable",
            "error": str(e)
        }), 503

@app.route('/chat', methods=['POST'])
def chat() -> Response:
    """Handle regular chat requests without function calling."""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        prompt = data.get('prompt')
        model = data.get('model', DEFAULT_MODEL)
        temperature = data.get('temperature', DEFAULT_TEMPERATURE)
        stream = data.get('stream', False)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Query Ollama
        result = query_ollama(prompt, model, temperature, stream)
        
        if stream:
            return result
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Function calling server for Ollama")
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--model", default="pllum-fc", help="Default Ollama model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Update configuration
    OLLAMA_API_HOST = args.ollama_host
    DEFAULT_MODEL = args.model
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Using Ollama API at {OLLAMA_API_HOST}")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
