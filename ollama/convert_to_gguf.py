"""
This script converts a Hugging Face model to GGUF format for use with Ollama.

Dependencies:
  - git
  - cmake
  - make or build-essential
  - llama-cpp-python

Usage:
  python convert_to_gguf.py --input-dir models/pllum-function-calling-merged --output-file models/pllum-function-calling.gguf --quantization Q4_K_M
"""

import os
import sys
import argparse
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported quantization types
QUANT_TYPES = [
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",   # Original quant types
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",      # k-quants
    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",    # More k-quants
    "Q6_K", "Q8_K", "F16", "F32"               # Higher precision
]


def check_git():
    """Check if git is installed."""
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_cmake():
    """Check if cmake is installed."""
    try:
        subprocess.run(["cmake", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_make():
    """Check if make is installed."""
    try:
        subprocess.run(["make", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def clone_llamacpp(llama_cpp_dir):
    """Clone or update llama.cpp repository."""
    if os.path.exists(llama_cpp_dir):
        logger.info(f"Updating existing llama.cpp repository in {llama_cpp_dir}")
        subprocess.run(["git", "pull"], cwd=llama_cpp_dir, check=True)
    else:
        logger.info(f"Cloning llama.cpp repository to {llama_cpp_dir}")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
            check=True
        )

def build_llamacpp(llama_cpp_dir):
    """Build llama.cpp."""
    logger.info("Building llama.cpp")
    
    # Create build directory
    build_dir = os.path.join(llama_cpp_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    # Run cmake
    subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        check=True
    )
    
    # Run make
    subprocess.run(
        ["make", "-j"],
        cwd=build_dir,
        check=True
    )
    
    logger.info("llama.cpp build completed")

def convert_to_gguf(input_dir, output_file, llama_cpp_dir, quantization=None):
    """
    Convert a Hugging Face model to GGUF format.
    
    Args:
        input_dir: Path to the Hugging Face model
        output_file: Path to save the GGUF file
        llama_cpp_dir: Path to llama.cpp repository
        quantization: Quantization type (optional)
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for the converter script
    convert_script_path = os.path.join(llama_cpp_dir, "convert.py")
    if not os.path.exists(convert_script_path):
        logger.error(f"Converter script not found at {convert_script_path}")
        sys.exit(1)
    
    # Base command for running the converter
    cmd = [
        sys.executable,
        convert_script_path,
        input_dir,
        "--outfile", output_file,
    ]
    
    # Add quantization if specified
    if quantization:
        output_quant_file = output_file.replace(".gguf", f"-{quantization.lower()}.gguf")
        
        # First convert to regular GGUF
        logger.info("Converting model to GGUF...")
        subprocess.run(cmd, check=True)
        
        # Then quantize
        logger.info(f"Quantizing model to {quantization}...")
        quantize_cmd = [
            os.path.join(llama_cpp_dir, "build", "bin", "quantize"),
            output_file,
            output_quant_file,
            quantization.lower()
        ]
        subprocess.run(quantize_cmd, check=True)
        
        # Replace original output file with quantized version if successful
        if os.path.exists(output_quant_file):
            # If we successfully created the quantized file, use it as the result
            if os.path.exists(output_file):
                os.remove(output_file)  # Remove the unquantized version
            output_file = output_quant_file
    else:
        # Just convert to GGUF without quantization
        logger.info("Converting model to GGUF...")
        subprocess.run(cmd, check=True)
    
    logger.info(f"Conversion completed! Output saved to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GGUF format")
    parser.add_argument("--input-dir", required=True, help="Path to Hugging Face model")
    parser.add_argument("--output-file", required=True, help="Path to save GGUF file")
    parser.add_argument("--llama-cpp-dir", default="./llama.cpp", help="Path to llama.cpp repository")
    parser.add_argument(
        "--quantization", 
        choices=QUANT_TYPES,
        help=f"Quantization type (optional): {', '.join(QUANT_TYPES)}"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_deps = []
    if not check_git():
        missing_deps.append("git")
    if not check_cmake():
        missing_deps.append("cmake")
    if not check_make():
        missing_deps.append("make or build-essential")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install them before running this script")
        sys.exit(1)
    
    # Clone and build llama.cpp
    clone_llamacpp(args.llama_cpp_dir)
    build_llamacpp(args.llama_cpp_dir)
    
    # Convert model to GGUF
    convert_to_gguf(args.input_dir, args.output_file, args.llama_cpp_dir, args.quantization)
