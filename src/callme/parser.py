
import json
import argparse
from .schemas import Prompt, FunctionDefinition
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic import TypeAdapter

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"

DEFAULT_FUNCTION_DEFINITIONS = INPUT_DIR / "functions_definition.json"
DEFAULT_FUNCTION_CALLING_TESTS = INPUT_DIR / "function_calling_tests.json"
DEFAULT_FUNCTION_CALLS_OUTPUT = OUTPUT_DIR / "function_calls.json"

def load_test_prompts(file_path: str) -> list[Prompt]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Test prompts file is not valid JSON: {e}")
    except Exception as e:
        raise Exception(f"Error loading test prompts: {e}")

    return TypeAdapter(list[Prompt]).validate_python(data)

def load_functions_definition(file_path: str) -> list[FunctionDefinition]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Function definitions file is not valid JSON: {e}")
    except Exception as e:
        raise Exception(f"Error loading function definitions: {e}")

    return TypeAdapter(list[FunctionDefinition]).validate_python(data)

def parse():
    parser = argparse.ArgumentParser(description="Run function calling tests.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_FUNCTION_CALLING_TESTS), help="function calling test inputs (json).")
    parser.add_argument("--output", type=str, default=str(DEFAULT_FUNCTION_CALLS_OUTPUT), help="function calls output (json).")
    parser.add_argument("--functions_definition", type=str, default=str(DEFAULT_FUNCTION_DEFINITIONS), help="functions definition (json).")
    args = parser.parse_args()
    
    return args
