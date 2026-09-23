
import json
from .schemas import Prompt
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic import TypeAdapter

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "input"

function_definitions_file = DATA_DIR / "functions_definition.json"
function_calling_tests_file = DATA_DIR / "function_calling_tests.json"

def load_test_prompts() -> list[Prompt]:
    try:
        with open(function_calling_tests_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test prompts file not found: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Test prompts file is not valid JSON: {e}")
    except Exception as e:
        raise Exception(f"Error loading test prompts: {e}")

    return TypeAdapter(list[Prompt]).validate_python(data)


def main():
    test_prompts = load_test_prompts()
    print(test_prompts)
    pass

if __name__ == "__main__":
    main()