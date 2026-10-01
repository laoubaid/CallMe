# pyrefly: ignore [missing-import]
from time import time
import json
# pyrefly: ignore [missing-import]
import numpy as np
from callme.schemas import FunctionCall
from callme.schemas import FunctionDefinition
from llm_sdk import Small_LLM_Model

class Mymodel():
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.model: Small_LLM_Model | None = None
        self.vocab: dict[str, int] | None = None

    def load_model(self) -> None:
        try:
            print(f"Loading model {self.model_name}...")
            self.model = Small_LLM_Model(model_name=self.model_name)
            vocab_path = self.model.get_path_to_vocab_file()
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_function_token_paths(self, fn_defs: list[FunctionDefinition]) -> list[list[int]]:
        """
        Encode each function name into its full sequence of token IDs.
        Returns: list of int arrays, e.g. [[86, 120, 4500], [85, 304, 8910]]
        """
        paths = []
        for fn_def in fn_defs:
            # add_special_tokens=False ensures you don't get <s> or [CLS] prepended
            tokens = self.model.encode(fn_def.name)
            # Handle tensor or list output from your wrapper
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()[0]
            paths.append(tokens)
        return paths

    def _get_parameter_token_paths(self, chosen_fn: FunctionDefinition) -> list[list[int]]:
        paths = []
        for key in chosen_fn.parameters.keys():
            tokens = self.model.encode(f"\"{key}\" : ")
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()[0]
            paths.append(tokens)
        return paths

    def _build_prompt(self, prompt: str, fn_defs: list[FunctionDefinition]) -> str:
        """
        Build the prompt for the model.
        """
        fn_defs_str = json.dumps([fn.model_dump() for fn in fn_defs], indent=4)
        return (
            f"You are a helpful assistant with access to these functions:\n"
            f"{fn_defs_str}\n"
            f"Given the user Prompt, return a JSON object followings this format:\n"
            f"{{\"name\": <string>, \"parameters\": <dict>}}\n"
            f"User request: {prompt}\n"
            f"JSON:\n"
        )

    def _autoregressive_generate(self, raw_logits: list[float], input_ids: list[int]) -> int:
        #   1. Turn raw logits into probabilities (Softmax)
        logits_arr = np.array(raw_logits)
        exp_logits = np.exp(logits_arr - np.max(logits_arr))
        probs = exp_logits / np.sum(exp_logits)
        #   2. Pick which token to select:
        next_token_id = int(np.argmax(probs))
        input_ids.append(next_token_id)
        decoded = self.model.decode([next_token_id])
        print(f"{decoded}", end="", flush=True)

        return next_token_id

    def _mask_logits(self, max_new_tokens: int, paths: list[list[int]], input_ids: list[int]) -> list[any]:
        generated_tokens = []

        for _ in range(max_new_tokens):
            raw_logits = self.model.get_logits_from_input_ids(input_ids)

            step = len(generated_tokens)
            valid_token_ids = []
            for path in paths:
                if step < len(path) and path[:step] == generated_tokens:
                    valid_token_ids.append(path[step])
            if not valid_token_ids:
                break

            mask = np.full_like(raw_logits, -np.inf)
            for t in valid_token_ids:
                mask[t] = 0.0
            
            filtered_logits = raw_logits + mask
            next_token_id = self._autoregressive_generate(filtered_logits, input_ids)
            generated_tokens.append(next_token_id)
        return generated_tokens

    def generate(self, prompt: str, fn_defs: list[FunctionDefinition]) -> None:
        if self.model is None:
            raise Exception("Model not loaded")
        
        input_ids = self.model.encode(self._build_prompt(prompt, fn_defs)).tolist()[0]
        initial_length = len(input_ids)
        input_ids.extend(self.model.encode("{\"name\": \"").tolist()[0])
        print("{\"name\": \"", end="", flush=True)

        max_new_tokens = 32
        paths = self._get_function_token_paths(fn_defs)
        generated_fn_tokens = self._mask_logits(max_new_tokens, paths, input_ids)

        selected_function = self.model.decode(generated_fn_tokens)
        chosen_fn = next(fn for fn in fn_defs if fn.name == selected_function)
        input_ids.extend(self.model.encode("\", \"parameters\": { ").tolist()[0])
        print("\", \"parameters\": { ", end="", flush=True)
        paths = self._get_parameter_token_paths(chosen_fn)

        for _ in paths:
            generated_param_tokens = self._mask_logits(max_new_tokens, paths, input_ids)

            count = 0
            for _ in range(max_new_tokens):
                raw_logits = self.model.get_logits_from_input_ids(input_ids)
                next_token_id = self._autoregressive_generate(raw_logits, input_ids)
                decoded = self.model.decode([next_token_id])
                if "," in decoded:
                    break
                count += decoded.count("}")
                if count >= 2:
                    break 

        # generated_tokens = input_ids[initial_length:]
        # generated_text = self.model.decode(generated_tokens)
        # print(f"Generated text: \n{generated_text}")
        print("\n========================================================================\n")



        # =========================================================================
        # STEP 4: Parse generated text into FunctionCall schema
        # =========================================================================
        # - The output text should contain a JSON string, e.g. {"name": "...", "parameters": {...}}
        # - Parse it using json.loads() (or regex/string cleaning if extra text was generated).
        # - Validate and return as:
        #   FunctionCall(prompt=prompt, name=data["name"], parameters=data["parameters"])

        