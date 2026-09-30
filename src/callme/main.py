
from callme.model import Mymodel
from callme.parser import load_functions_definition, load_test_prompts, parse

def main():
    args = parse()
    function_calls = []
    test_prompts = load_test_prompts(args.input)
    function_definitions = load_functions_definition(args.functions_definition)

    model = Mymodel()
    model.load_model()
    for test_prompt in test_prompts:
        print("Processing prompt: ", test_prompt.prompt)
        model.generate(test_prompt.prompt, function_definitions)
    # try:
            
        
    # except Exception as e:
    #     print(e)
    #     return 1
    return 0

if __name__ == "__main__":
    main()