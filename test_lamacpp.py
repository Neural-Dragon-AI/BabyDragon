from llama_cpp import Llama
llm = Llama(model_path="/Users/tommasofurlanello/Documents/Dev/models/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF/dolphin-2.5-mixtral-8x7b.Q8_0.gguf")
output = llm(
      "<|im_start|>user: \n What is polars library in python?<|im_start|>user \n<|im_start|>assistant  ", # Prompt
      max_tokens=500, # Generate up to 32 tokens # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

