!pip install transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")

inputs = tokenizer("AIによって私達の暮らしは、", return_tensors="pt").to(model.device)
with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)