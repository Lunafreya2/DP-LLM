import sys

from transformers import GPT2Model, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

model = GPT2Model.from_pretrained("distilgpt2")

example_text = sys.argv[1]

## Not sure what return_tensors="pt" does
encoded_input = tokenizer(example_text, return_tensors="pt")

## ** is expanding an array/list structure to comma separated arguments
output = model(**encoded_input)

print(output)
