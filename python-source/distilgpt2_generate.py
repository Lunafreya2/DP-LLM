import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=100,
    top_p=0.8,
    temperature=1.0,
):
    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                # outputs = model(generated)
                # print(outputs.shape)
                loss, logits = outputs[:2]
                # print(logits)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)

                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)

    return generated_list


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    example_text = sys.argv[1]

    # ## Not sure what return_tensors="pt" does
    # encoded_input = tokenizer(example_text, return_tensors="pt")

    # ## ** is expanding an array/list structure to comma separated arguments
    # output = model(**encoded_input)

    output = generate(model=model, tokenizer=tokenizer, prompt=example_text)

    output_text = "\n\n --- NEW PROMPT --- \n\n".join(output)

    os.makedirs("./output", exist_ok=True)

    output_file_path = f"./output/distilgpt2_test_out.txt"
    print(f"Saving the outputs to {output_file_path}")

    with open(output_file_path, "w") as outfile:
        outfile.writelines(output_text)


if __name__ == "__main__":
    main()
