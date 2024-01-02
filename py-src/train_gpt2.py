import datasets
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_e = datasets.load_dataset("dair-ai/emotion", name="unsplit", split="train")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"sep_token": "[SEP]"})
tokenizer.add_special_tokens({"bos_token": "[BOS]"})

model.resize_token_embeddings(len(tokenizer))

LABELS = ["sadness", "joy", "love", "anger", "fear", "suprise"]


def tokenization(example):
    # print(f"I have {len(example['text'])} rows in example")
    # print(example)

    for idx, row in enumerate(example["text"]):
        example["text"][
            idx
        ] = f'{tokenizer.bos_token}{LABELS[example["label"][idx]]}{tokenizer.sep_token}{row}{tokenizer.eos_token}'

    out_dict = tokenizer(example["text"])

    out_dict["text"] = example["text"]

    return out_dict


dataset_e = dataset_e.map(tokenization, batched=True, remove_columns=["label"])
dataset_e.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataset_e = dataset_e.with_format("torch", device=device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_e,
)

trainer.train()
