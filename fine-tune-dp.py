# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train GPT2 model series with DP (w/ optional parameter-efficient approach LoRA)"""

import logging
import sys
from dataclasses import asdict, dataclass, field

import datasets
import dp_transformers
import pandas as pd
import transformers
from dp_transformers.grad_sample.transformers import conv_1d

logger = logging.getLogger(__name__)


LABELS = ["negative", "positive"]


@dataclass
class ModelArguments:
    model_name: str = field(
        default="gpt2", metadata={"help": "Model name in HuggingFace, e.g. 'gpt2'"}
    )
    sequence_len: int = field(default=128, metadata={"help": "Maximum sequence length"})


@dataclass
class LoraArguments:
    enable_lora: bool = field(
        default=False, metadata={"help": "Whether to enable LoRA"}
    )
    lora_dim: int = field(default=8, metadata={"help": "LoRA dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.GPT2LMHeadModel.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load tokenizer
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    tokenizer.add_special_tokens({"bos_token": "[BOS]"})

    model.resize_token_embeddings(len(tokenizer))

    # Load data (train_test used as train_val)

    dataset = datasets.load_dataset("sst2", split="train+validation").train_test_split(
        0.2, seed=args.train.seed
    )

    # dataset = pd.read_json('./data/balanced/unsplit_emotion_balanced.jsonl', lines=True)
    # dataset = datasets.Dataset.from_pandas(dataset).train_test_split(0.2, seed=args.train.seed)

    def insert_labels(example):
        example[
            "sentence"
        ] = f'{tokenizer.bos_token}{LABELS[example["label"]]}{tokenizer.sep_token}{example["sentence"]}{tokenizer.eos_token}'

        return example

    dataset = dataset.map(insert_labels)

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(
                batch["sentence"],
                padding="max_length",
                truncation=True,
                max_length=args.model.sequence_len,
            ),
            batched=True,
            num_proc=8,
            desc="tokenizing dataset",
            remove_columns=dataset.column_names["label"],
        )

    if train_args.local_rank == 0:
        logger.info(
            f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}"
        )
        logger.info(
            f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}"
        )

    model = model.cuda()
    model.train()

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(
        tokenizer
    )

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        privacy_args=privacy_args,
    )

    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({"final_epsilon_prv": eps_prv, "final_epsilon_rdp": eps_rdp})

    trainer.save_model("./trainer_savedmodel")
    model.save_pretrained("./pytorch_savedmodel")


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (
            dp_transformers.TrainingArguments,
            dp_transformers.PrivacyArguments,
            ModelArguments,
            LoraArguments,
        )
    )
    train_args, privacy_args, model_args, _ = arg_parser.parse_args_into_dataclasses()
    main(
        Arguments(
            train=train_args,
            privacy=privacy_args,
            model=model_args,
        )
    )
