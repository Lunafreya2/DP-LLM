import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          BertTokenizer, get_linear_schedule_with_warmup)

############################ MODIFIED FROM ###################################
#### https://www.kaggle.com/code/praveengovi/classify-
#### emotions-in-text-with-bert/notebook
#############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Running torch on device: {device}")

SEED = 19

## Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if device == torch.device("cuda:0"):
    torch.cuda.manual_seed_all(SEED)


SYN_DATA_DIR = "./data/synthetic"
ORG_DATA_DIR = "./data/org"

df_train_syn = pd.read_csv(f"{SYN_DATA_DIR}/train_16k.csv", delimiter=",")
df_train_org = pd.read_csv(f"{ORG_DATA_DIR}/train_16k.csv", delimiter=",")

df_test = pd.read_csv(f"{ORG_DATA_DIR}/test.csv", delimiter=",")

# print(df_test)
# print(df_train_org)
# print(df_train_syn)

MAX_LEN = 256

## Use synthetic data only for now

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
input_ids = [
    tokenizer.encode(
        t, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True
    )
    for t in df_train_syn["text"]
]
labels = df_train_syn["label"].values

# print(f"Actual text before tokenization: {df_train_syn['text'][2]}")
# print(f"Encoded input: {input_ids[2]}")

attention_masks = []
attention_marks = [[float(i > 0) for i in seq] for seq in input_ids]

# print(attention_marks[2])

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=41, test_size=0.1
)
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, input_ids, random_state=41, test_size=0.1
)

# convert all our data into torch tensors, required data type for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=batch_size
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=6
).to(device)

## HParameters
lr = 2e-5
adam_epsilon = 1e-8

epocs = 3

num_warmup_steps = 0
num_training_steps = len(train_dataloader) * epocs

### In Transformers, optimizer and schedules are split and instantiated like this:
optimizer = AdamW(
    model.parameters(), lr=lr, eps=adam_epsilon, correct_bias=False
)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)  # PyTorch scheduler

## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []

# Gradients gets accumulated by default
model.zero_grad()

# tnrange is a tqdm wrapper around the normal python range
for _ in tqdm.tnrange(1, epocs + 1, desc="Epoch"):
    print("<" + "=" * 22 + f" Epoch {_} " + "=" * 22 + ">")
    # Calculate total loss for this epoch
    batch_loss = 0

    for step, batch in enumerate(train_dataloader):
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = outputs[0]

        # Backward pass
        loss.backward()

        # Clip the norm of the gradients to 1.0
        # Gradient clipping is not in AdamW anymore
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update learning rate schedule
        scheduler.step()

        # Clear the previous accumulated gradients
        optimizer.zero_grad()

        # Update tracking variables
        batch_loss += loss.item()

    # Calculate the average loss over the training data.
    avg_train_loss = batch_loss / len(train_dataloader)

    # store the current learning rate
    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ", param_group["lr"])
        learning_rate.append(param_group["lr"])

    train_loss_set.append(avg_train_loss)
    print(f"\n\tAverage Training loss: {avg_train_loss}")

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        # Move logits and labels to CPU
        logits = logits[0].to("cpu").numpy()
        label_ids = b_labels.to("cpu").numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        df_metrics = pd.DataFrame(
            {"Epoch": epocs, "Actual_class": labels_flat, "Predicted_class": pred_flat}
        )

        tmp_eval_accuracy = accuracy_score(labels_flat, pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1

    print(f"\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}")
    print(f"\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}")
