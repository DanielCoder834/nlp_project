# Rename this file to neurallm_utils.py!

# for word tokenization
import nltk
import csv
import sys
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd

LEARNING_RATE = 2e-5
EPSILON = 1e-8
EPOCHS=5
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

nltk.download('punkt')
nltk.download('punkt_tab')

MAX_LENGTH = 25

# PROVIDED


def load_dataset(datapath, test_pct=.9):
    df = pd.read_csv(datapath, nrows=500)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train, test = torch.utils.data.random_split(
        dataset, [test_pct, 1 - test_pct])
    train_dataloader = DataLoader(train, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=4, shuffle=True)
    return train_dataloader, test_dataloader

def preprocess_data(data):
    articles = [str(a) for a in data['article']]
    abstracts = [str(s) for s in data['abstract']]
    print(len(list(data['article'])))
    model_inputs = tokenizer(articles, max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(abstracts, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def finetune_model(epochs=EPOCHS, train_loader=None, lr=LEARNING_RATE, eps=EPSILON):
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=eps,  weight_decay=0.01)
    for epoch_i in range(epochs):
        print("Epoch:", epoch_i + 1, "/", epochs)

        # Reset the total loss for this epoch.
        total_train_loss = 0
        loss_count = 0

        # Put the model into training mode
        model.train()

        # For each batch of training data...
        # use tqdm to make visualizing progress easier
        # for step, batch in enumerate(train_loader):
        num_batches = len(train_loader)
        print(num_batches)
        # also tell it how many batches there will be
        for step, batch in enumerate(train_loader):
            print(f'batch: {step}')
            # recall `batch` contains three pytorch tensors: input ids, attention masks and labels
            input_ids = batch['input_ids'] #.to(device)
            input_mask = batch['attention_mask'] #.to(device)
            labels = batch['labels'] #.to(device)

            optimizer.zero_grad()
            result = model(input_ids,
                        attention_mask=input_mask,
                        labels=labels,
                        return_dict=True)
            
            total_train_loss += result.loss.item()
            loss_count += 1
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f'current loss {result.loss.item()}')
    # TODO: Calculate the average loss over all of the batches for this epoch
    # report that loss
    model.save_pretrained("fine_tuned1")
    tokenizer.save_pretrained("fine_tuned1")
    print('average loss', total_train_loss/loss_count)

