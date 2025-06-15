# Rename this file to neurallm_utils.py!

# for word tokenization
import nltk
import csv
import sys
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from transformers import BertTokenizer
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd

LEARNING_RATE = 2e-5
EPSILON = 1e-8
EPOCHS=5
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

nltk.download('punkt')
nltk.download('punkt_tab')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
MAX_LENGTH = 25

# PROVIDED


def read_file_spooky(datapath: str, ngram: int, by_character: bool = False, max_rows=10000,
                     max_length=MAX_LENGTH, padding="max_length", truncation=True) -> list:
    '''Reads and Returns the "data" as list of list (as shown above)'''
    x_data = []
    y_data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    csv.field_size_limit(sys.maxsize)
    with open(datapath, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break  # Stop reading if max_rows is reached
            # THIS IS WHERE WE GET CHARACTERS INSTEAD OF WORDS
            # replace spaces with underscores
            x_data.append(row['article'].lower())
            y_data.append(row['abstract'].lower())

            # encoded_train = tokenizer(
            #     sequences, padding="max_length", truncation=True, return_tensors="tf").data
            # x_data.append(tokenize_line(
            #     row['article'].lower(), ngram, by_char=by_character, space_char="_"))
            # y_data.append(tokenize_line(
            #     row['abstract'].lower(), ngram, by_char=by_character, space_char="_"))
    return tokenizer(x_data, return_tensors='pt',
                     max_length=max_length, padding=padding, truncation=truncation), tokenizer(y_data, return_tensors='pt',
                                                                                               max_length=max_length, padding=padding, truncation=truncation)


def save_word2vec(embeddings: Word2Vec, filename: str) -> None:
    """
    Saves weights of trained gensim Word2Vec model to a file.

    Params:
        obj: The object.
        filename: The destination file.
    """
    print('Saving Word2Vec')
    embeddings.save(filename)


def train_word2vec(data: list[list[str]], embeddings_size: int,
                   window: int = 5, min_count: int = 1, sg: int = 1) -> Word2Vec:
    """
    Create new word embeddings based on our data.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A gensim Word2Vec model
        https://radimrehurek.com/gensim/models/word2vec.html
    """
    print('Creating Word2Vec')
    return Word2Vec(sentences=data,
                    vector_size=embeddings_size,
                    window=window,
                    min_count=min_count,
                    sg=sg)


def load_word2vec(filename: str) -> Word2Vec:
    """
    Loads weights of trained gensim Word2Vec model from a file.

    Params:
        filename: The saved model file.
    """
    return Word2Vec.load(filename)


def load_dataset(datapath, test_pct=.9):
    df = pd.read_csv(datapath, nrows=250)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train, test = torch.utils.data.random_split(
        dataset, [test_pct, 1 - test_pct])
    train_dataloader = DataLoader(train, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=2, shuffle=True)
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
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
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

