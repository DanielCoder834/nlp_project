import pandas as pd
import utils
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
EMBEDDINGS_SIZE = 50
NUM_SEQUENCES_PER_BATCH = 128


def read_data(filepath='train.csv'):
    start_time = datetime.datetime.now()
    print(f'Started downloading at {start_time}')
    X, y = utils.read_file_spooky(filepath, 1, max_rows=1000)
    end_time = datetime.datetime.now()
    print(f'Downloading runtime {end_time-start_time}')
    return X, y


def create_dataloaders(X: list, y: list, num_sequences_per_batch: int,
                       test_pct: float = 0.1, shuffle: bool = True) -> tuple[torch.utils.data.DataLoader]:
    """
    Convert our data into a PyTorch DataLoader.    
    A DataLoader is an object that splits the dataset into batches for training.
    PyTorch docs: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        https://pytorch.org/docs/stable/data.html

    Note that you have to first convert your data into a PyTorch DataSet.
    You DO NOT have to implement this yourself, instead you should use a TensorDataset.

    You are in charge of splitting the data into train and test sets based on the given
    test_pct. There are several functions you can use to acheive this!

    The shuffle parameter refers to shuffling the data *in the loader* (look at the docs),
    not whether or not to shuffle the data before splitting it into train and test sets.
    (don't shuffle before splitting)

    Params:
        X: A list of input sequences
        Y: A list of labels
        num_sequences_per_batch: Batch size
        test_pct: The proportion of samples to use in the test set.
        shuffle: INSTRUCTORS ONLY

    Returns:
        One DataLoader for training, and one for testing.
    """
    # YOUR CODE HERE
    dataset = TensorDataset(torch.tensor(X["input_ids"]), torch.tensor(
        X["attention_mask"]), torch.tensor(y["input_ids"]))
    test_dataset, train_dataset = torch.utils.data.random_split(
        dataset, [test_pct, 1 - test_pct])
    test_loader, train_loader = DataLoader(test_dataset, batch_size=num_sequences_per_batch), DataLoader(
        train_dataset, batch_size=num_sequences_per_batch)
    return test_loader, train_loader


if __name__ == '__main__':
    X, y = read_data()
    # print("THE START: -->", X[0])
    # print(y)
    train_vec = utils.train_word2vec(X, EMBEDDINGS_SIZE)
    utils.save_word2vec(train_vec, "word_embeddings")
    embeddings = utils.load_word2vec("word_embeddings")
    test_loader, train_loader = create_dataloaders(
        X, y, NUM_SEQUENCES_PER_BATCH)
    print(len(embeddings.wv))
