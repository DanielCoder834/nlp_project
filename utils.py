# Rename this file to neurallm_utils.py!

# for word tokenization
import nltk
import csv
import sys
import torch
import torch.nn as nn
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('punkt_tab')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"


# PROVIDED
def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  space_char: str = ' ',
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
    Tokenize a single string. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
      line (str): text to tokenize
      ngram (int): ngram preparation number
      by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
      space_char (str): if by_char is True, use this character to separate to replace spaces
      sentence_begin (str): sentence begin token value
      sentence_end (str): sentence end token value

    Returns:
      list of strings - a single line tokenized
    """
    inner_pieces = None
    if by_char:
        line = line.replace(' ', space_char)
        inner_pieces = list(line)
    else:
        # otherwise use nltk's word tokenizer
        inner_pieces = nltk.word_tokenize(line)

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + \
            inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens

# PROVIDED
def read_file_spooky(datapath: str, ngram: int, by_character: bool = False, max_rows=10000) -> list:
    '''Reads and Returns the "data" as list of list (as shown above)'''
    x_data = []
    y_data = []
    csv.field_size_limit(sys.maxsize)
    with open(datapath, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break  # Stop reading if max_rows is reached
            # THIS IS WHERE WE GET CHARACTERS INSTEAD OF WORDS
            # replace spaces with underscores
            x_data.append(tokenize_line(
                row['article'].lower(), ngram, by_char=by_character, space_char="_"))
            y_data.append(tokenize_line(
                row['abstract'].lower(), ngram, by_char=by_character, space_char="_"))
    return x_data, y_data

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