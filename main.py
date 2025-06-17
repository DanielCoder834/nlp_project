from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import csv
import textwrap
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import pandas as pd
import utils
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
import math
import numpy as np
import sys
EMBEDDINGS_SIZE = 50
NUM_SEQUENCES_PER_BATCH = 128

from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap
import csv
from rouge_score import rouge_scorer


def predict_with_model(model_name="facebook/bart-base", dataloader=None):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print(model.config)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(model_name)
    # inputs = tokenizer("summarize: " + data, return_tensors="pt", max_length=1024, truncation=True).input_ids
    # sumary_ids = generated_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    predicted = []
    truth = []
    rouges = []
    for index, batch in enumerate(dataloader):
        if index > 1:
            break
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # sample = "summarize: This is a simple article about the benefits of exercise for cardiovascular health."
        # inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=1024)
        # inputs = {k: v for k, v in inputs.items()}
        # output = model.generate(**inputs, max_length=150, num_beams=4)
        # print('SAMPLE', tokenizer.decode(output[0], skip_special_tokens=True))
        # break
        with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=150, 
                    min_length=50, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print('PREDICTED', decoded_preds)
        predicted.append(decoded_preds)
        labels = labels.cpu().numpy()
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        truth.append(decoded_labels)
        print("TRUTH", decoded_labels)
        rouge_score = scorer.score(decoded_preds[1], decoded_labels[0])
        rouges.append(rouge_score)
    #summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)


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
    dataset = TensorDataset(
        X["input_ids"], X["attention_mask"], y["input_ids"])
    test_dataset, train_dataset = torch.utils.data.random_split(
        dataset, [test_pct, 1 - test_pct])
    test_loader, train_loader = DataLoader(test_dataset, batch_size=num_sequences_per_batch), DataLoader(
        train_dataset, batch_size=num_sequences_per_batch)
    return test_loader, train_loader




def predict_with_model_first(model_name="facebook/bart-base", data=None):
    # Does one instance of inference based on the inputted text
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode(
        "summarize: " + data, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary
def evaluate(pred, truth):
    # Calculates the ROUGE metrics for a singular prediction
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(pred, truth)
    return scores



def get_tdidif_scores(articles, max_rows=100):
    # stopWords = set(stopwords.words("english"))
    # freq = Counter(data)
    page_count = len(articles)
    # full_text_article = " ".join(
    #     articles[:(page_count / 2)]) + " ".join(abstracts)
    full_text_article = " ".join(articles)
    token_sent_article = sent_tokenize(full_text_article)
    stop_words = set(stopwords.words('english'))
    print("--- Finished the set up ---")
    # Set up Freq
    #######
    word_freq = Counter()
    sentence_word_freq = []
    for sentence in token_sent_article:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word not in stop_words]

        freq = Counter(words)
        sentence_word_freq.append(freq)
        word_freq += freq
    print("--- Finished Freq ---")
    # TF
    #####
    tf_matrix = []
    for freq in sentence_word_freq:
        sent_length = sum(freq.values())
        tf_sentence = {}
        for word, count in freq.items():
            tf_sentence[word] = count / sent_length
        tf_matrix.append(tf_sentence)
    print("--- TF ---")
    # IDF
    ######
    doc_per_word = {}
    for word in word_freq:
        doc_per_word[word] = 0
        for freq in sentence_word_freq:
            if word in freq:
                doc_per_word[word] += 1
    num_sentences = len(token_sent_article)
    idf_matrix = {}
    for word, count in doc_per_word.items():
        idf_value = math.log(num_sentences / (1 + count))
        idf_matrix[word] = idf_value
    tfidf_matrix = []
    for tf_sentence in tf_matrix:
        tfidf_sentence = {}
        for word, tf_value in tf_sentence.items():
            tfidf_value = tf_value * idf_matrix[word]
            tfidf_sentence[word] = tfidf_value
        tfidf_matrix.append(tfidf_sentence)

    print("--- IDF ---")
    #######
    sent_scores = []
    for tfidf in tfidf_matrix:
        if len(tfidf) == 0:
            sent_scores.append(0)
            continue
        sentence_score = sum(tfidf.values()) / len(tfidf)
        sent_scores.append(sentence_score)
    print("--- Scores ---")
    return sent_scores


def predict_with_basic(article, max_rows=100, summary_length=128):
    abstracts = []
    articles = []
    csv.field_size_limit(sys.maxsize)
    with open('test.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            articles.append(row['article'].lower())
            abstracts.append(row['abstract'].lower())
    # Set up
    #########
    sent_scores = get_tdidif_scores(articles, max_rows)
    ######
    threshold_factor = 1
    threshold = np.mean(sent_scores) * threshold_factor

    summary_sentence = [article[i]
                        for i in range(summary_length) if sent_scores[i] >= threshold]
    return " ".join(summary_sentence), " ".join(abstracts)


 def calculate_average_fmeasures(scores):
    # Calculates the average f1 scores for each of the ROUGE metrics
    rouge1 = []
    rouge2 = []
    rougel = []
    for score_dict in scores:
        rouge1.append(score_dict['rouge1'].fmeasure)
        rouge2.append(score_dict['rouge2'].fmeasure)
        rougel.append(score_dict['rougeL'].fmeasure)
    avg_rouge1_f = sum(rouge1) / len(rouge1)
    avg_rouge2_f = sum(rouge2) / len(rouge2)
    avg_rougeL_f = sum(rougel) / len(rougel)
    return avg_rouge1_f, avg_rouge2_f, avg_rougeL_f
  
if __name__ == '__main__':
    # Load the data
    train, test = utils.load_dataset('train.csv')
    fined_tuned_scores = []
    baseline_scores = []
    # Fine tune the model
    utils.finetune_model(epochs=12, train_loader=train)
    with open('test.csv', encoding='utf-8') as csvfile:
        # Do inference with both the baseline and fine tuned model and calculate the ROUGE
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            print(index)
            data = row['article'].lower()
            truth = row['abstract'].lower()
            
            pred = predict_with_model_first('fine_tuned1', data=data)
            fined_tuned_scores.append(evaluate(pred, truth))

            pred = predict_with_model_first(data=data)
            baseline_scores.append(evaluate(pred, truth))
            if index > 250: 
                break
    # Compare the f1 averages of the baseline and fine tuned
    fine_tune_results = calculate_average_fmeasures(fined_tuned_scores)
    baseline_results = calculate_average_fmeasures(baseline_scores)
    print('finetuned', fine_tune_results)
    print('baseline', baseline_results)
    with open("results.txt", "w") as f:
        f.write('baseline:' + str(baseline_results) + '\n')
        f.write('finetuned:'+ str(fine_tune_results) + '\n')
