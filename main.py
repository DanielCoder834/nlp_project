import pandas as pd
import utils
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
EMBEDDINGS_SIZE = 50
NUM_SEQUENCES_PER_BATCH = 128
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import textwrap
import csv
from rouge_score import rouge_scorer


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
    dataset = TensorDataset(X["input_ids"], X["attention_mask"], y["input_ids"])
    test_dataset, train_dataset = torch.utils.data.random_split(
        dataset, [test_pct, 1 - test_pct])
    test_loader, train_loader = DataLoader(test_dataset, batch_size=num_sequences_per_batch), DataLoader(
        train_dataset, batch_size=num_sequences_per_batch)
    return test_loader, train_loader

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

    return predicted, truth, rouges

def predict_with_model_first(model_name="facebook/bart-base", data=None):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + data, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary
def evaluate(pred, truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(pred, truth)
    return scores

def calculate_average_fmeasures(scores):
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
    train, test = utils.load_dataset('train.csv')
    print(len(train))
    print(len(test))
    fined_tuned_scores = []
    baseline_scores = []
    utils.finetune_model(epochs=7, train_loader=train)
    with open('test.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            print(index)
            data = row['article'].lower()
            truth = row['abstract'].lower()
            
            pred = predict_with_model_first('fine_tuned1', data=data)
            fined_tuned_scores.append(evaluate(pred, truth))

            pred = predict_with_model_first(data=data)
            baseline_scores.append(evaluate(pred, truth))
            if index > 10: 
                break
    print('finetuned', calculate_average_fmeasures(fined_tuned_scores))
    print('baseline', calculate_average_fmeasures(baseline_scores))