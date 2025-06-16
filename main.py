import pandas as pd
import utils
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
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

    return predicted, truth, rouges

def predict_with_model_first(model_name="facebook/bart-base", data=None):
    # Does one instance of inference based on the inputted text
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + data, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    return formatted_summary
def evaluate(pred, truth):
    # Calculates the ROUGE metrics for a singular prediction
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(pred, truth)
    return scores

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
        