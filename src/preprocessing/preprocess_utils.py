from typing import List
import re
import os
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict

def prepare_dataset_for_BC(sample, tokenizer, label):
    """Tokenize and prepare dataset for Binary Classification"""
    text        =   sample['text'].encode('ascii', 'ignore').decode('ascii') # Format to ASCII
    text        =   re.sub('http://\S+|https://\S+', '', text)  # Remove urls
    tokenized                   =   tokenizer.encode_plus(text, max_length=512, padding="max_length", truncation=True)
    sample['input_ids']         =   tokenized['input_ids']
    sample['attention_mask']    =   tokenized['attention_mask']
    sample['token_type_ids']    =   tokenized['token_type_ids']
    sample['label']             =   1 if sample['label'] == label else 0

    return sample

def preprocess_save_ag_news_for_BC(tokenizer, path_out, sampling_prop=1/3):
    """Form 4 datasets, from the 4 classes, for Binary classification (to train the experts)
        1. Sample a balanced subset of each category of Ag News to train the experts on
        2. Makes one dataset for each expert to be trained on 
    """
    sampled_dsets   =   []
    ag_news =   load_dataset('ag_news')
    labels  =   list(set(ag_news["train"]["label"]))

    for id_label in labels:
        filtered_dset       =   ag_news["train"].filter(lambda sample: sample['label'] == id_label)
        under_sampled_dset  =   filtered_dset.select(range(int(sampling_prop * len(filtered_dset))))
        sampled_dsets.append(under_sampled_dset)
    sampled_dsets   =   concatenate_datasets(sampled_dsets)

    for id_label in labels:
        expert_dset =   sampled_dsets.map(prepare_dataset_for_BC, fn_kwargs={"tokenizer" : tokenizer, "label" : id_label})
        expert_dset =   expert_dset.train_test_split(0.1).shuffle()
        ## Save dataset
        category    =   id2label_ag_news[id_label]
        save_path   =   os.path.join(path_out, f"ag_news_{re.sub('/', '_', category)}")
        expert_dset.save_to_disk(os.path.join(path_out, f"ag_news_{id2label_ag_news[id_label]}"))

def tokenize_func_ag_news(sample, tokenizer:BertTokenizer) -> dict:
    """Preprocess each sample's text Tokenizes a sample"""
    text        =   sample['text'].encode('ascii', 'ignore').decode('ascii') # Format to ASCII
    text        =   re.sub('http://\S+|https://\S+', '', text)  # Remove urls
    tokenized   =   tokenizer.encode_plus(text, max_length=512, padding="max_length", truncation=True)
    return tokenized

def load_preprocess_ag_news(tokenizer:BertTokenizer) -> DatasetDict:
    """Preprocess Ag News dataset.
        2. Create a test, val and train set
        3. Preprocess and Tokenizes each samples"""
    ag_news =   load_dataset('ag_news')
    ag_news['test'] =   ag_news['test'].train_test_split(0.2)   # Create validation dataset
    ## Reformat dataset to be nicer
    ag_news = DatasetDict({
    'train': ag_news['train'],
    'test': ag_news['test']['train'],
    'val': ag_news['test']['test']})

    dset    =   ag_news.map(tokenize_func_ag_news, fn_kwargs={"tokenizer" : tokenizer}, remove_columns=["text"])
    return dset

id2label_ag_news    =   {0 : "World",
                        1 :"Sports",
                        2 : "Business",
                        3 : "Sci/Tech"}