from typing import List
import re
from transformers import BertTokenizer
from datasets import load_dataset, DatasetDict

def tokenize_func_ag_news(sample, tokenizer:BertTokenizer) -> dict:
    """Preprocess each sample's text Tokenizes a sample"""
    text        =   sample['text'].encode('ascii', 'ignore').decode('ascii') # Format to ASCII
    text        =   re.sub('http://\S+|https://\S+', '', text)  # Remove urls
    tokenized   =   tokenizer.encode_plus(sample['text'], max_length=512, padding="max_length", truncation=True)
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

def tokenize_func_eli5(sample:dict, tokenizer:BertTokenizer, categories:List[str]) -> dict:
    """Tokenizes a sample and assigns a label given the category"""
    
    tokenized                   =   tokenizer.encode_plus(sample["answers"]['text'][0], max_length=512, padding="max_length", truncation=True)
    sample["input_ids"]         =   tokenized["input_ids"]
    sample["attention_mask"]    =   tokenized["attention_mask"]
    sample["token_type_ids"]    =   tokenized["token_type_ids"]
    sample["labels"]            =   categories.index(sample["category"])
    return sample


def load_preprocess_eli5(tokenizer:BertTokenizer, categories:List[str]):
    """Preprocess ELI5 dataset.
        1. Filters the specified categories
        2. Create a test and train set
        3. Tokenizes each samples and assigns a label to each sample given the category"""
    
    eli_5   =   load_dataset('eli5_category')

    eli_5           =   eli_5['train'].filter(lambda example: example['category'] in categories)
    eli_5           =   eli_5.train_test_split(0.2).shuffle()
    tokenized_eli_5 =   eli_5.map(
                    tokenize_func_eli5, 
                    fn_kwargs={"tokenizer" : tokenizer, "categories" : categories},
                    remove_columns=['q_id', 'title', 'selftext', 'category', 'subreddit', 'answers', 'title_urls', 'selftext_urls',])
    return tokenized_eli_5