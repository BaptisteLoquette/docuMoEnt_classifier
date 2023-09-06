"""Preprocess Ag News for Binary Classification as each expert will be trained to predict only if a document belongs to an only category"""
import argparse
from transformers import BertTokenizer
from preprocess_utils import preprocess_save_ag_news_for_BC

parser  =   argparse.ArgumentParser(
    prog="preprocess_ag_news_for_MoE",
    description="Preprocess AG News dataset for BERT MoE")
parser.add_argument('--out', help="Output dir of the preprocessed files", type=str, default="../data/ag_news_preprocessed_MoE")
parser.add_argument('--tokenizer_path', help="Path to tokenizer", type=str, default="bert-base-uncased")
parser.add_argument('--sampling_prop', help="Portion of the dataset to only train the experts on", type=float, default=1/3)

if __name__ == "__main__":
    args            =   parser.parse_args()
    save_path       =   args.out
    tokenizer_path  =   args.tokenizer_path
    sampling_prop   =   args.sampling_prop

    tokenizer   =   BertTokenizer.from_pretrained(tokenizer_path)   # Initialize tokenizer
    preprocess_save_ag_news_for_BC(tokenizer, save_path, sampling_prop=sampling_prop)