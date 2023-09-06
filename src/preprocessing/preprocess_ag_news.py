import argparse
from transformers import BertTokenizer
from preprocess_utils import load_preprocess_ag_news

parser  =   argparse.ArgumentParser(
    prog="preprocess_ag_news",
    description="Preprocess AG News dataset for Multi-Label Classification")
parser.add_argument('--out', help="Output dir of the preprocessed files", type=str, default="../data/ag_news_preprocessed")
parser.add_argument('--tokenizer_path', help="Path to tokenizer", type=str, default="bert-base-uncased")

if __name__ == "__main__":
    args            =   parser.parse_args()
    save_path       =   args.out
    tokenizer_path  =   args.tokenizer_path

    tokenizer   =   BertTokenizer.from_pretrained(tokenizer_path)   # Initialize tokenizer

    agnews_preprocessed =   load_preprocess_ag_news(tokenizer)  # Load ag news, preprocess and tokenizes it

    agnews_preprocessed.save_to_disk(save_path)