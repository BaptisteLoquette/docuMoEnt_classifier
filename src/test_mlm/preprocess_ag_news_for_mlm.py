import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader

parser  =   argparse.ArgumentParser(
    prog="preprocess_ag_news_for_mlm",
    description="Preprocess ag_news for MLM Multi-Class prediction"
    )
parser.add_argument('--save_path', help="Path to save the preprocessed dataset", type=str, default="..data/ag_news_for_mlm")
parser.add_argument('--prefix', help="The prefix sentence to append to the original texts (with the [MASK])", type=str, default="[MASK] News : ")

def preprocess_ag_news_for_mlm(sample, prefix="[MASK] News : "):
    sample['text']  =    prefix + sample['text']

    return sample

if __name__ ==  "__main__":
    args        =   parser.parse_args()
    prefix      =   args.prefix
    save_path   =   args.save_path

    ag_news =   load_dataset('ag_news')
    ag_news_for_mlm =   ag_news.map(preprocess_ag_news_for_mlm, fn_kwargs={"prefix" : prefix})
    ag_news_for_mlm.save_to_disk(save_path)