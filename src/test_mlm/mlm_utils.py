import torch
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer

tokenizer   =   BertTokenizer.from_pretrained('bert-base-uncased')
token2label =   {"World" : tokenizer.convert_tokens_to_ids('world'),
                "Sports" : tokenizer.convert_tokens_to_ids('sport'),
                "Business" : tokenizer.convert_tokens_to_ids('business'),
                "Sci/Tech" : tokenizer.convert_tokens_to_ids('science')}

def eval_classification_with_bert_mlm(test_dloader:DataLoader, device):
    
    bert_mlm    =   BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    bert_mlm.eval()
    acc         =   0
    for sample in tqdm(test_dloader):
        tokenized   =   tokenizer.batch_encode_plus(sample['text'], max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
        mask_idx    =   (tokenized['input_ids'][0] == 103).nonzero().item() # Find where the mask token is
        preds       =   bert_mlm(**tokenized).logits[:, mask_idx]   # Select the predictions for the mask_idx only 
        probs       =   torch.nn.functional.softmax(preds, dim=-1)  # Create a probability distribution over the vocabulary
        probs       =   probs[:, list(token2label.values())].cpu().detach() # Select only the probabilities for the tokens of interests
        acc         +=  compute_metrics(probs, label=sample['label'])['accuracy']   # Compute accuracy for the batch

    return  acc/len(test_dloader)

def compute_metrics(logits, label):
    """Computes the accuracy of the batches of the validation dataset"""
    metric      =   evaluate.load("accuracy")
    predictions =   np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=label)