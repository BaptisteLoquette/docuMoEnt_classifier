import re
import torch
import numpy as np
from nltk import sent_tokenize
from torch.nn.functional import softmax
from summarizer import Summarizer

summarize_model =   Summarizer()
def classify_document(document:str, tokenizer, model, id2label:dict, extractive_sum=False, full_document=False, max_length=512, device="cuda") -> str:
    """
    Classify any document to a predefined set of categories (here : "World", "Sport", "Business", "Sci/Tech")

    Inputs
        - document : The document to classify
        - tokenizer : the corresponding tokenizer of the model
        - model : the classifier
        - id2label : the mapping between the argmax and the category's name
        • if the document is longer than the model's max_length :
            - extractive_sum : apply extractive summarization (if True)
            - full_document : iteratively pass each sentence through the classifier then average each predictions to make the final pred
        - max_length : the model's max_length
        - device
    
    Outputs :
        - The predicted category's name
    """
    document    =   re.sub('http://\S+|https://\S+', '', document)      # Remove urls
    document    =   document.encode('ascii', 'ignore').decode('ascii')  # Format to ASCII

    sentences   =   sent_tokenize(document)
    sent_length =   [len(tokenizer.tokenize(sent)) for sent in sentences]   # Computes sentences length, needed to find an optimal number of sentences if summarization

    if not full_document:
        if sum(sent_length) > max_length and extractive_sum:
            print("extractive sum")
            # Average an optimal number of sentences
            average_sent_len    =   np.mean(sent_length)
            ideal_number_sents  =   np.ceil(max_length/average_sent_len)
            document            =   summarize_model(document, num_sentences=int(ideal_number_sents))
        tokenized   =   tokenizer.encode_plus(document, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(torch.device(device))
        
        preds   =   model(**tokenized)
        probs   =   softmax(preds.logits, dim=1).detach().cpu()

        pred_id     =   torch.argmax(probs)
    else:
        tokenized   =   tokenizer.batch_encode_plus(sentences, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(torch.device(device))

        preds       =   model(**tokenized)
        probs   =   softmax(preds.logits, dim=1).detach().cpu()

        pred_id     =   torch.argmax(probs.mean(dim=0))

    pred_label  =   id2label[pred_id.item()]

    return pred_label