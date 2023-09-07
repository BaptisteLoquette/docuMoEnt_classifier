import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, AdamW

class MoE_MultiLabelClassification(nn.Module):
    """
    Mixture Of Experts of BERTs for Multi-Label classification :
    • Each BERT Expert has been pre-fine-tuned on its respective type of documents (resp. "World", "Sports", "Business", "Sci/Tech") for Binary classification
    • The MoE uses a simple Gating method
    • The Gating Model is a BERT Model followed by a Linear layer
    • The Model is Trained with Cross Entropy Loss
    • The model pass is as follows :
        1. The input sequence is passed to each experts from which we extract their 'positive' prediction logit
        2. The input sequence is passed to the Gating model that will outputs the expert's weights that will be the Softmax of the outputs of the Linear Layer
        3. The experts weights and the predictions of each experts are multiplied element-wise
    """
    
    def __init__(self, expert_model_world:BertForSequenceClassification, expert_model_sports:BertForSequenceClassification, expert_model_business:BertForSequenceClassification, expert_model_sci:BertForSequenceClassification) -> None:
        super().__init__()
        ## EXPERT Layers
        self.expert_model_world     =  expert_model_world
        self.expert_model_sports    =  expert_model_sports
        self.expert_model_business  =  expert_model_business
        self.expert_model_sci       =  expert_model_sci

        ## GATING Layers
        self.gating_bert    =   BertModel.from_pretrained('bert-base-uncased')
        self.gating_layer   =   nn.Linear(self.gating_bert.config.hidden_size, 4)
        self.dropout        =   nn.Dropout(0.1)
        self.softmax        =   nn.functional.softmax
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        probs   =   self.experts_pass(input_ids, attn_mask, token_type_ids)

        experts_weights =   self.gating(input_ids, attn_mask, token_type_ids)

        final_pred  =   experts_weights * probs

        return final_pred

    def experts_pass(self, input_ids, attn_mask, token_type_ids):
        out_wordl       =   self.expert_model_world(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        out_sports      =   self.expert_model_sports(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        out_business    =   self.expert_model_business(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        out_sci         =   self.expert_model_sci(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        probs   =   torch.cat([out_wordl.logits[:, 1], out_sports.logits[:, 1], out_business.logits[:, 1], out_sci.logits[:, 1]], dim=0).reshape(-1, 4)   # Select only the "positive" logit

        return probs
    
    def gating(self, input_ids, attn_mask, token_type_ids):
        pooler_out  =   self.gating_bert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output
        gating_out  =   self.dropout(self.gating_layer(pooler_out))
        prob_gating =   self.softmax(gating_out, dim=1)

        return prob_gating