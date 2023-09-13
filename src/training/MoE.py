import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertModel

class MoE_MultiLabelClassification(nn.Module):
    """
    Mixture Of Experts of BERTs for Multi-Label classification :
    • Each BERT Expert has been pre-fine-tuned on its respective type of documents (resp. "World", "Sports", "Business", "Sci/Tech") for Binary classification
    • The MoE uses a simple Gating method
    • The Gating Model is a BERT Model followed by Linear layers that maps from the hidden size of BERT to the number of experts (4 here)
    • The Model is Trained with Cross Entropy Loss (As it is for Multi-Class classification)
    • The model pass is as follows :
        1. The input sequence is passed to each experts from which we extract their pooler_output
        2. The input sequence is passed to the Gating Network that will outputs the expert's weights that will be the Softmax of the outputs of the Linear Layer
        3. The experts weights and the pooler_output of each experts are multiplied
        4. The result of 3. is passed to the Classifier Network (Linear Layers with GeLU and Dropout)
    """
    
    def __init__(self, expert_model_world:BertModel, expert_model_sports:BertModel, expert_model_business:BertModel, expert_model_sci:BertModel, hidden_dim=256, dropout=0.1) -> None:
        super(MoE_MultiLabelClassification, self).__init__()
        ## EXPERT Layers
        self.expert_model_world     =  expert_model_world
        self.expert_model_sports    =  expert_model_sports
        self.expert_model_business  =  expert_model_business
        self.expert_model_sci       =  expert_model_sci

        ## GATING Layers
        self.gating_bert    =   BertModel.from_pretrained('bert-base-uncased')
        self.gating_layer1  =   nn.Linear(self.gating_bert.config.hidden_size, hidden_dim)
        self.gating_layer2  =   nn.Linear(hidden_dim, 4)

        ## CLASSIFIER Layers
        self.classifier_layer1  =   nn.Linear(self.gating_bert.config.hidden_size, hidden_dim)
        self.classifier_layer2  =   nn.Linear(hidden_dim, 4)

        ## ACTIVATION FUNCTIONS, SOFTMAX & DROPOUT
        self.dropout        =   nn.Dropout(dropout)
        self.softmax        =   nn.Softmax(dim=1)
        self.gelu           =   torch.nn.GELU()
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        # Getting the stacked pooler output's of each experts
        pooler_outs =   self.experts_pass(input_ids, attn_mask, token_type_ids)

        # Getting the experts weights
        experts_weights =   self.gating(input_ids, attn_mask, token_type_ids).unsqueeze(2)

        # Taking the sum of the pooler ouputs to get the final embedding
        weighted_preds      =   pooler_outs * experts_weights
        final_embeddings    =   weighted_preds.sum(dim=1)

        # Passing the final embedding to the classifier
        final_preds =   self.classifier(final_embeddings)

        return final_preds

    def experts_pass(self, input_ids:Tensor, attn_mask:Tensor, token_type_ids:Tensor) -> Tensor:
        """Take the input sequence, pass it through each expert, stack them"""
        out_wordl       =   self.expert_model_world(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output
        out_sports      =   self.expert_model_sports(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output
        out_business    =   self.expert_model_business(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output
        out_sci         =   self.expert_model_sci(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output

        pooler_outs =   torch.stack([out_wordl, out_sports, out_business, out_sci], dim=1)

        return pooler_outs
    
    def classifier(self, weighted_sum_embeddings:Tensor) -> Tensor:
        """Pass the weighted sum of the embedding's of each experts to the classifier"""
        output  =   self.gelu(self.classifier_layer1(weighted_sum_embeddings))
        output  =   self.dropout(output)
        output  =   self.classifier_layer2(output)

        return output
    
    def gating(self, input_ids:Tensor, attn_mask:Tensor, token_type_ids:Tensor) -> Tensor:
        """The Gating Networks : Takes the input sequence (the same as the experts) and outputs weights for each of the experts"""
        pooler_out  =   self.gating_bert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids).pooler_output
        gating_out  =   self.gelu(self.gating_layer1(pooler_out))
        gating_out  =   self.dropout(gating_out)
        gating_out  =   self.gating_layer2(gating_out)
        prob_gating =   self.softmax(gating_out)

        return prob_gating