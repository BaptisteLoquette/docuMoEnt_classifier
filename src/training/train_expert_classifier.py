import os
import torch
import argparse
from datasets import load_from_disk
from training_utils import compute_metrics
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification

parser  =   argparse.ArgumentParser(
    prog="train_bert_classifier",
    description="Train BERT Binary Classification"
    )
parser.add_argument('dset_path', help="Input dir of the preprocessed files", type=str)
parser.add_argument('--path_out', help="Output dir of model", type=str, default="experts")
parser.add_argument('--learning_rate', help="Learning Rate", type=float, default=5e-6)
parser.add_argument('--num_epochs', help="Number of epochs", type=int, default=2)

if __name__ == "__main__":
    args        =   parser.parse_args()
    lr          =   args.learning_rate
    num_epochs  =   args.num_epochs
    dset_path   =   args.dset_path
    path_out    =   args.path_out
    expert_path =   os.path.basename(dset_path)

    device  =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dset            =   load_from_disk(dset_path)   # Loading Dataset (Ag News)
    bert_classifier =   BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device) # Loading Classifier model

    ## Specify Arguments for Training
    training_args   =   TrainingArguments(
        output_dir=os.path.join(path_out, f"expert_{expert_path}"),
        learning_rate=lr,
        warmup_ratio=0.1,
        evaluation_strategy='steps',
        eval_steps=int(0.2 * len(dset["train"])),
        num_train_epochs=num_epochs,
        logging_dir=f"logs/{path_out}",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        report_to="wandb"
        )

    ## Initialize the huggingface's Trainer
    trainer =   Trainer(
        model=bert_classifier,
        args=training_args,
        train_dataset=dset["train"],
        eval_dataset=dset["test"],
        compute_metrics=compute_metrics,
        )
    
    trainer.train()
    trainer.model.save(os.path.join(path_out, f"expert_{expert_path}/model"))  # Saving model