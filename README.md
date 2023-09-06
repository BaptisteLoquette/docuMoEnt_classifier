# Multi-Label Document Classifier
Multi Label Document Classifier using different methods of training and inference

## Fine-Tuning :

### Single BERT Classifier :
- Dataset : `ag_news`
- BERT Model : `bert-base-cased`
- Epochs : 2
- lr : 5e-6
- Training Samples : 170_000
- Train Batch Size : 8
- Eval Batch Size : 16
 
#### Training Results :
- **Val Accuracy :**

  !["train_loss"](https://github.com/BaptisteLoquette/docuMoEnt_classifier/blob/main/images/val_acc.png) 
- **Train Loss :**
 !["train_loss"](https://github.com/BaptisteLoquette/docuMoEnt_classifier/blob/main/images/train_loss.png)

- **Learning Rate Schedule :**
 !["lr"](https://github.com/BaptisteLoquette/docuMoEnt_classifier/blob/main/images/Learning_rate.png)

### MoE BERT Classifier :

#### MoE with Gating
 Incoming...
 Method :
 - Train `n` BERT Experts models, for Binary Classification, on subsets of the `n` categories, respectively
 - Implement a BERT model + Linear layer + Softmax, as the gating model that outputs `n` weights (one for each expert)
 - Apply the weights from the gating model to each Expert's output
 - Taking the argmax of the weighted outputs of the Experts
 - Train the Experts and the Gating model together on the full dataset

## Inference
The inference can be ran using different methods depending on the length of the document :

- On the first 512 tokens of the document
- Using extractive summarization, by finding the optimal number of sentences of the summary (for long documents)
- By classifying each chunk of the document then aggregating the predictions (for long documents)


## Possible improvements :
- Training on a larger dataset
- Training on more epochs
- Using a larger BERT model
- Fine-Tuning BERT model summarizer
