import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """Computes the accuracy of the batches of the validation dataset"""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)