path_data = "./dataset"
path_results = "./results"
path_logs = "./logs"
path_models = "./models"


import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import re
import joblib


model_name = "Rostlab/prot_bert_bfd"


class toot_bert_m(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasetFolderPath = path_data
        self.trainFilePath = os.path.join(self.datasetFolderPath, 'train.csv')
        self.testFilePath = os.path.join(self.datasetFolderPath, 'test.csv')
        self.valFilePath = os.path.join(self.datasetFolderPath, 'validation.csv')


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        if split=="train":
          self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        elif split == "test":
          self.seqs, self.labels = self.load_dataset(self.testFilePath)
        else:
          self.seqs, self.labels = self.load_dataset(self.valFilePath)

        self.max_length = max_length

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['sequence','label'],skiprows=1)
        self.labels_dic = {0:'nonmembrane',
                           1:'membrane'}

        df['labels'] = np.where(df['label']=='membrane', 1, 0)
        
        seq = list(df['sequence'])
        label = list(df['labels'])

        assert len(seq) == len(label)
        return seq, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample


train_dataset = toot_bert_m(split="train")
val_dataset = toot_bert_m(split="valid")
test_dataset = toot_bert_m(split="test")



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }


def model_init():
  return AutoModelForSequenceClassification.from_pretrained(model_name)



training_args = TrainingArguments(
    output_dir=path_results,          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=path_logs,            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after eachh epoch
    gradient_accumulation_steps=64,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="toot_bert_m",       # experiment name
    seed=3,                           # Seed for experiment reproducibility 3x3
    metric_for_best_model="eval_mcc",
    greater_is_better=True
)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)

trainer.train()


trainer.save_model(path_models)

predictions, label_ids, metrics = trainer.predict(test_dataset)
