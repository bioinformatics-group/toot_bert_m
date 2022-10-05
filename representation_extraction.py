from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re

path_model = './models/'

path_data = "./dataset/"

path_representation = "./representations/"


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)

# Print loading model from
print("Loading model from: " + path_model)

model = BertModel.from_pretrained(path_model)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()
# Read train data from CSV file
df_train = pd.read_csv(path_data + "train.csv", names=['sequence', 'label'], skiprows=1)
# Read validation data from CSV file
df_val = pd.read_csv(path_data + "validation.csv", names=['sequence', 'label'], skiprows=1)

# Combine train and validation data
df_train = pd.concat([df_train, df_val])

# Print train size
print("Train size: " + str(len(df_train)))

# Read test data from CSV file
df_test = pd.read_csv(path_data + "test.csv", names=["sequence", "label"], skiprows=1)

# Print test size
print("Test size: " + str(len(df_test)))


representation_task_dict = {"train": df_train, "test": df_test}

# Print the start of the representation extraction
print("Start representation extraction")

# For each task, extract the representation
for task in representation_task_dict:
    # Print the progress
    print("Extracting representation for task: " + task)
    # Get the dataframe
    df = representation_task_dict[task]
    # Get the sequences
    sequence_list = df['sequence'].tolist()
    # Get the labels
    label_list = df['label'].tolist()

    representation_list = []

    for sequence in sequence_list:

        seqs = [sequence]
        seqs = [" ".join("".join(seq.split())) for seq in seqs]
        seqs = [re.sub(r"[UZOB]", "X", sequence)
                    for sequence in seqs]

        ids = tokenizer.batch_encode_plus(
            seqs, add_special_tokens=True, padding=True, truncation=True, max_length=10000)

        input_ids = torch.tensor(ids['input_ids']).to(device)

        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        embedding = embedding.cpu().numpy()

        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)

        mean_pool = np.mean(features[0], axis=0)

        # Append the representation to the list
        representation_list.append(mean_pool)

    # Create a dataframe with the representation_list with 1024 columns
    df_representation = pd.DataFrame(representation_list, columns=["representation_" + str(i) for i in range(1024)])

    df_representation['label'] = label_list
    
    # Print the progress
    print("Saving representation for task: " + task)

    # Save the dataframe to CSV file
    df_representation.to_csv(path_representation + task + ".csv", index=False)




