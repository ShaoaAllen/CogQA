import torch
from transformers import BertTokenizer, AlbertModel
tokenizer = BertTokenizer.from_pretrained("./albert_base")
bert_model = AlbertModel.from_pretrained("./albert_base")
"""下游微调任务"""
input_ids = torch.tensor(tokenizer.encode("Which magazine was started first Arthur's Magazine or First for Women")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
sequence_output = outputs[0]
pooled_output = outputs[1]
print(sequence_output)
print(pooled_output)
print(sequence_output.shape)    ## 字向量py
print(pooled_output.shape)      ## 句向量ls