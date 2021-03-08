import torch
from pytorch_transformers import BertTokenizer,BertModel

tokenizer = BertTokenizer.from_pretrained('/home/shaoai/CogQA/uncased_L-2_H-128_A-2')
model = BertModel.from_pretrained('/home/shaoai/CogQA/uncased_L-2_H-128_A-2')

input_ids = torch.tensor(tokenizer.encode("Which magazine was started first Arthur's Magazine or First for Women")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
sequence_output = outputs[0]
pooled_output = outputs[1]
print(sequence_output)
print(pooled_output)
print(sequence_output.shape)    ## 字向量
print(pooled_output.shape)      ## 句向量ls
