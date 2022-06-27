import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

@torch.no_grad()
def word2vec(sentence:str) -> torch.Tensor:
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    features = phobert(input_ids)  # Models outputs are now tuples
    return features['pooler_output'].view(-1) #convert to 1d-tensor
a = word2vec("cộng")


print("===============================")

phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
@torch.no_grad()
def word2vec(sentence:str) -> torch.Tensor:
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    features = phobert(input_ids)  # Models outputs are now tuples
    return features['pooler_output'].view(-1) #convert to 1d-tensor

b = word2vec("cộng")
print(a-b)

print(a[:16])
print(b[:16])