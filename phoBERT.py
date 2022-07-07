import torch, re
from string import punctuation
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

pattern = '[{}]'.format(punctuation)
def preprocess_text(text:str) -> str:
    r"""
    xử lý loại bỏ các ký tự đặc biệt trong text
    ví dụ EMA::IL: -> email
    """
    return re.sub(pattern, '', text.lower(), count=len(text)).strip()


@torch.no_grad()
def word2vec(sentence:str) -> torch.Tensor:
    sentence = preprocess_text(sentence)
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    features = phobert(input_ids)  # Models outputs are now tuples
    return features['pooler_output'].view(-1) #convert to 1d-tensor


if __name__ == '__main__':
    sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  

    # input_ids = torch.tensor([tokenizer.encode(sentence)])

    # with torch.no_grad():
    #     features:BaseModelOutputWithPoolingAndCrossAttentions \
    #         = phobert(input_ids)  # Models outputs are now tuples
    

