import torch
from pytorch_transformers import BertModel, BertTokenizer
from constants import device

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)  # 텍스트를 토큰 단위로 분해 : str

masked_index = 8  # 가릴 부분 설정
tokenized_text[masked_index] = "[MASK]"

indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)  # numpy로 변환
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 두 문장을 구분. [SEP]까지 0 작성

tokens_tensor = torch.tensor([indexed_token])  # torch.tensor는 numpy를 tensor로 변환
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

model.to(device)
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)

with torch.no_grad():  # grad 계산을 안함으로써 계수 수정을 하지 않는다. pretrained model의 계수를 그대로 쓰기위해
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    cls = outputs[1]  # 0번째 값은 글자들의 벡터값, 1번째 값은 문장의 class값

