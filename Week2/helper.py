# import os
# import sys
# import pandas as pd
# import numpy as np
# import torch
# import random
# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
# from transformers import BertTokenizer, BertModel
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.nn import CrossEntropyLoss

def set_device():
    # device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    print(device)
    return device

class CustomDataset(Dataset):
  """
  - input_data: list of string
  - target_data: list of int
  """

  def __init__(self, input_data:list, target_data:list) -> None:
      self.X = input_data
      self.Y = target_data

  def __len__(self):
      return len(self.X)

  def __getitem__(self, index):
      return self.X[index],self.Y[index]


def custom_collate_fn(batch):
    """
    - batch: list of tuples (input_data(string), target_data(int))
    
    한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
    이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
    
    한 배치 내 레이블(target)은 텐서화 함.
    
    (input, target) 튜플 형태를 반환.
    """
    input_list, target_list = [], []
    
    for _input, _target in batch:
        input_list.append(_input)
        target_list.append(_target)
    
    tensorized_input = tokenizer_bert(
        input_list,
        add_special_tokens=True,
        padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
        truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
        max_length=512,
        return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
    )
    
    tensorized_label = torch.tensor(target_list)
    
    return tensorized_input, tensorized_label

class CustomClassifier(nn.Module):
    def __init__(self, hidden_size: int, n_label: int):
        super(CustomClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("klue/bert-base")

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, linear_layer_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(linear_layer_hidden_size, 2)
        )  # torch.nn에서 제공되는 Sequential, Linear, ReLU, Dropout 함수 활용

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # BERT 모델의 마지막 레이어의 첫번재 토큰을 인덱싱
        # 마지막 layer의 첫 번째 토큰 ("[CLS]") 벡터를 가져오기, shape = (1, hidden_size)
        cls_token_last_hidden_states = outputs['pooler_output']

        logits = self.classifier(cls_token_last_hidden_states)

        return logits

