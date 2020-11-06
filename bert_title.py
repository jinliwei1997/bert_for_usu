from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import sys
from tqdm import trange
import torch
import json

def assign_GPU(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda')

    output = {'input_ids' : tokens_tensor,
          'token_type_ids' : token_type_ids,
          'attention_mask' : attention_mask}
    return output

if __name__ == '__main__':
    with open(sys.argv[1],'r',encoding='utf-8') as f:
        title_dict = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('./model')
    model = AutoModel.from_pretrained("./model")
    model.to('cuda')

    lines = []
    for key in title_dict:
        lines.append((key,title_dict[key]))
    tot = len(lines)

    title_feature_dict = {}

    with torch.no_grad():
        for i in trange(tot):
            line = lines[i][1].rstrip()
            id = lines[i][0]
            print(tokenizer.tokenize((line)))
            inputs = assign_GPU(tokenizer(line, return_tensors="pt"))
            outputs = model(**inputs)
            feature = outputs[1][0].tolist()
            title_feature_dict[id] =feature

    with open('title_feature_list.json', 'w', encoding='utf-8') as file:
        json.dump(title_feature_dict, file, ensure_ascii=False)