from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import sys
from tqdm import trange
import torch
import json
import os
import ast
def assign_GPU(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda')

    output = {'input_ids' : tokens_tensor,
          'token_type_ids' : token_type_ids,
          'attention_mask' : attention_mask}
    return output

def extract_feature(vid_path):
    vid_path = vid_path.strip()
    video_id = vid_path.split('/')[-1].split('.')[0]
    video_dir = '/'.join(vid_path.split('/')[1:-1])
    frame_dir = os.path.join("frame_extract", video_dir, video_id)
    vid_path_new = os.path.join("videos", '/'.join(vid_path.split('/')[1:]))
    ocr_dir = os.path.join("ocr_zh", video_dir)
    ocr_feature_dir = os.path.join("ocr_feature", video_dir)

    if not os.path.exists(ocr_feature_dir):
        print('create dir: ' + ocr_feature_dir)
        os.popen(f'mkdir -p {ocr_feature_dir}')

    if os.path.exists(ocr_feature_dir + '/' + video_id + '.json'):
        return

    f = open(ocr_dir+'/'+video_id+'.zh')

    try:
        ocr_record = ast.literal_eval(f.readlines()[0])
    except:
        return
    f.close()

    # print(f'frame number: {ocr_record[0]} ; fps: {ocr_record[1]} ; len: {len(ocr_record)}')
    feature_dict = {}
    feature_dict['tot_frame'] = ocr_record[0]
    feature_dict['fps'] = ocr_record[1]
    with torch.no_grad():
        for i in range(2,len(ocr_record)):
            frame_id = ocr_record[i][0]
            item_list = ocr_record[i][1]
            sentence = ''
            for item in item_list:
                sentence += item[1]
            # print(f'frame_id: {frame_id}, sentence:{sentence}')
            inputs = assign_GPU(tokenizer(sentence, return_tensors="pt"))
            outputs = model(**inputs)
            feature = outputs[1][0].tolist()
            feature_dict[frame_id] = feature

    with open(ocr_feature_dir + '/' + video_id + '.json', 'w', encoding='utf-8') as file:
        json.dump(feature_dict, file, ensure_ascii=False)

if __name__ == '__main__':
    split_id = sys.argv[1]

    f = open(f'split_list/video_list_split_{split_id}.txt','r',encoding='utf-8')
    lines = f.readlines()
    f.close()

    tokenizer = BertTokenizer.from_pretrained('./bert/model')
    model = AutoModel.from_pretrained("./bert/model")
    model.to('cuda')

    tot = len(lines)
    for i in trange(tot):
        extract_feature(lines[i])






    # with torch.no_grad():
    #     for i in trange(tot):
    #         line = lines[i][1].rstrip()
    #         id = lines[i][0]
    #         print(tokenizer.tokenize((line)))
    #         inputs = assign_GPU(tokenizer(line, return_tensors="pt"))
    #         outputs = model(**inputs)
    #         feature = outputs[1][0].tolist()
    #         title_feature_dict[id] =feature
    #
    # with open('title_feature_list.json', 'w', encoding='utf-8') as file:
    #     json.dump(title_feature_dict, file, ensure_ascii=False)
