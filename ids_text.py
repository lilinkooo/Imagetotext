import token
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import nltk
from nltk.corpus import stopwords
import spacy
import pickle

# 定义一个空字典来存储图像描述
text_tokens = {}
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 指定解压缩后的模型文件夹路径
model_path = 'en_core_web_sm'
nlp = spacy.load(model_path)

# 加载停用词
stop_words = set(stopwords.words('english'))

count = 0

def get_text(filepath):
    image_descriptions = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                # 分割图像名和描述
                parts = line.split('\t')
                image_name, description = parts[0], parts[1]
                
                # 提取图像序号
                image_id = image_name.split('#')[0]
                
                # 将描述添加到字典中
                if image_id in image_descriptions:
                    image_descriptions[image_id].append(description)
                else:
                    image_descriptions[image_id] = [description]
        return image_descriptions



image_descriptions = get_text('data_text/Flickr8k.lemma.token.txt')
# print(len(image_descriptions))

for image_id, descriptions in image_descriptions.items():
    text_tokens[image_id] = []
    l = []
    for desc in descriptions:
        tokens = [token.text for token in nlp(desc)]
        # 
        # tokens = tokenizer.tokenize(desc)
        tokens = ['[CLS]'] + tokens +['[SEP]']
        # tokens = [word for word in tokens if word.lower() not in stop_words]
        l.append(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        text_tokens[image_id].append(input_ids)
    count += 1
    if count % 300 == 0:
        print(f'{count}/8092')
        
print(text_tokens)
with open('text_tokens.pkl', 'wb') as pickle_file:
    pickle.dump(text_tokens, pickle_file)
    # print(l)
