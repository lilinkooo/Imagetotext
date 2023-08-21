import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from Image_desc_model import ImageCaptioningModel
import os

# # data
# # 指定要加载的.npy文件的路径
# file_path = 'image_features/10815824_2997e03d76.npy'
# # 使用numpy.load()加载.npy文件
# loaded_data = np.load(file_path)
# # 现在loaded_data中包含了.npy文件中的数据
# # print(loaded_data)
# image_features = {}
# image_features['10815824_2997e03d76.jpg'] = torch.tensor(loaded_data)
# description_sequences = {'10815824_2997e03d76.jpg': [[101, 100, 100, 100, 8256, 143, 100, 9375, 8217, 143, 9296, 100, 8995, 100, 8243, 143, 11229, 8217, 143, 100, 119, 102], [101, 100, 9375, 8256, 11908, 100, 100, 8120, 143, 11229, 119, 102], [101, 100, 9375, 100, 143, 100, 100, 100, 100, 143, 11229, 119, 102], [101, 100, 8791, 117, 8256, 9375, 8256, 10380, 100, 8995, 100, 143, 100, 11229, 119, 102], [101, 100, 8777, 8256, 10380, 100, 100, 143, 11229, 119, 102]]}

# 加载BERT模型和分词器
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = 21128
stop_tokens = [tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]

# 初始化模型
model = ImageCaptioningModel(bert_model, hidden_size=256, vocab_size=vocab_size)

# 使用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 假设使用交叉熵损失
criterion = nn.CrossEntropyLoss()
# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(image_id, image_feature, description_sequences):
    seq_list = description_sequences[image_id]
    for seq in seq_list:
        input_ids = torch.tensor(seq).unsqueeze(0).to(device)
        # image_feature = torch.tensor(image_features[image_id]).unsqueeze(0).to(device)
        outputs = model(image_feature, input_ids[:, :-1])
        targets = input_ids[:, 1:]  # 目标序列

        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

def generate_description(model, image_feature, max_length, tokenizer):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)
        image_feature = torch.tensor(image_feature).unsqueeze(0).to(device)
        
        for _ in range(max_length):
            outputs = model(image_feature, input_ids)
            predictions = outputs.argmax(dim=-1)
            
            # 添加预测的词
            input_ids = torch.cat((input_ids, predictions[:, -1:]), dim=-1)
            
            # 如果生成了任意停止标志，停止生成
            if predictions[0, -1] in stop_tokens:
                break
        
        # 将生成的词转换回文本
        generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
        return generated_text

# num_epochs = 100
# for i in range(num_epochs):
#     loss = train('10815824_2997e03d76.jpg', image_features, description_sequences)
#     if i % 50 == 0:
#         print(f'epoch{i+1},loss={loss}')

files = os.listdir('utils_data/image_features')

with open('utils_data/text_tokens.pkl','rb') as file:
    description_sequences = pickle.load(file)
with open('data_text/Flickr_8k.testImages.txt') as f:
# with open('test.txt') as f:
    text = f.read()
    image_files = text.split('\n')
batch_image_ids = image_files

# print(description_sequences[batch_image_ids[0][:-3]+'jpg'])

count = 0
num_epochs = 100

for batch_image_id in batch_image_ids:
    for i in range(num_epochs):
        image_feature = torch.tensor(np.load('utils_data/image_features/' + batch_image_id[:-3]+'npy'))
        batch_loss = train(batch_image_id, image_feature, description_sequences)
    count += 1
    print(f'count:{count},loss={batch_loss}')
    if count % 30 == 0:
        print(f'epoch:{count},loss={batch_loss}')
    if  count == 500:
        break

# torch.save(model.state_dict(), 'weights/model.pth')
image_feature = torch.tensor(np.load('utils_data/image_features/' + batch_image_ids[0][:-3]+'npy'))
generated_desc = generate_description(model, image_feature, max_length=30, tokenizer=tokenizer)
print("Generated Description:", generated_desc)
