from typing import List, Union, Optional, Dict
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets.splits import NamedSplit
import random
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import config

import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_IDX
randSeed = config.RANDSEED
test_num = config.TEST_NUM

class DatasetReader:
    def __init__(self,
                 dataset_path:str = config.DATASET_PATH,
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 ) -> None:
        """
        load train / validation / test dataset.
        """
        self.input_columns = input_columns
        self.output_column = output_column
        
        dataset = load_dataset(dataset_path)  # type: Dict
        self.train_ds:Dataset = dataset['train']
        self.validation_ds = dataset['validation']
        
        self.test_ds = dataset['test']
        random.seed(randSeed)
        selected_indices = random.sample(range(len(self.test_ds)), test_num)
        self.test_ds = self.test_ds.select(selected_indices)  # 提取 test_num 条测试数据

        with open(f'selected_test_idx_{randSeed}.json', 'w') as json_file: # 存序号
            json.dump(selected_indices, json_file)


    
    def train_faiss_load(self, train_faiss_index = config.TRAIN_FAISS_INDEX_PATH):
        """读取 训练集 向量数据库"""
        self.train_index = faiss.read_index(train_faiss_index)

    def embedding_with_LM(self, embed_content, model_path = config.EMBED_MODEL_PATH):
        """用 LM 做前向传播 获取 训练集/测试集 的 embedding, 保存 pkl 文件到本地"""
        # load embed model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        model.eval()

        # select data
        if embed_content not in set(['train', 'test']):
            raise ValueError(f'embedding dataset {embed_content} not exist.')
        else:
            data = self.test_ds if embed_content == 'test' else self.train_ds

        # embed & save
        embed_list = []
        for p_id, item in tqdm(enumerate(data), desc=f'{embed_content} dataset is embedding...'):
            inputs = tokenizer(item['text'], return_tensors='pt', padding=True, truncation=True, max_length=32000)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                all_hidden_states = outputs.hidden_states
            last_layer_hidden = all_hidden_states[-1]
            sentence_embedding = last_layer_hidden.mean(dim=1)  # shape = torch.Size([1, 3584]), 最后一层隐藏层 平均池化结果

            sentence_embedding = np.float64(sentence_embedding.tolist())  # 类型转换：array([[np.float64]])
            sentence_embedding = sentence_embedding/np.linalg.norm(sentence_embedding)  # 归一化

            embed_list.append({
                'id': p_id,
                "embed": sentence_embedding,
                'raw_text': item['text']
            })
            if p_id % 50 == 0:
                with open(f'qwen_embed_{embed_content}{config.RANDSEED}.pkl', 'wb') as file:
                    pickle.dump(embed_list, file)
        with open(f'qwen_embed_{embed_content}{config.RANDSEED}.pkl', 'wb') as file:
            pickle.dump(embed_list, file)

    def embedding_with_bge(self, embed_content, model_path = config.EMBED_MODEL_PATH):
        """用 LM 做前向传播 获取 训练集/测试集 的 embedding, 保存 pkl 文件到本地"""
        # load model
        embedding_model = SentenceTransformer(model_path)
        embedding_model = embedding_model.to('cuda')
        embedding_model.eval()

        # select data
        if embed_content not in set(['train', 'test']):
            raise ValueError(f'embedding dataset {embed_content} not exist.')
        else:
            data = self.test_ds if embed_content == 'test' else self.train_ds

        # embed & save
        embedding_ds = []
        for p_id, item in tqdm(enumerate(data), desc=f'{embed_content} dataset is embedding...'):
            res = embedding_model.encode(item['text'], show_progress_bar=False)
            embedding_ds.append({
                'embed': res,  # 维度 1024
                'id': p_id,
                'raw_text': item['text']
            })

            if p_id % 50 == 0:
                with open(f'bge_embed_{embed_content}{config.RANDSEED}.pkl', 'wb') as file:
                    pickle.dump(embedding_ds, file)
        with open(f'bge_embed_{embed_content}{config.RANDSEED}.pkl', 'wb') as file:
            pickle.dump(embedding_ds, file)



if __name__ == '__main__':
    dataReader = DatasetReader()
    print(len(dataReader.train_ds))
    
    