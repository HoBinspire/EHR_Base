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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,6,7'
randSeed = 36

class DatasetReader:
    def __init__(self,
                 dataset_path:str = '/data/lhb/huggingface/dataset/mimic3_1.4_readmission_prediction',
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 embed = False,  # 是否进行实时 embed
                 embedding_model_path = None
                 ) -> None:
        
        self.input_columns = input_columns
        self.output_column = output_column
        
        dataset = load_dataset(dataset_path)  # Dict
        self.train_ds:Dataset = dataset['train']
        self.validation_ds = dataset['validation']
        
        self.test_ds = dataset['test']
        # begin--------------设置随机种子--------------------------
        random.seed(randSeed)  # 设置随机种子
        selected_indices = random.sample(range(len(self.test_ds)), 200)
        self.test_ds = self.test_ds.select(selected_indices)  # 提取 200条数据
        
        with open(f'selected_indices_{randSeed}.json', 'w') as json_file:   # 存序号
            json.dump(selected_indices, json_file)
        # end------------------------------------------------------



        if embed:
            if 'Qwen' in embedding_model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, use_fast=False, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(embedding_model_path, device_map="auto", trust_remote_code=True)
                self.model.eval()
            else: # Todo
                pass


    
    def train_faiss_load(self):
        """对向量语料库进行赋值"""
        self.train_index = faiss.read_index('/data/lhb/test-openicl-0.1.8/EHR_Base/faiss_index/faiss_index_qwen7B_train_readmission.index')

    def embedding_with_LM(self):
        """用 LM 做前向传播 获取 训练集的 embedding"""
        embed_list = []
        for p_id, item in tqdm(enumerate(self.test_ds)):
            inputs = self.tokenizer(item['text'], return_tensors='pt', padding=True, truncation=True, max_length=32000)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                all_hidden_states = outputs.hidden_states
            last_layer_hidden = all_hidden_states[-1]
            sentence_embedding = last_layer_hidden.mean(dim=1)  # shape = torch.Size([1, 3584]), 最后一层隐藏层 平均池化结果

            sentence_embedding = np.float64(sentence_embedding.tolist())  # 类型转换：array([[np.float64]])
            sentence_embedding = sentence_embedding/np.linalg.norm(sentence_embedding)  # 归一化

            embed_list.append({
                'id': p_id,
                "embed": sentence_embedding
            })
            if p_id % 50 == 0:
                with open('qwen_embed_test36.pkl', 'wb') as file:
                    pickle.dump(embed_list, file)
        with open('qwen_embed_test36.pkl', 'wb') as file:
            pickle.dump(embed_list, file)

        self.test_embedding_ds = embed_list  # 成员变量赋值

    def embedding_with_bge(self, embedding_model_path = '/data/lhb/huggingface/model/sentence_embedding/bge-m3'):
        # todo
        embedding_model = SentenceTransformer(embedding_model_path)
        embedding_model = embedding_model.to('cuda')
        embedding_model.eval()

        self.test_embedding_ds = []  # 测试集 的嵌入向量
        for _, item in tqdm(enumerate(self.test_ds), desc='test_ds is embedding...'):
            res = embedding_model.encode(item['text'], show_progress_bar=False)
            self.test_embedding_ds.append({
                'embed': res,  # 维度 1024
                'id': _,
                'raw_text': item['text']
            })

            if _ % 50 == 0:
                with open('bge_embed_test36.pkl', 'wb') as file:
                    pickle.dump(self.test_embedding_ds, file)
        with open('bge_embed_test36.pkl', 'wb') as file:
            pickle.dump(self.test_embedding_ds, file)



if __name__ == '__main__':
    dataReader = DatasetReader()
    dataReader.embedding_dataset()
    
    