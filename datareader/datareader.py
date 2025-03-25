from typing import List, Union, Optional, Dict
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets.splits import NamedSplit
import random
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'

class DatasetReader:
    def __init__(self,
                 dataset_path:str = '/data/lhb/huggingface/dataset/mimic3_1.4',
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 ) -> None:
        
        self.input_columns = input_columns
        self.output_column = output_column
        
        dataset = load_dataset(dataset_path)  # Dict
        self.train_ds:Dataset = dataset['train']
        self.validation_ds = dataset['validation']
        self.test_ds = dataset['test']

    
    def embedding_dataset(self, embedding_model_path = None, tokenizer_path = ''):
        """对向量语料库进行赋值"""
        # todo 实时 嵌入
        


        # 训练集 的 faiss索引
        self.train_index = faiss.read_index('/data/lhb/huggingface/faiss_index/mimic3_train_faiss_idx.index')

        # 测试集的 检索向量
        self.__embedding()


    def __embedding(self, embedding_model_path = '/data/lhb/huggingface/model/sentence_embedding/bge-m3'):
        # todo
        embedding_model = SentenceTransformer(embedding_model_path)
        embedding_model = embedding_model.to('cuda')
        embedding_model.eval()

        self.test_embedding_ds = []  # 测试集 的嵌入向量
        for _, item in tqdm(enumerate(self.test_ds), desc='test_ds is embedding...'):
            if _ not in range(0, 200):  # 20条
                continue

            res = embedding_model.encode(item['text'], show_progress_bar=False)
            self.test_embedding_ds.append({
                'embed':res,  # 维度 1024
                'id': _,
                'raw_text': item['text']
            })

if __name__ == '__main__':
    dataReader = DatasetReader()
    dataReader.embedding_dataset()
    
    