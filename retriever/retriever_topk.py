"""Topk Retriever"""

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

from datareader.datareader import DatasetReader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'



class TopkRetriever:
    def __init__(self,
                 dataset_path:str = '/data/lhb/huggingface/dataset/mimic3_1.4',
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 ) -> None:
        
        # 加载数据
        self.dataReader = DatasetReader(dataset_path, input_columns, output_column)  # 加载：训练集，测试集，验证集
        # self.dataReader.embedding_dataset()  # 加载训练集的 faiss 向量索引， 测试集的 embedding 向量

        # 储存结果
        self.res_idx_list = []

    def topk_retrive(self, ice_num = 3):
        """Topk 检索结果"""
        for item in tqdm(self.dataReader.test_embedding_ds, desc=f"Top{ice_num} retrive is going..."):
            embed = np.expand_dims(item['embed'], axis=0)  # 由于 search 的参数需要是 二维的，这里增加一个维度
            near_ids = self.dataReader.train_index.search(embed, ice_num)[1][0].tolist()  # search 返回结果 (array([[0.9051953, 0.904572 , 0.9034882]], dtype=float32),array([[36228, 30725, 26155]]))
            
            self.res_idx_list.append(near_ids)
        
        return self.res_idx_list

    def retrive(self):
        return self.topk_retrive()


if __name__ == '__main__':    
    x = TopkRetriever()
    print(x.retrive())
