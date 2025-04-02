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
import os
import json

from datareader.datareader import DatasetReader
import config

class TopkRetriever:
    def __init__(self,
                 dataset_path:str = config.DATASET_PATH,
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 ) -> None:
        
        # load dataset & faiss_index
        self.dataReader = DatasetReader(dataset_path, input_columns, output_column)
        self.dataReader.train_faiss_load()

        # 储存结果
        self.res_idx_list = []

    def topk_retrive(self, test_embed_path, ice_num = config.ICE_NUM):
        """Topk 检索结果
        
        test_embed_path: test_data_embedding pkl 文件
        """
        if not test_embed_path:
            raise ValueError(f'test_embed_path not provided.')

        with open(test_embed_path, 'rb') as file:
            test_embedding_ds = pickle.load(file)

        # TODO: test_embed.pkl 文件中的数据以什么统一格式 存储
        
        res_idx_list = []
        for item in tqdm(test_embedding_ds, desc=f"Top{ice_num} retrive is going..."):
            embed = np.expand_dims(item['embed'], axis=0)  # 由于 search 的参数需要是 二维的，这里增加一个维度 # ①
            # embed = np.float32(item['embed'])  # ② qwen embed 测试集检索方式
            near_ids = self.dataReader.train_index.search(embed, ice_num)[1][0].tolist()  # search 返回结果 (array([[0.9051953, 0.904572 , 0.9034882]], dtype=float32),array([[36228, 30725, 26155]]))
            
            res_idx_list.append(near_ids)
        
        return res_idx_list

    def retrive(self, test_embed_path, ice_num = config.ICE_NUM):
        res_idx_list = self.topk_retrive(test_embed_path, ice_num)

        with open(f'top{ice_num}_idx.json', 'w', encoding='utf-8') as file:
            json.dump(res_idx_list, file, ensure_ascii=False, indent=4)        

        return res_idx_list


if __name__ == '__main__':    
    x = TopkRetriever()
    print(x.retrive('/data/lhb/test-openicl-0.1.8/EHR_Base/results/readmission_prediction/randseed42/bge_embed_test42.pkl'))
