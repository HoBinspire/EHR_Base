"""Cone Retriever"""

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

from datareader import DatasetReader
from retriever_topk import TopkRetriever

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class ConeRetriever(TopkRetriever):
    def __init__(self,
                 dataset_path:str = '/data/lhb/huggingface/dataset/mimic3_1.4',
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 inferrence_model_path: str = '/data/lhb/huggingface/model/tokenizer/Qwen2.5-7B-instruct'
                 ) -> None:
        
        # 初始化父类对象
        super().__init__(dataset_path, input_columns, output_column)
        self.device = None
        
        # 加载推理模型 & 分词器
        self.inference_model = AutoModelForCausalLM.from_pretrained(inferrence_model_path, device_map="auto", trust_remote_code=True)
        self.inference_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(inferrence_model_path, use_fast = False, trust_remote_code = True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 配置 填充符 为结束符
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        

    def cone_retrive(self, candidate_ice_num = 10, ice_num = 3):
        """Topk+Cone 检索结果"""
        self.candidate_ice_idx_list = self.topk_retrive(ice_num=candidate_ice_num)  # 获取 topk 候选检索结果

        task_head = "Here are a few examples, where <text> represents a description of the patient's health condition, and <label> represents whether the patient will die within 14 days.\n"

        res_idx_list = []
        for p_idx, patient_ice_idx in tqdm(enumerate(self.candidate_ice_idx_list), desc=f"病人开始 cone 检索..."):  # 每个病人
            ## 1. 组装 prompt，计算 两种掩码
            prompt_list = []
            ice_mask_lengths = []
            test_mask_lengths = []
            current_patient = '## test input:\n<text>\n' + self.dataReader.test_ds[p_idx]['text'] + '\n</text>'
            for _, idx in enumerate(patient_ice_idx):
                # 拼接 prompt 计算 ice 掩码长度
                prompt = task_head + f'## example:\n<text>\n' + self.dataReader.train_ds[idx]['text'] + '\n</text>\n<label>\n' + self.dataReader.train_ds[idx]['label'] + '\n</label>\n'
                mask_length = len(self.tokenizer(prompt, verbose=False)['input_ids'])

                # 拼接 prompt，计算 ice+test input 掩码长度
                prompt += current_patient
                test_length = len(self.tokenizer(prompt, verbose=False)['input_ids'])
                
                ice_mask_lengths.append(mask_length)
                test_mask_lengths.append(test_length)
                prompt_list.append(prompt)

            ## 2. 计算所有候选样例的条件熵
            print('计算该病人样例的交叉熵：')
            logits_list = []
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
            with torch.no_grad():
                for ice_idx, prompt in enumerate(prompt_list):
                    input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']  # 分词 torch.Size([1, 1809]
                    logits = self.inference_model(input_ids).logits  # 前向传播 torch.Size([1, 1809, 152064])

                    logits = logits[..., :-1, :].contiguous()  # (torch.Size([1, 1808, 152064]), torch.Size([1, 1808]))
                    input_ids = input_ids[..., 1:].contiguous()
                    
                    
                    logits = logits.view(-1, logits.size(-1))
                    loss = loss_fct(logits, input_ids.view(-1)).view(input_ids.size())  # 得到整个 prompt 损失

                    mask = torch.zeros_like(input_ids)
                    for i in range(ice_mask_lengths[ice_idx]-1, test_mask_lengths[ice_idx]-1):
                        mask[0][i] = 1  # 得到掩码
                    
                    loss = loss * mask
                    logits_list.append(torch.sum(loss, 1))

                    del input_ids, logits, loss, mask  # 显示清除显存
                    torch.cuda.empty_cache()

            ## 3. 将条件熵 进行排序, 获取 top3 序号
            _, indices = torch.topk(torch.tensor([t.item() for t in logits_list]), k=ice_num, largest=False)


            res_idx_list.append([self.candidate_ice_idx_list[p_idx][i] for i in indices.tolist()])  # 将当前病人的 检索结果返回
            with open('cone_idx.json', "w") as file:
                json.dump(res_idx_list, file, indent=4)     

        return res_idx_list


    def retrive(self):
        return self.cone_retrive()



if __name__ == '__main__':
    pass