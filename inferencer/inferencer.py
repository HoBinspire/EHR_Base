import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from template.prompt_template import PromptTemplate
from datareader import DatasetReader
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'



class Inferencer:
    def __init__(self,
                 model_path = '/data/lhb/huggingface/model/tokenizer/Qwen2.5-7B-instruct',
                 task_type = 'mortality_prediction',
                 inference_type = 'straight_forward'
                 ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                    use_fast=False,
                                                    trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained( model_path,
                                                    device_map="auto",
                                                    trust_remote_code=True)
        self.model.eval()
        self.task_type = task_type
        self.inference_type = inference_type

        self.prompt_template = PromptTemplate()
        self.prompt_template.inference_type = 'deep_seek_r1'
        self.dataloader = DatasetReader()
        pass

    def generate_direct_prompt(self):
        """
        生成提问 prompt.
        """
        with open('/data/lhb/test-openicl-0.1.8/openicl/ehrBase_result/cone_ice_idx.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        prompt_list = []
        for p_id, ice_idxs in enumerate(data):
            ice = ""
            prompt = ''
            for _, idx in enumerate(ice_idxs):
                ice += f'Exmaple {_+1}:\n<Visit Sequence>' + self.dataloader.train_ds[idx]['text'] + '</Visit Sequence>\n<label>' + self.dataloader.train_ds[idx]['label'] + '</label>\n'
            
            prompt = self.prompt_template.identity_head + self.prompt_template.data_description_head + 'Here are some examples for reference:\n' + ice + "current_patient_information:\n" + self.dataloader.test_ds[p_id]['text'] + '\n'+ self.prompt_template.task_head_generate() + self.prompt_template.inference_head_generate()
            
            prompt_list.append({
                'id': p_id,
                'prompt': prompt
            })
            with open('cone_prompt_deepseek_r1.json', 'w', encoding='utf-8') as file:
                json.dump(prompt_list, file, ensure_ascii=False, indent=4)

    def generete_deepseek_r1_prompt(self):
        with open('/data/lhb/test-openicl-0.1.8/openicl/ehrBase_result/topk/topk_prompt_deepseek_r1.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        prompt_list = []
        for item in tqdm(data):
            id = item['id']

            if int(id) < 101:
                continue

            input, response = self.get_response(item['prompt'])

            if "<answer>" not in response:
                if response.endswith("<|im_end|>"):
                    response = response.rstrip("<|im_end|>").strip()

                response = response + "<answer> "
            else:
                response = re.split(r"(<answer>)", response)[0] + "<answer>"

            prompt_list.append({
                'id': id,
                'prompt': input + response
            })
            with open('topk_prompt_deepseek_r1_with_response.json', 'w', encoding='utf-8') as file:
                json.dump(prompt_list, file, ensure_ascii=False, indent=4)

            
    def inference(self):
        """
        用读取 prompt，进行单个 token 预测
        """
        with open('/data/lhb/test-openicl-0.1.8/openicl/topk_prompt_deepseek_r1_with_response.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        answers = []
        for item in tqdm(data):
            input_ids = self.tokenizer(item['prompt'], return_tensors='pt')['input_ids']
            with torch.no_grad():
                logits = self.model(input_ids).logits

            next_token_logits = logits[:, -1, :]  # 只取最后一个 token's 的 logits

            candidate_ans_logits_list = []
            for candidate_ans in ['A', 'B']:
                candidate_ans_ids = self.tokenizer.convert_tokens_to_ids(candidate_ans)
                candidate_ans_logits = next_token_logits[0, candidate_ans_ids]  # 读取 A, B 的 logits
                candidate_ans_logits_list.append(float(candidate_ans_logits))
            
            # prediction = F.softmax(torch.tensor(candidate_ans_logits_list), dim=0).tolist()  # 归一化 logits 为概率
            prediction = [x/sum(candidate_ans_logits_list) for x in candidate_ans_logits_list]


            answers.append({
                'id': item['id'],
                'logits': candidate_ans_logits_list,
                'prediction': prediction,
                'label': self.dataloader.test_ds[item['id']]['label']
            })
            with open('/data/lhb/test-openicl-0.1.8/openicl/topk_res_deepseek_r1.json', 'w', encoding='utf-8') as file:
                json.dump(answers, file, ensure_ascii=False, indent=4)

            del input_ids, logits, next_token_logits
            torch.cuda.empty_cache()

    def get_response(self, prompt):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}, ]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True,)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens= 2048,
                temperature = 0.7,   
                top_p = 0.8, 
                top_k = 20,              
                do_sample=True, 
                repetition_penalty = 1.05,           
                eos_token_id=self.tokenizer.eos_token_id)
        
        
        # only return new generated token by llm
        input_ids = inputs.input_ids
        prompt_in_chat_template = response = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response_in_chat_template = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        return response, response_in_chat_template

        

if __name__ == '__main__':
    exm = Inferencer()
    # exm.generete_deepseek_r1_prompt()
    exm.inference()
