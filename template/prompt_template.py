from typing import Dict, Optional, Union, Hashable


class PromptTemplate:
    def __init__(self):
        self.task_name_set = set(['mortality_prediction', 'readmission_prediction'])
        self.inference_type_set = set(['direct', 'cot'])
        self.dataset_set = set(['MIMICIII'])

        """
            embedding_prompt = task_head + current_patient_EHR_info
            inference_prompt = identity_head + data_description_head + in-context examples + current_patient_info + task_head + inference_head
        """
        
    def identity_head_generate(self):
        identity_head = 'You are a medical expert with extensive knowledge in analyzing electronic health records (EHR).\n'
        return identity_head
    
    def data_description_head_generate(self, dataset):
        if dataset not in self.dataset_set:
            raise ValueError(f"参数 'dataset' 错误：不存在这个数据集的描述信息 '{dataset}'。请检查输入的数据集名称是否正确。")

        if dataset == "MIMICIII":
            data_description_head = 'The structured temporal electronic health records (EHR) data is identified by <Visit Sequence>. It includes the patient’s multiple visits to medical facilities, capturing diagnosed diseases, laboratory test information, and medication details.\n'
        else: # Todo: 补充其他数据集
            pass

        return data_description_head

    def task_head_generate(self, task_name):
        '''
            get task specific prompt for llm
            telling llm what to do
        '''
        if task_name not in self.task_name_set:
            raise ValueError(f"task type {self.task_name} not implemented")
        
        if task_name == 'mortality_prediction':
            # task_head = '<Question>: What is the likelihood that the patient will die within the next 14 days? Select one of the following options: A. Probability greater than 50%. B. Probability less than 50%.\n'
            task_head = '<Question>: will the patient die within the next 14 days? Select one of the following options: A. will die. B. will not die.\n'
        
        elif task_name == 'readmission_prediction': # 再入院预测
            task_head = '<Question>: Will the patient be readmitted to the hospital within two weeks? Select one of the following options: A. Yes. B. No.\n'
        return task_head
    

    def inference_head_generate(self, inference_type):
        '''
            tell llm how to inference, like 'think step by step' or 'give answer straight_forward'
        '''
        if inference_type not in self.inference_type_set:
            raise ValueError(f"task type {inference_type} not implemented")
        
        if inference_type == 'direct':
            # straight forward give answer without any explannation
            inference_head = 'Important: Provide only the letter corresponding to your chosen answer. Do not include any explanation or additional text. Your answer is:'
        elif inference_type == 'cot':
            # deep_seek_r1 style 'think and answer'
            inference_head = 'Important: First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> A or B here </answer>.'
        return inference_head


    