from typing import Dict, Optional, Union, Hashable


class PromptTemplate:
    def __init__(self, task_name='readmission_prediction', inference_type = 'straight_forward'):
        self.task_name = task_name
        self.inference_type = inference_type

        # patient_prompt = identity_head + data_description_head + current_patient_information + task_head + inference_head
        self.identity_head = 'You are a medical expert with extensive knowledge in analyzing electronic health records (EHR).\n'
        self.data_description_head = 'The structured temporal electronic health records (EHR) data is identified by <Visit Sequence>. It includes the patient’s multiple visits to medical facilities, capturing diagnosed diseases, laboratory test information, and medication details.\n'


    def task_head_generate(self):
        '''
            get task specific prompt for llm
            telling llm what to do
        '''
        if self.task_name not in ['mortality_prediction', 'readmission_prediction']:
            raise ValueError(
                    f"task type {self.task_name} not implemented"
                )
        
        # for different task_name, get different task_head
        if self.task_name == 'mortality_prediction':
            # task_head = '<Question>: What is the likelihood that the patient will die within the next 14 days? Select one of the following options: A. Probability greater than 50%. B. Probability less than 50%.\n'
            task_head = '<Question>: will the patient die within the next 14 days? Select one of the following options: A. will die. B. will not die.\n'
        
        elif self.task_name == 'readmission_prediction': # 再入院预测
            task_head = '<Question>: Will the patient be readmitted to the hospital within two weeks? Select one of the following options: A. Yes. B. No.\n'
        return task_head
    

    def inference_head_generate(self):
        '''
            tell llm how to inference, like 'think step by step' or 'give answer straight_forward'
        '''
        if self.inference_type not in ['straight_forward', 'deep_seek_r1']:
            raise ValueError(
                    f"task type {self.inference_type} not implemented"
                )
        
        if self.inference_type == 'straight_forward':
            # straight forward give answer without any explannation
            inference_head = 'Important: Provide only the letter corresponding to your chosen answer. Do not include any explanation or additional text. Your answer is:'
        elif self.inference_type == 'deep_seek_r1':
            # deep_seek_r1 style 'think and answer'
            inference_head = 'Important: First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> A or B here </answer>. '
        return inference_head


    