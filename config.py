###-- randseed --###
RANDSEED = 42
TEST_NUM = 200

CONDIDATE_NUM = 10
ICE_NUM = 3

###-- dataset_path --###
DATASET_PATH = '/data/lhb/huggingface/dataset/mimic3_1.4_readmission_prediction'

###-- train_faiss_index --###
TRAIN_FAISS_INDEX_PATH = '/data/lhb/test-openicl-0.1.8/EHR_Base/faiss_index/faiss_index_bge_train_readmission.index'

###-- model_path --###
LLM_PATH = '/data/lhb/huggingface/model/tokenizer/Qwen2.5-7B-instruct'

###-- embed_model_path --###
EMBED_MODEL_PATH = '/data/lhb/huggingface/model/tokenizer/Qwen2.5-7B-instruct'
# EMBED_MODEL_PATH = '/data/lhb/huggingface/model/sentence_embedding/bge-m3'

###-- CUDA --### 
CUDA_IDX = '0,3,6,7'

###-- task / inference type --###
TASK_SET = set(['mortality_prediction', 'readmission_prediction'])
TASK_TPYE = 'readmission_prediction'

INFERENCE_TYPE_SET = set(['direct', 'cot'])
INFERENCE_TYPE = 'direct'