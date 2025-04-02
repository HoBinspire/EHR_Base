import random
import json
from typing import List, Union, Optional, Dict

import config
from datareader.datareader import DatasetReader


randSeed = config.RANDSEED

class RandomRetriever:
    def __init__(self,
                 dataset_path:str = config.DATASET_PATH,
                 input_columns: Union[List[str], str] = ['text'],
                 output_column: str = 'label',
                 ) -> None:
        
        # load data
        self.dataReader = DatasetReader(dataset_path, input_columns, output_column)

        # set config
        self.train_ds_size = len(self.dataReader.train_ds)
        random.seed(randSeed)

    def retrive(self, test_num = config.TEST_NUM, ice_num = config.ICE_NUM):
        """
        生成一个二维列表, 每个元素是一个包含3个随机整数的一维列表。
        存储 样例 index
        """
        results = []
        for _ in range(test_num):
            random_list = [random.randint(0, self.train_ds_size-1) for _ in range(ice_num)]
            results.append(random_list)

        with open(f'random{randSeed}_idx.json', 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        return results


if __name__ == '__main__':
    x = RandomRetriever()

    print(x.retrive())