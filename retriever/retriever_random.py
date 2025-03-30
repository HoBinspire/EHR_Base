import random
import json

random.seed(42)  # 设置 random 随机种子
train_size = 47180  # 训练集大小

def generate_random_lists(num_lists):
    """
    生成一个二维列表，每个元素是一个包含3个随机整数的一维列表。
    随机整数范围是0到4000。

    参数:
        num_lists (int): 二维列表中一维列表的数量。

    返回:
        list: 生成的二维列表。
    """
    result = []
    for _ in range(num_lists):
        # 生成一个包含3个随机整数的一维列表
        random_list = [random.randint(0, train_size-1) for _ in range(3)]
        # 将一维列表添加到二维列表中
        result.append(random_list)
    return result

# 示例用法
num_lists = 200  # 假设我们想要生成5个一维列表
random_2d_list = generate_random_lists(num_lists)
print(random_2d_list)


with open('rand_idx.json', 'w', encoding='utf-8') as file:
    json.dump(random_2d_list, file, ensure_ascii=False, indent=4)