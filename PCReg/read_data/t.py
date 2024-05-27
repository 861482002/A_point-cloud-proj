
import numpy as np
pc=[[1,2,5,],
    [3,1,5],
    [1,2,8]]
# mean = np.mean(pc, axis=0)
# print(mean)
# pc -= mean
# print(pc)
# m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
# print(np.power(pc, 2))
# print(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
#
# print(m)
# pc /= m
# print(pc)

import numpy as np

def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc, mean, m

def pc_denormalize(pc_normalized, mean, m):
    pc = pc_normalized * m
    pc += mean
    return pc

# 示例用法
original_pc = pc  # 生成一个随机的 10x3 的矩阵
normalized_pc, mean, m = pc_normalize(original_pc)
print("原始矩阵:")
print(original_pc)
print("归一化后的矩阵:")
print(normalized_pc)

# 假设你现在有了归一化后的矩阵，想要恢复原始矩阵
recovered_pc = pc_denormalize(normalized_pc, mean, m)
print("恢复后的矩阵:")
print(recovered_pc)
