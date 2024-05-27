# import torch
import os
import os.path as osp

import faiss
import numpy as np

# np.random.seed(12)

# a = np.random.rand(512)
# a2 = np.random.rand(512)
# b = np.random.rand(512)

# a_linalg = np.linalg.norm(a,ord=2)
# b_linalg = np.linalg.norm(b,ord=2)

# a_normaled = a / a_linalg
# b_normaled = b / b_linalg

# print(a_linalg,b_linalg)

# dot_a_b = a.dot(b)
# similarity = dot_a_b / (a_linalg * b_linalg)
# print(f'使用np计算的余弦相似性:{similarity}')

# a_torch = torch.from_numpy(a)
# b_torch = torch.from_numpy(b)

# torch_similarity = F.cosine_similarity(a_torch[None,...],b_torch[None,...])
# print(f'使用torch计算的余弦相似性:{torch_similarity}')
'''
PYTHONUNBUFFERED python unbuffered
'''
def feature_match_test():
	pcl_files = os.listdir('./classification_dir_512')
	out_dir = './searched_dir_512_HNSW32/'
	for pcl_file in pcl_files:
		# 读取当前需要处理的点云
		cur_pcl = np.loadtxt('./classification_dir_512/' + pcl_file)
		dim = cur_pcl.shape[0]
		cur_pcl_reshaped = cur_pcl.reshape(-1, dim)
		# 归一化当前点云文件
		cur_pcl_norm = np.linalg.norm(cur_pcl_reshaped, ord=2, axis=-1)
		cur_pcl_reshaped = cur_pcl_reshaped / cur_pcl_norm
		# 定义查询集
		idx = faiss.index_factory(dim,'HNSW32',faiss.METRIC_L2)
		# idx = faiss.IndexFlatL2(dim)

		# 读取非当前的点云文件，并stack成张量
		other_pcl = [pcl for pcl in pcl_files if pcl != pcl_file]
		unstacked_otcher_pcl = []
		for other_pcl_file in other_pcl:
			tmp_pcl = np.loadtxt('./classification_dir_512/' + other_pcl_file).reshape(-1, dim)
			# 进行归一化
			tmp_pcl_norm = np.linalg.norm(tmp_pcl, ord=2, axis=-1)
			tmp_pcl = tmp_pcl / tmp_pcl_norm
			unstacked_otcher_pcl.append(tmp_pcl)
		stacked_other_pcl = np.concatenate(unstacked_otcher_pcl, axis=0)

		# 构建查询集
		idx.add(stacked_other_pcl)

		# 使用查询向量去查询集里面查询
		D, I = idx.search(cur_pcl_reshaped, k=5)
		# Distances, Idxs = 1 - D.squeeze(), I.squeeze()
		Distances, Idxs = D.squeeze(), I.squeeze()
		D_str = np.array2string(Distances, separator=',')
		I_str = np.array2string(Idxs, separator=',')
		if not osp.exists(out_dir):
			os.makedirs(out_dir)
		with open(osp.join(out_dir, pcl_file.split('.')[0] + '.txt'), 'a') as fp:
			fp.write(pcl_file + '\n')
			fp.write(f'相似性：{D_str}' + '\n')
			fp.write(I_str + '\n')
			ranked_filename = [other_pcl[Idx] + '\t' for Idx in Idxs]
			fp.writelines(ranked_filename)
	# print(D,I.shape)

def feature_match(query:np.ndarray,query_dir: str, Index:np.ndarray):
	'''

	Parameters
	----------
	query       查询集             shape[batch ,512/256]
	query_dir   查询集的文件名
	Index       用于查询的向量      shape[n     ,512/256]

	Returns
	-------

	'''
	file_names = os.listdir(query_dir)

	batch,dim = query.shape

	# idx = faiss.index_factory(dim, 'LSH', faiss.METRIC_L2)
	idx = faiss.IndexFlatL2(dim)

	# 构建查询集
	idx.add(query)

	# 使用查询向量去查询集里面查询
	# D: shape[n,batch]     I: shape[n,batch]
	D, I = idx.search(Index, k=2)


	# Distances, Idxs = 1 - D.squeeze(), I.squeeze()
	Distances, Idxs = D.squeeze(), I.squeeze()
	matched_file = file_names[Idxs[-1]]
	return matched_file



def test_np_linalg():
	np.random.seed(12)
	input = np.random.choice([0,1,2,3,4,5],(3,3))
	print(input)
	output = np.linalg.norm(input, ord=2, axis=0)
	print(output)
if __name__ == '__main__':
	feature_match()
