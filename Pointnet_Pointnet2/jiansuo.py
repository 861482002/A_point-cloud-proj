import numpy as np
# d = 1024                           # dimension
# nb = 10                      # database size
# nq = 1                       # nb of queries
# np.random.seed(11334)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.
# print(xb.shape)
# print(xq)
xb1 = np.loadtxt('array_data10_512_2.txt')
xq1 = np.loadtxt('array_data12_512_2.txt')
xb = xb1.reshape((1,512))
xq = xq1.reshape((1,512))

print(xb)
print(xq)
# xb = xq
# print(xb1.reshape(1,-1))
# print(xq1)
import faiss                   # make faiss available
# index = faiss.IndexFlatL2(d)   # build the index
#
# index.add(xb)                  # add vectors to the index


dim, measure = 512, faiss.METRIC_L2
param = 'LSH'
index = faiss.index_factory(dim, param, measure)
# index = faiss.IndexFlatL2(dim)

print(index.is_trained)                          # 此时输出为True
index.add(xb)
import math

k = 1                          # we want to see 4 nearest neighbors
D, I = index.search(xq, k)     # actual search
print(D)
print(I)
# print(I[0][0])                   # neighbors of the 5 first queries
# print(D[0][0])                  # neighbors of the 5 last queries
# top_1_list = I[0][0]
# print(top_1_list)
AAA = D[0][0]/100
# AAA = D[0][0]
# te = 1-(math.exp(-AAA))
print(1 / (1 + np.log(AAA + 1)))
# print(1-te)
print(1-(AAA/(AAA+1)))
print(1/(1+AAA))