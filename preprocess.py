import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc


pd.set_option('display.max_columns', None)#显示所有的列
pd.set_option('display.max_rows', None)#显示所有的行
pd.set_option('expand_frame_repr', False)#不自动换行
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=2)

# dataset='./datasets/yelp'
dataset = './datasets/movielens'
device = 'cuda:7'
print(dataset)
df_train = pd.read_csv(dataset + r'/train_sparse.csv')

# yelp
# user,item=25677,25815
# ML-1M
user, item = 6040, 3952
# citeulike
# user,item=5551,16981
# pinterest
# user,item=37501,9836
# gowalla
# user,item=29858,40981


alpha = 1

# interaction matrix
rate_matrix = torch.zeros(user, item).to(device)

for row in df_train.itertuples():
    rate_matrix[row[1], row[2]] = 1

# save interaction matrix
np.save(dataset + r'/rate_sparse.npy', rate_matrix.cpu().numpy())

D_u = rate_matrix.sum(1) + alpha
D_i = rate_matrix.sum(0) + alpha

for i in range(user):
    if D_u[i] != 0:
        D_u[i] = 1 / D_u[i].sqrt()

for i in range(item):
    if D_i[i] != 0:
        D_i[i] = 1 / D_i[i].sqrt()

# \tilde{R}
rate_matrix = D_u.unsqueeze(1) * rate_matrix * D_i

# free space
del D_u, D_i
gc.collect()
torch.cuda.empty_cache()

'''
q:the number of singular vectors in descending order.
to make the calculated singular value/vectors more accurate,
q shuld be (slightly) larger than K.
'''

print('start!')
start = time.time()
q = 400
niter = 30
print(f'q {q} niter {niter}')
U, value, V = torch.svd_lowrank(rate_matrix, q=q, niter=niter)
# U, value, V = torch.svd(rate_matrix)
end = time.time()
print(U.shape, value.shape, V.shape)
print('processing time is %f' % (end - start))
print('singular value range %f ~ %f' % (value.min(), value.max()))

# start = time.time()
# print('the rank of matrix ', torch.linalg.matrix_rank(rate_matrix))
# print(time.time() - start)


np.save(dataset + r'/svd_u.npy', U.cpu().numpy())
np.save(dataset + r'/svd_v.npy', V.cpu().numpy())
np.save(dataset + r'/svd_value.npy', value.cpu().numpy())
