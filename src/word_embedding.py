import numpy as np
import scipy.io

# load word vectors of ModelNet10
set_index = np.int16([1, 2, 8, 12, 14, 22, 23, 30, 33, 35])

wordvector = scipy.io.loadmat('../data/ModelNet40_w2v')
glovevector = scipy.io.loadmat('../data/ModelNet40_glove')

w2v = wordvector['word']
w2v_set = w2v[set_index, :]

glove = glovevector['word']
glove_set = glove[set_index, :]

print(glove_set)
print(glove_set.shape)
