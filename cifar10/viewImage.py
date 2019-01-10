import itertools
import matplotlib.pyplot as plt
import numpy as np

# data_dir = "/home/users/gmertz/cifar-10-batches-py/"
# save_dir = "/home/users/gmertz/modeldir/"
data_dir = "C:/Users/Test/code/AI/projects/deeplearning/cifar-10-batches-py/"
save_dir = "C:/Users/Test/code/AI/projects/deeplearning/modeldir/"

def unpickle(file):
	'''adapted from the CIFAR page: https://www.cs.toronto.edu/~kriz/cifar.html '''
	import pickle
	with open(file, 'rb') as fo:
		return pickle.load(fo, encoding='bytes')

train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in range(1,5)]
valid = unpickle(data_dir + 'data_batch_5')
X_train = np.concatenate([t[b'data'] for t in train],axis=0)
y_train = np.array(list(itertools.chain(*[t[b'labels']for t in train])))
X_valid = valid[b'data']
y_valid = np.array(valid[b'labels'])
labels = unpickle(data_dir + 'batches.meta')[b'label_names']


def viewImage(X,y,i):
	img = X_train.reshape(-1,3,32,32)[i].T
	plt.imshow(img)
	plt.show()
	print(labels[y[i]])
