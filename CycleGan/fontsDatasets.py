import numpy as np

def load_npy(datasets):
    return (np.load('../datasets/npy/' + datasets + '/X_train.npy'), np.load('../datasets/npy/' + datasets + '/Y_train.npy')), (np.load('../datasets/npy/' + datasets + '/X_test.npy'), np.load('../datasets/npy/' + datasets + '/Y_test.npy'))
