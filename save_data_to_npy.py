import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_FontsData():
    fonts_set = ['kaiu', 'HanyiSentyBubbleTea']
    dirs = ['datasets/' + _ for _ in fonts_set]
    n_max = 1000
    n_train = 700
    n_test = n_max - n_train
    img_height, img_width,channels = 128, 128, 3
    X_train, Y_train = np.empty((n_train, img_height, img_width, channels), dtype='uint8'), np.empty((n_train, img_height, img_width, channels), dtype='uint8')
    X_test, Y_test = np.empty((n_test, img_height, img_width, channels), dtype='uint8'), np.empty((n_test, img_height, img_width, channels), dtype='uint8')
    n = 0
    for fileA, fileB in zip(os.listdir(dirs[0]),os.listdir(dirs[1])): # 4808 words
        if n >= n_max:
            break
        if fileA and fileB:
            img_kaiu = dirs[0] + '/' + fileA
            img_HSBT = dirs[1] + '/' + fileB
            if n < n_train:
                X_train[n,:,:] = cv2.imdecode(np.fromfile(img_kaiu, dtype=np.uint8), cv2.IMREAD_COLOR)
                Y_train[n] = cv2.imdecode(np.fromfile(img_HSBT, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                X_test[n-n_train,:,:] = cv2.imdecode(np.fromfile(img_kaiu, dtype=np.uint8), cv2.IMREAD_COLOR)
                Y_test[n-n_train] = cv2.imdecode(np.fromfile(img_HSBT, dtype=np.uint8), cv2.IMREAD_COLOR)
            n += 1
            print('\rLoading data... [{}{}{}] {:.2f}% ({}/{})'.format('='*int(n/(n_max*0.05)),'>','.'*((20-int(n/(n_max*0.05)))-1),n*100/n_max,n,n_max),end='')
    print('\rLoading data successed. [{}] {:.2f}% ({}/{})'.format('='*20,n*100/n_max,n,n_max))
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == '__main__':
    (X_train, Y_train) , (X_test, Y_test) = load_FontsData()

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i])
        # plt.gray()
        plt.axis('off')

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(Y_test[i])
        # plt.gray()
        plt.axis('off')
    # plt.savefig('./resultsImg/Figure_1.png')
    plt.show()

    Path('./datasets/npy/').mkdir(parents=True, exist_ok=True)
    np.save('./datasets/npy/X_train', X_train)
    np.save('./datasets/npy/Y_train', Y_train)
    np.save('./datasets/npy/X_test', X_test)
    np.save('./datasets/npy/Y_test', Y_test)
