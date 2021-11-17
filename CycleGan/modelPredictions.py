from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import random
from fontTools.ttLib import TTFont
from getFontDiff import get_font_diff
sys.path.append('..')
from build_img_datasets import set_params, get_words

def getResShort(words, model_serial, mode):
    path = './PredictRes/' + model_serial + '/' + mode
    diff = words
    if os.path.exists(path):
        curRes = [ i[0] for i in os.listdir(path)]
        diff = list(set(words) - set(curRes))
    return diff

def getTargetUniMap(fontName):
    fontType = os.path.join('..', 'fonts', fontName + '.ttf')
    font = TTFont(fontType)
    uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
    return uniMap

def genPredictImg(words, model_serial = '095000', mode = 'pair', replace = False):
    serial_num = int(model_serial[0:3])
    # print(serial_num)
    if not (serial_num <= 100 and serial_num % 5 == 0):
        print('Illegal model serial number.')
        return
    model = load_model('./saved_model/kaiu2HanyiSentyBubbleTea/g_model_AtoB_' + model_serial + '.h5', custom_objects={'InstanceNormalization':InstanceNormalization})
    # model.summary()

    if replace:
        gen_target = words
    else:
        gen_target = getResShort(words, model_serial, mode)
    # print(gen_target)
    if gen_target:
        uniMap = getTargetUniMap('HanyiSentyBubbleTea')
        img_height, img_width, channels = 128, 128, 3
        word = np.empty((len(gen_target),img_height, img_width, channels), dtype='uint8')
        tar_word = np.empty((len(gen_target),img_height, img_width, channels), dtype='uint8')
        for i in range(len(gen_target)):
            word[i] = cv2.imdecode(np.fromfile('../datasets/kaiu/' + gen_target[i] + '.png', dtype=np.uint8), cv2.IMREAD_COLOR)
            if ord(gen_target[i]) in uniMap.keys():
                tar_word[i] = cv2.imdecode(np.fromfile('../datasets/HanyiSentyBubbleTea/' + gen_target[i] + '.png', dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                tar_word[i] = np.ones((img_height, img_width, channels), dtype='uint8') * 255
        word = (word - 127.5) / 127.5
        # print(word.shape)
        res = model.predict(word)


        if mode == 'single':
            set_params()
            Path('./PredictRes/' + model_serial + '/single').mkdir(parents=True, exist_ok=True)
            for i in range(len(gen_target)):
                plt.clf()
                plt.figure()

                plt.imshow((res[i] * 255).astype(np.uint8))
                plt.axis('off')

                plt.savefig('./PredictRes/' + model_serial + '/single/' + gen_target[i] + '.png')
                print('\r{} completed of {} words.'.format(i+1, len(gen_target)),end='')

                # plt.show()
        else:
            Path('./PredictRes/' + model_serial + '/pair').mkdir(parents=True, exist_ok=True)
            for i in range(len(gen_target)):
                plt.clf()
                plt.figure()

                plt.subplot(1,2,1)
                plt.imshow((tar_word[i]).astype(np.uint8))
                plt.axis('off')

                plt.subplot(1,2,2)
                plt.imshow((res[i] * 255).astype(np.uint8))
                plt.axis('off')

                plt.savefig('./PredictRes/' + model_serial + '/pair/' + gen_target[i] + '.png')
                print('\r{} completed of {} words.'.format(i+1, len(gen_target)),end='')

                # plt.show()
        print('\nCompleted.')
    else:
        print('No target to be generate.')
if __name__ == '__main__':
    # testWord = ['噹','永']
    existWord, notExistWord = [], []
    diff = get_font_diff('kaiu','HanyiSentyBubbleTea')
    words = get_words('../datasets/Words/words.csv')
    for word in words:
        if word not in diff:
            existWord.append(word)
        else:
            notExistWord.append(word)
    testWord = [ existWord[i] for i in random.sample(range(0,len(existWord)), 50)]
    # print(len(existWord),len(notExistWord))
    genPredictImg(testWord)
    # genPredictImg(testWord, mode='single')
