from getFontDiff import get_font_diff
import sys
sys.path.append('..')
from build_img_datasets import get_words
from modelPredictions import genPredictImg
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from math import log10, sqrt
import keras.backend as K
from numpy import array, average, linalg, dot, prod
from functools import reduce

def PSNR(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
    # max_pixel = 8.0
    # return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1)))) / 2.303

def SSIM(image1, image2):
    score = ssim(array(image1),array(image2),multichannel=True)
    return score

def cosin_Similarity(filepath1, filepath2):

    def image_similarity_vectors_via_numpy(image1, image2):

        image1 = get_thumbnail(image1)
        image2 = get_thumbnail(image2)

        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # If we did not resize the images to be equal, we would get an error here
        # ValueError: matrices are not aligned
        res = dot(a / a_norm, b / b_norm)
        return res

    def get_thumbnail(image, size=(128,128), greyscale=False):
        #get a smaller version of the image - makes comparison much faster/easier
        image = image.resize(size, Image.ANTIALIAS)
        if greyscale:
            #convert image to greyscale
            image = image.convert('L')
        return image

    similarity = image_similarity_vectors_via_numpy(image1, image2)
    return similarity

def Euclidean_distance(image1, image2):
    norml2 = linalg.norm(array(image1)-array(image2)) / prod(array(image1).shape)
    return norml2

if __name__ == '__main__':
    existWord, notExistWord = [], []
    diff = get_font_diff('kaiu','HanyiSentyBubbleTea')
    words = get_words('../datasets/Words/words.csv')
    for word in words:
        if word not in diff:
            existWord.append(word)
        else:
            notExistWord.append(word)

    # print(len(existWord),len(notExistWord))
    genPredictImg(existWord, mode = 'single')

    # word = '永'
    kaiu2HanyiSentyBubbleTea_MSE, kaiu2HanyiSentyBubbleTea_PSNR = [], []
    # kaiu2HanyiSentyBubbleTea_SSIM, kaiu2HanyiSentyBubbleTea_cosin, kaiu2HanyiSentyBubbleTea_Euclidean = [], [], []
    for i in range(len(existWord)):
        filepath1 = 'PredictRes/095000/single/' + existWord[i] + '.png'
        filepath2 = '../datasets/HanyiSentyBubbleTea/' + existWord[i] + '.png'
        with Image.open(filepath1) as image1, Image.open(filepath2) as image2:
            # kaiu2HanyiSentyBubbleTea_MSE.append(mse(array(image2).flatten(),array(image1).flatten()))
            kaiu2HanyiSentyBubbleTea_PSNR.append(PSNR(array(image2),array(image1)))
            # kaiu2HanyiSentyBubbleTea_SSIM.append(SSIM(image1,image2))
            # kaiu2HanyiSentyBubbleTea_cosin.append(cosin_Similarity(image1,image2))
            # kaiu2HanyiSentyBubbleTea_Euclidean.append(Euclidean_distance(image1,image2))
        print('\r{} images calculated of {} images.'.format(i+1, len(existWord)),end='')
    print()
    # score_MSE = reduce(lambda x, y: x + y, kaiu2HanyiSentyBubbleTea_MSE) / len(kaiu2HanyiSentyBubbleTea_MSE)
    score_PSNR = reduce(lambda x, y: x + y, kaiu2HanyiSentyBubbleTea_PSNR) / len(kaiu2HanyiSentyBubbleTea_PSNR)
    # score_SSIM = reduce(lambda x, y: x + y, kaiu2HanyiSentyBubbleTea_SSIM) / len(kaiu2HanyiSentyBubbleTea_SSIM)
    # score_cosin = reduce(lambda x, y: x + y, kaiu2HanyiSentyBubbleTea_cosin) / len(kaiu2HanyiSentyBubbleTea_cosin)
    # score_Euclidean = reduce(lambda x, y: x + y, kaiu2HanyiSentyBubbleTea_Euclidean) / len(kaiu2HanyiSentyBubbleTea_Euclidean)

    # print('Average MSE: {}'.format(score_MSE))
    print('Average PSNR: {}'.format(score_PSNR))
    # print('Average SSIM: {}'.format(score_SSIM))
    # print('Average cosin Similarity: {}'.format(score_cosin))
    # print('Average Euclidean distance: {}'.format(score_Euclidean))
