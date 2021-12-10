import sys
sys.path.append('..')
from build_img_datasets import get_words
from modelPredictions import getTargetUniMap, genPredictImg

if __name__ == '__main__':
    testWords = input("Input Chinese characters:")
    trainData = get_words('../datasets/Words/words.csv')
    words = [ j for i in testWords.split() for j in i]
    notExistWord = []
    for word in words:
        if word not in trainData:
            notExistWord.append(word)
    if notExistWord:
        print('The character:', end='')
        for word in notExistWord:
            print(word, end='')
        print(' is not in the 4808 train samples.')
        exit(0)
    mode = input('mode(pair, single):')
    if not(mode == 'pair' or mode == 'single'):
        print('Input error.')
        exit(0)
    tar_dir = input('target directory(default or specify):')
    replace = input('replace?(y/n):')
    if replace == 'y':
        replace = True
    elif replace == 'n':
        replace = False
    else:
        print('Input error.')
        exit(0)

    # print(words, mode, replace)
    genPredictImg(words, mode = mode, tar_dir = tar_dir, replace = replace)
