import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
import csv
from fontTools.ttLib import TTFont
import os

def addfont_to_plt():
    font_dirs = ['./fonts/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

def set_params():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.figsize"] = (1.28, 1.28)

def draw_grid():
    plt.axvline(color='k',linestyle='solid',alpha=0.5)
    plt.axvline(5,color='k',linestyle='solid',alpha=0.5)
    plt.axvline(10,color='k',linestyle='solid',alpha=0.5)
    plt.axhline(color='k',linestyle='solid',alpha=0.5)
    plt.axhline(5,color='k',linestyle='solid',alpha=0.5)
    plt.axhline(10,color='k',linestyle='solid',alpha=0.5)

def save_data(fonts, words):
    for i in range(len(fonts)):
        plt.rcParams['font.family'] = fonts[i]
        fontType = os.path.join('fonts', fonts[i] + '.ttf')
        font = TTFont(fontType)
        uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
        # draw_grid()
        Path('./datasets/' + fonts[i]).mkdir(parents=True, exist_ok=True)
        for j in words:
            if ord(j) in uniMap.keys():
                plt.clf()
                plt.axis('off')
                image = plt.text(.5, .5, j, fontsize=50, ha='center', va='center') # if draw grid (x = 5, y = 5), else (x = .5, y = .5)
                plt.savefig('./datasets/' + fonts[i] + '/' + j + '.png')
                # image = plt.show()
                # break

def get_words(path='./datasets/Words/words.csv'):
    words = []
    with open(path, newline='', encoding='utf-8') as csvfile: # 4808 個常用字
      rows = csv.reader(csvfile)
      for row in rows:
          if(row):
              words.append(row[0])
    words[0] = '一'
    return words


if __name__ == '__main__':
    # fonts = ['kaiu', 'SimHei']
    fonts = ['HanyiSentyBubbleTea']
    addfont_to_plt()
    set_params()
    save_data(fonts,get_words())
