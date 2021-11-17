import os

def get_font_diff(fontA = 'kaiu', fontB = 'HanyiSentyBubbleTea', pt = False):
    pathA = '../datasets/' + fontA
    pathB = '../datasets/' + fontB
    diff = list(set(os.listdir(pathA)) - set(os.listdir(pathB)))
    if not diff:
        print('No diff.')
    elif pt == True:
        print('Total:',len(diff))
        print(diff)
    diff = [ x[0] for x in diff]
    return diff

if __name__ == '__main__':
    diff = get_font_diff(pt=True)
