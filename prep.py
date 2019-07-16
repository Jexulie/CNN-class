from PIL import Image
from os import listdir, remove
from os.path import splitext, split
import re

paths = [
    'datasets/apple/',
    'datasets/huawei/',
    'datasets/samsung/'
]

def convertToPng(paths):
    target = '.png'

    for path in paths:
        for file in listdir(path):
            name, ext = splitext(file)
            try:
                if ext != target:
                    im = Image.open(path+name + ext)
                    im.save(path+name + target)
                    remove(path+name+ext)
                    print(f'from {path}, Converted {name + ext} to {name + target} and removed ...')
            except OSError as err:
                print("Conversion Error ",err)
    
    print("All Done")

def seperateDirs(path):
    # for linux systems
    return re.split(r'/', path)

def resize(paths):
    size = 32, 32

    for path in paths:
        for file in listdir(path):
            root, first, second, _ = seperateDirs(path)
            src = path+file
            target = f'{root}/{first}/cropped/{second}/{file}' 
            try:
                im = Image.open(src)
                im.thumbnail(size, Image.ANTIALIAS)
                im.save(target,"png")
                print(f'from {src}, converted and saved to {target}')
            except OSError as err:
                print("Conversion Error ",err)
    
    print("All Done")

convertToPng(paths)
# resize(paths)
