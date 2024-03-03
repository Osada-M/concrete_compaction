import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from PIL import Image

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm


DIR = "/workspace/fullframe/result/540x540"
TEXT = lambda fold: f"/workspace/Dataset/fullframe/text_dataset/fold{fold}/test.txt"
SAVE_DIR = "/workspace/visualization"



def circle(img, extend_const:int=50):
    """
    円
    """
    row, col, *_ = img.shape
    origin_x, origin_y = col/2, row/2
    radius = math.sqrt(col*row/(2*math.pi))
    
    for y in range(row):
        for x in range(col):
            if ((x-origin_x)**2 + (y-origin_y)**2 <= radius**2):
                img[y, x] += extend_const
    img = np.uint8(np.clip(img, 0, 255))
    
    return img


def half_slant(img, extend_const:int=50):
    """
    斜め
    """
    row, col, *_ = img.shape
    dydx = row / col
    for y in range(row):
        for x in range(col):
            if (y <= dydx * x):
                img[y, x] += extend_const
    img = np.uint8(np.clip(img, 0, 255))
    
    return img


def all_pixel(img, extend_const:int=50):
    """
    全部
    """
    row, col, *_ = img.shape
    img += extend_const
    img = np.uint8(np.clip(img, 0, 255))
    
    return img


def half_right(img, extend_const:int=50):
    """
    半分
    """
    row, col, *_ = img.shape
    img[:, col//2:] += extend_const
    img = np.uint8(np.clip(img, 0, 255))
    
    return img


def main():
    Utils.makedir(f"{SAVE_DIR}/luminance")
    
    with open(TEXT(1), mode="r") as f:
        line, *_ = f.readlines()
        img = np.array(Image.open(line.split(" ")[0]), dtype=np.int32)
        
        names = ["right", "all", "slant", "circle"]
        funcs = [half_right, all_pixel, half_slant, circle]
        
        for name, func in zip(names, funcs):
            cp.cprint(name, "cyan")
            for const in [100, 75, 50, 25, -25, -50, -75, -100]:
                copied = np.copy(img)
                buf = func(copied, const)
                extended = Image.fromarray(buf)
                extended.save(f"{SAVE_DIR}/luminance/{name}_{str(const).replace('-', 'in')}.png")
                cp.cprint(f" > {const}", "gray")


main()
