import cv2
import numpy as np

from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/semanticSegmentation/masked"
CLASSES = [0, 64, 128, 255]
SIZES = [224, 256, 270, 288]


def main():
    
    for i, c in enumerate(CLASSES):
        for size in SIZES:
            image = np.zeros((size, size)) + c
            cv2.imwrite(f"{DIR}/class{i}_{size}.png", image)
            cp.cprint(f"class : {i}, size : {size}", "orange")
        
    cp.cprint(f"completed", "cyan")


main()
