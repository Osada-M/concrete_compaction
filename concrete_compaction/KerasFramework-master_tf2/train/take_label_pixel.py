from PIL import Image
import numpy as np

from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/fullframe/text_dataset"
# TEXT = "all_after.txt"
TEXT = "all_after_2class_rectified.txt"
CLASSES = [0, 128, 255]


def check():
    
    cp.cprint(f"\ntext : {TEXT}\n", "pink")
    with open(f"{DIR}/{TEXT}") as f:
        lines = f.readlines()
    length = len(lines)
    
    label = [0]*len(CLASSES)
    
    for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines[::10])):
        val = line.split(" ")
        img, msk, *_ = val
        
        msk = np.array(Image.open(msk), dtype=np.uint8)
        for j, c in enumerate(CLASSES):
            label[j] += np.sum(msk == c)
        
        cp.cprint(f"\033[1A{i+1} / {length} , {label}{' '*10}", "orange")

    cp.cprint(label, "cyan")
    

def main():
    
    check()


main()
