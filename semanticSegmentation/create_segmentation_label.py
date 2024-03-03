import cv2
import numpy as np
import sys

# my module
from colorPrint import Cprint as cp

print = lambda *string: cp.cprint(" ".join(map(str, string)), color="cyan")


original_paths = {"test"       : "/workspace/osada_ws/text_dataset/ngc_docker/fold1/test.txt",
                  "train"      : "/workspace/osada_ws/text_dataset/ngc_docker/fold1/train.txt",
                  "validation" : "/workspace/osada_ws/text_dataset/ngc_docker/fold1/validation.txt"}

home_dir = "/workspace/Dataset/semanticSegmentation"
fold = 1

# buf = ""


# マスク画像の作成とテキストへの保存
def make_mask(target, path, label:int, index, size=(270, 270), is_resize:bool=False):
    img = cv2.imread(path)
    buf = ""
    if is_resize:
        buf = "_resized"
        img = cv2.resize(img, size)
    cv2.imwrite(f"{home_dir}/{target}{buf}/origin/origin_{index}.png", img)
    
    mask = np.zeros((*size, 1)) + (255*int(label))
    cv2.imwrite(f"{home_dir}/{target}{buf}/masked/masked_{index}.png", mask)
    
    with open(f"{home_dir}/text_dataset{buf}/fold{fold}/{target}.txt", mode="a") as f:
        f.write(f"{home_dir}/{target}{buf}/origin/origin_{index}.png {home_dir}/{target}{buf}/masked/masked_{index}.png\n")
    

# 画像の読み込み
def read_image(is_resize:bool=False):
    
    index = 0
    
    for target, path in original_paths.items():
        with open(path, mode="r", encoding="utf8") as f:
            data = list(map(lambda x : (x.split(" "))[0:2], f.read().split("\n")))
        length = len(data)
        print(f"- target : {target}, length : {length} -\n")
        
        for i, val in enumerate(data):
            if(len(val) < 2):
                length -= 1
                cp.cprint(f"[!] warning : No.{index+1}, this path is blank. corrected length : {length}", "orange")
                continue
            img, label = val
            make_mask(target, img, label, index, (256, 256), is_resize)
            # make_mask(target, img, label, index, (270, 270), is_resize)
            index += 1
            cp.cprint(f"\033[1Acompleted : {i+1}/{length}", "green")
    
    print("- finished ! -")
            

# データセットのテキストファイルの初期化
def file_init(filenames, buf):
    for name in filenames:
        with open(f"{home_dir}/text_dataset{buf}/fold{fold}/{name}.txt", mode="w"): pass
    print("- created new text datasets -")


if (__name__ == "__main__"):
    args = sys.argv
    is_resize = False; buf = ""
    if(len(args) > 1):
        if(args[1] == "resize"):
            is_resize = True; buf = "_resized_256"
            print("- image resizable -")
    is_create_file = bool(input(cp.colored("Create new text datasets? (y/n) : ", "cyan")) == "y")
    if is_create_file: file_init(original_paths.keys(), buf)
    read_image(is_resize)
