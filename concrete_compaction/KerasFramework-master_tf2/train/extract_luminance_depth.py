import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

## my modules
from MyUtils import Utils


ALL_TEXT = "/workspace/Dataset/fullframe/text_dataset/all.txt"
SAVE_DIR = "/workspace/visualization/luminance"
SKIP = 10


def extract_luminance():
    """
    照度の頻度分析
    """
    luminance = np.array([0]*256, dtype=np.int32)
    
    with open(ALL_TEXT, mode="r") as f:
        lines = f.readlines()
    length = len(lines); print()
    for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
        if i%SKIP or not len(line): continue
        
        img_path, *_ = line.split(" ")
        img = np.array(Image.open(img_path), dtype=np.uint8)
        
        luminance += np.int32(np.histogram(img, 256)[0])
        
        print(f"\033[1A{i+1} / {length}")
    
    return luminance


def save_data(luminance):
    """
    テキストへの書き込みとグラフの描画
    """
    save_dir = f"{SAVE_DIR}/histogram"
    Utils.makedir(save_dir)
    
    with open(f"{save_dir}/histogram.txt", mode="w") as f:
        f.write(" ".join(map(str, luminance)))
    
    plt.bar(range(256), luminance)
    plt.savefig(f"{save_dir}/histogram.png")
    


def main():
    
    luminance = extract_luminance()
    save_data(luminance)


main()
