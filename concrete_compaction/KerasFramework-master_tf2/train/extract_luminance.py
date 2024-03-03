import resource
import matplotlib.pyplot as plt
import numpy as np

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from ExplainableFuncs import ExplainableFuncs
from GifMaker import GifMaker


## ================ config ================


LOAD_DIR = "/workspace/explain/flow/"
LOAD_ID = "seg-grad-cam_unet_20220319_AutoLearning_fold3_540x540"
## LOAD_NUMBER : [a, b)
LOAD_NUMBER = [0, 9]
## LOAD_RANGE : [a, b)
LOAD_RANGE = [0, 19]

SIZE = 33
HEATMAP_SIZE = 100


## ========================================


path = lambda target : f"{LOAD_DIR}/{LOAD_ID}/{LOAD_ID}_{target}.txt"


def extract(resourcepath):
    """
    @機能：
    @引数：
    @戻値：
    """
    with open(resourcepath, mode="r") as f:
        lines = list(map(lambda x: list(map(float, x.split(" "))), f.readlines()))
    lines = np.array(lines)
    lines = np.resize(lines, (SIZE, SIZE))
    
    output = []
    for l in lines:
        output += list(l)

    return output


def make_png(load_number):
    """
    @機能：
    @引数：
    @戻値：
    """
    before = [None]*LOAD_RANGE[-1]
    just = [None]*LOAD_RANGE[-1]
    images = [None]*LOAD_RANGE[-1]
    for i in range(*LOAD_RANGE):
        for label in [0, 1]:
            target = f"{i:04d}_{load_number:02d}_{label:02d}"
            if label:
                before[i] = extract(path(target))
            else:
                just[i] = extract(path(target))
        images[i] = extract(path(f"{i:04d}_image"))
    
    before = np.array(before)
    just = np.array(just)
    images = np.array(images)
    
    before_accum = []
    just_accum = []
    images_accum = []
    
    for b, j, i in zip(before, just, images):
        for index, _ in enumerate(i):
            before_accum.append(b[index])
            just_accum.append(j[index])
            images_accum.append(i[index])
    
    
    before_heat = [[0]*HEATMAP_SIZE for _ in range(HEATMAP_SIZE)]
    for i, b in zip(images_accum, before_accum):
        if not b or np.isnan(b): continue
        i *= HEATMAP_SIZE-1; b *= HEATMAP_SIZE-1
        i = int(i); b = int(b)
        before_heat[HEATMAP_SIZE-1-b][i] += 1
    before_heat = np.array(before_heat)
    before_heat_accum = np.sum(before_heat, axis=1)
    before_heat_accum_decoded = []
    for i, accum in enumerate(before_heat_accum):
        before_heat_accum_decoded += [i]*accum
    before_quartile = np.percentile(before_heat_accum_decoded, [50, 25, 75])/HEATMAP_SIZE
    before_quartile = 1 - before_quartile
    
    just_heat = [[0]*HEATMAP_SIZE for _ in range(HEATMAP_SIZE)]
    for i, j in zip(images_accum, just_accum):
        if not j or np.isnan(j): continue
        i *= HEATMAP_SIZE-1; j *= HEATMAP_SIZE-1
        i = int(i); j = int(j)
        just_heat[HEATMAP_SIZE-1-j][i] += 1
    just_heat = np.array(just_heat)
    just_heat_accum = np.sum(just_heat, axis=1)
    just_heat_accum_decoded = []
    for i, accum in enumerate(just_heat_accum):
        just_heat_accum_decoded += [i]*accum
    just_quartile = np.percentile(just_heat_accum_decoded, [50, 25, 75])/HEATMAP_SIZE
    just_quartile = 1 - just_quartile
    
    heat_xlim = 1.05*max(np.max(before_heat_accum), np.max(just_heat_accum))
    
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Before")    
    ax1.imshow(before_heat, extent=[0, 1, 0, 1], interpolation="nearest", vmin=0, vmax=np.max(before_heat), cmap="inferno")
    ax1.set_xlabel("Luminance")
    ax1.set_ylabel("Grad-CAM")
    
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_title("Just")
    ax2.imshow(just_heat, extent=[0, 1, 0, 1], interpolation="nearest", vmin=0, vmax=np.max(just_heat), cmap="inferno")
    ax2.set_xlabel("Luminance")
    ax2.set_ylabel("Grad-CAM")
    
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.set_title("Before")
    ax3.plot([0, heat_xlim], [before_quartile[0]]*2, color="#dd2255")
    for i, val in enumerate(before_quartile[1:]):
        ax3.plot([0, heat_xlim], [val]*2, color="#22dd55")
    ax3.barh(np.array(range(HEATMAP_SIZE))/HEATMAP_SIZE, before_heat_accum[::-1], color="#8844dd", height=1/HEATMAP_SIZE)
    ax3.set_xlabel("Accumulated number")
    ax3.set_ylabel("Grad-CAM")
    ax3.set_xlim(0, heat_xlim)
    ax3.set_ylim(0, 1)
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Just")
    ax4.plot([0, heat_xlim], [just_quartile[0]]*2, color="#dd2255")
    for i, val in enumerate(just_quartile[1:]):
        ax4.plot([0, heat_xlim], [val]*2, color="#22dd55")
    ax4.barh(np.array(range(HEATMAP_SIZE))/HEATMAP_SIZE, just_heat_accum[::-1], color="#8844dd", height=1/HEATMAP_SIZE)
    ax4.set_xlabel("Accumulated number")
    ax4.set_ylabel("Grad-CAM")
    ax4.set_xlim(0, heat_xlim)
    ax4.set_ylim(0, 1)
    
    fig.suptitle("Relation of between Grad-CAM and luminance")
    
    fig.savefig(f"{LOAD_DIR}/{LOAD_ID}/{LOAD_ID}_{load_number:02d}.png")


def main():
    cp.cprint("\ncompleted : - / -", "pink")
    for number in range(*LOAD_NUMBER):
        make_png(number)
        cp.cprint(f"\033[1Acompleted : {number+1} / {LOAD_NUMBER[1]}", "pink")
    cp.cprint("- finished -", "cyan")


if(__name__ == "__main__"):
    main()
    