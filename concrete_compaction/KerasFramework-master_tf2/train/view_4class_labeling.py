import matplotlib.pyplot as plt
import numpy as np

## my modules
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from line_sender import send_master
from image_path_timer import ImagePathTime as IPT


ipt = IPT()


## ================ config ================


DIR = "/workspace/mesh_dataset"
# PIC_SAVE_DIR = "/workspace/osada_ws/ex_4class/origin"
PIC_SAVE_DIR = "/workspace/osada_ws/ex_4class/myself_0-25"


## ========================================


def get_all_data():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    img, ans, fsh = [], [], []
    with open(f"{DIR}/all_4class_myself.txt", mode="r") as f:
        readlines = f.readlines()
        for i, line in enumerate(readlines):
            if len(line.rstrip("\n")):
                buf = line.split(" ")
                img.append(buf[0])
                ans.append(buf[1])
                fsh.append(buf[2:])
    
    length = len(img)
    
    return img, ans, fsh, length


def draw_graph(img, ans, fsh):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    data = dict()
    
    for img_val, ans_val in zip(img, ans):
        key = im.get_image_key(img_val)
        key0, key1, mesh_id = key.split("_")
        key = "_".join([key0, key1])
        if not key in data.keys():
            # data[key] = [[[], []] for _ in range(24)]
            data[key] = [[[], ipt.parent[key]] for _ in range(24)]
        data[key][int(mesh_id)-1][0].append(int(ipt.ipt(img_val) >= 0))
        # data[key][int(mesh_id)-1][1].append(int(ans_val))
        # data[key][int(mesh_id)-1][1].append(ipt.ipt(img_val))
    
    print(ipt.debug_set)
        
    for key, val in data.items():
        plt.figure(figsize=(10, 10))
        for i, data_list in enumerate(val):
            true_ans, umap_ans = data_list
            range_y_top = [i+0.5, 0.5]
            range_y_bottom = [i+1, 0.5]
            
            # plt.scatter(range(len(umap_ans)), umap_ans, c=true_ans)
            
            # umap_bar_0 = [index for index, val in enumerate(umap_ans) if (val == 0)]
            # umap_bar_0 = list(zip(umap_bar_0, np.ones(len(umap_bar_0))))
            
            # umap_bar_1 = [index for index, val in enumerate(umap_ans) if (val == 1)]
            # umap_bar_1 = list(zip(umap_bar_1, np.ones(len(umap_bar_1))))
            
            # umap_bar_2 = [index for index, val in enumerate(umap_ans) if (val == 2)]
            # umap_bar_2 = list(zip(umap_bar_2, np.ones(len(umap_bar_2))))
            
            # umap_bar_3 = [index for index, val in enumerate(umap_ans) if (val == 3)]
            # umap_bar_3 = list(zip(umap_bar_3, np.ones(len(umap_bar_3))))
            
            # umap_bar_0 = [index for index, val in enumerate(umap_ans) if (val < -0.25)]
            # umap_bar_0 = list(zip(umap_bar_0, np.ones(len(umap_bar_0))))
            
            # umap_bar_1 = [index for index, val in enumerate(umap_ans) if (-0.25 <= val < 0)]
            # umap_bar_1 = list(zip(umap_bar_1, np.ones(len(umap_bar_1))))
            
            # umap_bar_2 = [index for index, val in enumerate(umap_ans) if (0 <= val < 0.25)]
            # umap_bar_2 = list(zip(umap_bar_2, np.ones(len(umap_bar_2))))
            
            # umap_bar_3 = [index for index, val in enumerate(umap_ans) if (0.25 <= val)]
            # umap_bar_3 = list(zip(umap_bar_3, np.ones(len(umap_bar_3))))
            
            true_bar_just = [index for index, val in enumerate(true_ans) if val]
            
            ####           
            true_length = len(true_bar_just)
            ####
            
            true_bar_just = list(zip(true_bar_just, np.ones(len(true_bar_just))))
            
            true_bar_before = [index for index, val in enumerate(true_ans) if not val]
            
            ####
            umap_ans = list(map(lambda x: int(x*0.25), umap_ans))
            before_length = len(true_bar_before)
            
            if (umap_ans[0] <= before_length):
                umap_bar_0 = list(range(0, before_length - umap_ans[0]))
                umap_bar_0 = list(zip(umap_bar_0, np.ones(len(umap_bar_0))))
                umap_bar_1 = list(range(before_length - umap_ans[0], before_length))
                umap_bar_1 = list(zip(umap_bar_1, np.ones(len(umap_bar_1))))
                plt.broken_barh(xranges=umap_bar_0, yrange=range_y_top, facecolor="#ffffff")
                plt.broken_barh(xranges=umap_bar_1, yrange=range_y_top, facecolor="#ee33ee")
            else:
                umap_bar_1 = list(range(0, before_length))
                umap_bar_1 = list(zip(umap_bar_1, np.ones(len(umap_bar_1))))
                plt.broken_barh(xranges=umap_bar_1, yrange=range_y_top, facecolor="#ee33ee")
            
            if (umap_ans[1] <= true_length):
                umap_bar_2 = list(range(before_length, before_length + umap_ans[1]))
                umap_bar_2 = list(zip(umap_bar_2, np.ones(len(umap_bar_2))))
                umap_bar_3 = list(range(before_length + umap_ans[1], before_length + true_length))
                umap_bar_3 = list(zip(umap_bar_3, np.ones(len(umap_bar_3))))
                plt.broken_barh(xranges=umap_bar_2, yrange=range_y_top, facecolor="#eeee33")
                plt.broken_barh(xranges=umap_bar_3, yrange=range_y_top, facecolor="#33eeee")
            else:
                umap_bar_2 = list(range(before_length, before_length + true_length))
                umap_bar_2 = list(zip(umap_bar_2, np.ones(len(umap_bar_2))))
                plt.broken_barh(xranges=umap_bar_2, yrange=range_y_top, facecolor="#eeee33")
            ####
            
            true_bar_before = list(zip(true_bar_before, np.ones(len(true_bar_before))))
            
            # plt.broken_barh(xranges=umap_bar_0, yrange=range_y_top, facecolor="#ffffff")
            # plt.broken_barh(xranges=umap_bar_1, yrange=range_y_top, facecolor="#ee33ee")
            # plt.broken_barh(xranges=umap_bar_2, yrange=range_y_top, facecolor="#eeee33")
            # plt.broken_barh(xranges=umap_bar_3, yrange=range_y_top, facecolor="#33eeee")
            plt.broken_barh(xranges=true_bar_just, yrange=range_y_bottom, facecolor="green")
            plt.broken_barh(xranges=true_bar_before, yrange=range_y_bottom, facecolor="red")
        
        plt.xlabel("frame")
        plt.ylabel("mesh number")
        
        plt.savefig(f"{PIC_SAVE_DIR}/{key}.png")
        cp.cprint(f"@ Saved plot image. key : {key}", "orange")
        
        plt.cla()
        plt.clf()
        plt.close()


def view_label():
    
    img, ans, fsh, length = get_all_data()
    draw_graph(img, ans, fsh)
    
    
def main():
    Utils.makedir(PIC_SAVE_DIR)
    view_label()


main()
