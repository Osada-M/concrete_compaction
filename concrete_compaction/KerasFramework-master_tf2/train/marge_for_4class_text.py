import pickle
import matplotlib.pyplot as plt
import numpy as np

## my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
import MetricLearning_for_semseg as metric_semseg
from line_sender import send_master
from mymodel.SemSegLight import SemSegLight
from mymodel.CreateModel import CreateModel
from image_path_timer import ImagePathTime as IPT


ipt = IPT()


## ================ config ================


DIR = "/workspace/mesh_dataset"
# PIC_SAVE_DIR = "/workspace/osada_ws/ex_4class/origin"
PIC_SAVE_DIR = "/workspace/osada_ws/ex_4class/mean"
FRESH_PICKLE = "/workspace/mesh_dataset/fresh.pickle"


## ========================================



def get_all_data():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    data = []
    with open(f"{DIR}/all_4class_mypc.txt", mode="r") as f:
        readlines = f.readlines()
        for i, line in enumerate(readlines):
            if  len(line.rstrip("\n")):
                data.append(line)
    with open(f"{DIR}/all_4class_dlbox.txt", mode="r") as f:
        readlines = f.readlines()
        for i, line in enumerate(readlines):
            if len(line.rstrip("\n")):
                data.append(line)
    
    # with open(f"{DIR}/all_4class_marge.txt", mode="r") as f:
    #     readlines = f.readlines()
    #     for i, line in enumerate(readlines):
    #         if len(line.rstrip("\n")):
    #             data.append(line)
    
    length = len(data)
    
    return data, length


def write_data(data):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    # with open(f"{DIR}/all_4class_marge.txt", mode="w") as f:
    with open(f"{DIR}/all_4class.txt", mode="w") as f:
        f.write("\n".join(data))


def draw_graph(img, ans):
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
            data[key] = [[[], []] for _ in range(24)]
        data[key][int(mesh_id)-1][0].append(int(ipt.ipt(img_val) >= 0))
        data[key][int(mesh_id)-1][1].append(int(ans_val))
        
    for key, val in data.items():
        plt.figure(figsize=(10, 10))
        for i, data_list in enumerate(val):
            true_ans, umap_ans = data_list
            
            range_y_top = [i+0.5, 0.5]
            range_y_bottom = [i+1, 0.5]
            
            umap_bar_0 = [index for index, val in enumerate(umap_ans) if (val == 0)]
            umap_bar_0 = list(zip(umap_bar_0, np.ones(len(umap_bar_0))))
            
            umap_bar_1 = [index for index, val in enumerate(umap_ans) if (val == 1)]
            umap_bar_1 = list(zip(umap_bar_1, np.ones(len(umap_bar_1))))
            
            umap_bar_2 = [index for index, val in enumerate(umap_ans) if (val == 2)]
            umap_bar_2 = list(zip(umap_bar_2, np.ones(len(umap_bar_2))))
            
            umap_bar_3 = [index for index, val in enumerate(umap_ans) if (val == 3)]
            umap_bar_3 = list(zip(umap_bar_3, np.ones(len(umap_bar_3))))
            
            true_bar_just = [index for index, val in enumerate(true_ans) if val]
            true_bar_just = list(zip(true_bar_just, np.ones(len(true_bar_just))))
            
            true_bar_before = [index for index, val in enumerate(true_ans) if not val]
            true_bar_before = list(zip(true_bar_before, np.ones(len(true_bar_before))))
            
            plt.broken_barh(xranges=umap_bar_0, yrange=range_y_top, facecolor="#ffffff")
            plt.broken_barh(xranges=umap_bar_1, yrange=range_y_top, facecolor="#ee33ee")
            plt.broken_barh(xranges=umap_bar_2, yrange=range_y_top, facecolor="#eeee33")
            plt.broken_barh(xranges=umap_bar_3, yrange=range_y_top, facecolor="#33eeee")
            plt.broken_barh(xranges=true_bar_just, yrange=range_y_bottom, facecolor="green")
            plt.broken_barh(xranges=true_bar_before, yrange=range_y_bottom, facecolor="red")
        
        plt.xlabel("frame")
        plt.ylabel("mesh number")
        
        plt.savefig(f"{PIC_SAVE_DIR}/{key}.png")
        cp.cprint(f"@ Saved plot image. key : {key}", "orange")
        
        plt.cla()
        plt.clf()
        plt.close()
        

def edit_data(data, length):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    result = [None]*length
    four_classes = dict()
    four_classes_number = dict()
    data = list(sorted(data))
    image, answer = [None]*length, [None]*length
    
    ## UMAPとk-meansによる判定を取得
    for i, vals in enumerate(data):
        img, ans, *frs = vals.split(" ")
        ans = int(ans)
        ipt_id = ipt.ipt(img)
        true_ans = int(ipt_id >= 0)
        
        key = im.get_image_key(img)
        if not key in four_classes.keys():
            four_classes[key] = np.zeros((2, 4))
            four_classes_number[key] = np.zeros((2, 4))
        four_classes[key][true_ans, ans] += ipt_id
        four_classes_number[key][true_ans, ans] += 1

    ## 最も大きな幅を取るラベルと、それ以外を合計したものでBefore,Justそれぞれ２クラスずつに分ける    
    for cls_key in four_classes.keys():
        ## それぞれの時間的平均(IPTの平均)を取る
        four_classes[cls_key] /= (four_classes_number[cls_key] - (four_classes_number[cls_key] == 0))
        four_classes[cls_key] *= (four_classes_number[cls_key] > 100)

    search_just = dict()
    just = dict()
    before = dict()
    
    ## データを実際に分類
    for i, vals in enumerate(data):
        img, ans, *frs = vals.split(" ")
        key = im.get_image_key(img)
        ipt_id = ipt.ipt(img)
        true_ans = int(ipt_id >= 0)
        
        ## k-meansにより分類されたクラスを算出
        four_class_buf = np.power(four_classes[key][true_ans] - ipt_id, 2)
        ans = np.argmin(four_class_buf)
        
        ## Beforeを修正
        if (true_ans == 0): ans = [0, 1, 1, 1][ans]
        
        ## b-JustとJustがクラス２と３のどっちなのかを動的に算出(最後に更新された値が正解)
        key_mesh = im.get_image_key(img, True)
        mesh_id, _ = im.get_id(img)
        if not (key_mesh in search_just.keys()): search_just[key_mesh] = [0]*24
        if (true_ans == 1):
            if (ans in [2, 3]): search_just[key_mesh][mesh_id] = ans-2
            if (ans == 1): search_just[key_mesh][mesh_id] = 0
        ## クラス２と３の振り分けを平均より算出
        just[key_mesh] = int((sum(search_just[key_mesh]) / 24) >= 1/2)
        
        ## クラス０と１の飛び越えを検知
        if (ans in [0, 1]) and not (key in before.keys()):
            before[key] = ans

        image[i] = img
        answer[i] = ans
    
    ## 特殊なことをするためのフラグ(過学習してて４クラスになってないもの)
    just["190731_06"] = -1
    just["190731_07"] = -2
    
    ## 分類結果から、４クラスを算出
    for i, (img, ans) in enumerate(zip(image, answer)):
        ipt_id = ipt.ipt(img)
        true_ans = int(ipt_id >= 0)
        key = im.get_image_key(img)
        key_mesh = im.get_image_key(img, True)
        mesh_id, _ = im.get_id(img)

        ## 必要に応じてb-JustとJustを反転
        if not just[key_mesh]: ans = [0, 1, 3, 2][ans]
        
        ## Justを修正
        if (true_ans == 1):
            if (just[key_mesh] == -1):
                if (mesh_id in [5, 8]):
                    ans = [3, 3, 2, 2][ans]
                elif (mesh_id in [9, 16]):
                    ans = [2, 3, 3, 2][ans]
                elif (mesh_id in [11, 17, 23]):
                    ans = [2, 3, 2, 2][ans]
                elif (mesh_id in [13]):
                    ans = [2, 2, 3, 3][ans]
                elif (mesh_id in [15]):
                    ans = [2, 3, 2, 3][ans]
                else:
                    ans = [2, 3, 3, 3][ans]
            elif (just[key_mesh] == -2):
                ans = [2, 3, 2, 2][ans]
            else:
                ans = [2, 2, 2, 3][ans]
            
        ## Justのクラス間の飛び越えを修正
        if (just[key_mesh] >= 0):
            if (search_just[key_mesh][mesh_id] != just[key_mesh]):
                ans = [0, 1, 2, 2][ans]
        
        ## Beforeのクラス間の飛び越えを修正
        if before[key]: ans = [1, 1, 2, 3][ans]
        
        answer[i] = ans
        
        ## フレッシュ性状データを正しく改変
        fresh = im.get_fresh(img)
        fresh = " ".join(map(str, fresh))
        
        ## テキストへの保存文言
        result[i] = f"{img} {ans} {fresh}"
    
    # draw_graph(image, answer)
    
    return result


def main():
    
    data, length = get_all_data()
    data = edit_data(data, length)
    write_data(data)


if(__name__ == "__main__"):
    main()
    