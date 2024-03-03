import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from colorPrint import Cprint as cp
from luminance_extender import LuminanceExtender
from affine_transform import AffineTransform as AT


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")

LE = LuminanceExtender((576, 576))
im.fold = 3
im.AE_model_id = "20221014-1"


DATASET_DIR = "/workspace/Dataset/fullframe"
# RESULT_DIR = "/workspace/visualization/correct_map"
RESULT_DIR = "/workspace/visualization"
LOAD_DIR = "/workspace/fullframe/result/540x540"
# LOAD_ID = lambda fold: f"e-unet_4class_adam_dropout_20220805_AutoLearning_fold{fold}_576x576"
# LOAD_ID = lambda fold: f"e-unet_4class_flip_20221030_AutoLearning_fold{fold}_576x576"
LOAD_ID = lambda fold: f"e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221109_AutoLearning_fold{fold}_576x576"
# LOAD_ID = lambda fold: f"e-unet_4class_use-AE-input_adam_dropout_20221014-1_AutoLearning_fold{fold}_576x576"
AE_CONFIG = [True, "20221014_ssim_mse"]

IS_AVG_IMG = False

KEYS = ["max", "min", "avg", "med", "dif"]
LABELS = ["before", "just"]

MODE = "default"
# MODE = "flip-ud"
# MODE = "flip-ud-lr"
# MODE = "flip-lr"
# MODE = "LE"

# save_path = lambda fold: f"{RESULT_DIR}/correct_map/{LOAD_ID(fold)}"
# save_path = lambda fold: f"{RESULT_DIR}/correct_map/buf"
save_path = lambda fold: f"{RESULT_DIR}/correct_map/{MODE}_{LOAD_ID(fold)}"
# save_path = lambda fold: f"{RESULT_DIR}/correct_map/LE_{LOAD_ID(fold)}"
path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"
    

def gen_cmap_name(colors):
    """
    カラーマップ
    """
    
    mx = float(len(colors)-1)
    color_list = []
    for i, c in enumerate(colors):
        color_list.append((i / mx, c))

    return mpl.colors.LinearSegmentedColormap.from_list('cmap', color_list)


def test(fold, mode=0):
    """
    集計元データ作成
    """
    
    ## 推論込み
    if (mode == 0):
        model = load_model(f"{LOAD_DIR}/{LOAD_ID(fold)}")
        Utils.makedir(save_path(fold))
            
        length = Utils.datacounter(path(fold, "test"))
        timecounter = TimeCounter(length)
        index = 0
        
        with open(path(fold, "test")) as f:
            readlines = f.readlines()
        length = len(readlines)
        
        if (MODE == "LE"):
            le_mode = ["all", "circle", "half", "slant"]
            le_const = [75, -75]
        else:
            le_mode = ["none"]
            le_const = [0]
            
        acc = dict(zip(le_mode, [np.zeros(3) for _ in le_mode]))
        denominator = dict(zip(le_mode, [np.zeros(3) for _ in le_mode]))
        
        for mode in le_mode:
            for const in le_const:
                if (mode == "none"):
                    buf = ""
                else:
                    buf = f"_{mode}-{str(const).replace('-', 'in')}"
                    
                with open(f"{save_path(fold)}/correct_map{buf}.txt", mode="w"): pass
                with open(f"{save_path(fold)}/correct_map_before{buf}.txt", mode="w"): pass
                with open(f"{save_path(fold)}/correct_map_just{buf}.txt", mode="w"): pass
                with open(f"{save_path(fold)}/accracy{buf}.txt", mode="w"): pass
        
        cp.cprint(f"mode : {MODE}", "orange")
        im.set_AE_id(AE_CONFIG[1])
        
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if (i >= length): break
            if i%10: continue
            
            linebuffer = line.split(" ")    
            
            img = Image.open(linebuffer[0])
            img = img.resize((576, 576))
            img = np.array(img, dtype=np.uint8)
            msk = Image.open(linebuffer[1])
            msk = msk.resize((576, 576))
            msk = np.array(msk, dtype=np.uint8)

            if (MODE == "flip-ud"):
                img = np.flipud(img)
                msk = np.flipud(msk)
            elif (MODE == "flip-ud-lr"):
                img = np.flipud(img)
                msk = np.flipud(msk)
                img = np.fliplr(img)
                msk = np.fliplr(msk)
            elif (MODE == "flip-lr"):
                img = np.fliplr(img)
                msk = np.fliplr(msk)

            img = np.reshape(img, (1, *img.shape))
            msk = np.reshape(msk, (1, *msk.shape))
            
            for section, mode in enumerate(le_mode):
                for element, const in enumerate(le_const):
                    img_data, msk_data = im.adjust_data(np.copy(img), np.copy(msk), True, False, 
                                                        IS_AVG_IMG, (576, 576), False, 4,
                                                        is_use_LE=MODE=="LE", LE_mode=mode, LE_const=const, noise_type="linear",
                                                        use_AE_input=AE_CONFIG[0])
                
                    pred ,= model.predict([img_data])
                    buf = np.zeros((*pred.shape[:2], 2))
                    buf[:, :, 0] = pred[:, :, 0] + pred[:, :, 1]
                    buf[:, :, 1] = pred[:, :, 2] + pred[:, :, 3]
                    pred = np.argmax(buf, axis=2)
                    msk_data = np.argmax(msk_data[0], axis=2)
                    
                    correct = np.uint8((pred ^ msk_data) == 0)
                    correct_before = correct * (msk_data == 0)
                    correct_just = correct * msk_data
                    acc[mode][0] += np.sum(correct)
                    acc[mode][1] += np.sum(correct_before)
                    acc[mode][2] += np.sum(correct_just)
                    denominator[mode][0] += 576*576
                    denominator[mode][1] += np.sum(msk_data == 0)
                    denominator[mode][2] += np.sum(msk_data)

                    encoded = [None]*correct.shape[0]
                    for name, cor in zip(["", "_before", "_just"], [correct, correct_before, correct_just]):
                        for row, val in enumerate(cor):
                            encoded[row] = im.encode_bin_img(val)
                        text = " ".join(encoded)
                        if (mode == "none"):
                            buf = ""
                        else:
                            buf = f"_{mode}-{str(const).replace('-', 'in')}"
                        with open(f"{save_path(fold)}/correct_map{name}{buf}.txt", mode="a") as f:
                            print(f"{text}", file=f)
                    
            if not index:
                print("\n")
                index = 1
                        
            remining_time = timecounter.predictTime(i+1)
            
            if (MODE == "LE"):
                cp.cprint(f"\033[1A{i+1} / {length} \t {remining_time} \t Acc : {round(100 * acc['all'][0] / denominator['all'][0], 3)} [%]{' '*30}", "green")
            else:
                cp.cprint(f"\033[1A{i+1} / {length} \t {remining_time} \t Acc : {round(100 * acc['none'][0] / denominator['none'][0], 3)} [%]{' '*30}", "green")
        
        for mode in le_mode:
            for const in le_const:
                if (mode == "none"):
                    buf = ""
                else:
                    buf = f"_{mode}-{str(const).replace('-', 'in')}"
                    if (mode != "all"):
                        denominator[mode] = denominator[mode] / 2
                    
                with open(f"{save_path(fold)}/accracy{buf}.txt", mode="a") as f:
                    print(f"{acc[mode] / denominator[mode]}", file=f)
            
    
    ## 照度計測
    elif (mode == 1):
        Utils.makedir(f"{RESULT_DIR}/luminance_map")
        
        length = Utils.datacounter(path(fold, "test"))
        timecounter = TimeCounter(length)
        index = 0
        
        with open(path(fold, "test")) as f:
            readlines = f.readlines()
        length = len(readlines)
        
        for key in KEYS:
            with open(f"{RESULT_DIR}/luminance_map/{key}.txt", mode="w") : pass
            
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if (i >= length): break
            if i%10: continue
            
            linebuffer = line.split(" ")    
            
            img = Image.open(linebuffer[0])
            img = img.resize((576, 576))
            img = np.array(img, dtype=np.float32)

            imgs = dict()
            imgs["max"] = np.max(img, axis=2)
            imgs["min"] = np.min(img, axis=2)
            imgs["avg"] = np.mean(img, axis=2)
            imgs["med"] = np.median(img, axis=2)
            imgs["dif"] = imgs["max"] - imgs["min"]
            
            for key in KEYS:
                text = ""
                for row, val in enumerate(imgs[key]):
                    text += "-".join(map(lambda x: hex(int(x))[2:], val))
                    text += " "
                text += "end"
                text = text.replace(" end", "")
                
                with open(f"{RESULT_DIR}/{key}.txt", mode="a") as f:
                    print(f"{text}", file=f)
            
            if not index:
                print("\n\n")
                index = 1
                
            remining_time = timecounter.predictTime(i+1)
            
            cp.cprint(f"\033[1A{i+1} / {length} \t{remining_time}{' '*30}", "green")


    ## ラベル集計
    elif (mode == 2):
        
        Utils.makedir(f"{RESULT_DIR}/answer_map")
        
        length = Utils.datacounter(path(fold, "test"))
        timecounter = TimeCounter(length)
        index = 0
        accum = dict(zip(LABELS, [0]*len(LABELS)))
        
        with open(path(fold, "test")) as f:
            readlines = f.readlines()
        length = len(readlines)
        
        for key in LABELS:
            with open(f"{RESULT_DIR}/answer_map/{key}.txt", mode="w") : pass
            
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if (i >= length): break
            if i%10: continue
            
            linebuffer = line.split(" ")
            
            msk = Image.open(linebuffer[1])
            msk = msk.resize((576, 576))
            msk = np.array(msk, dtype=np.uint8)
            
            msks = dict()
            msks["before"] = np.uint8(msk[:, :, 0] == 0)
            msks["just"] = np.uint8(msk[:, :, 0] == 255)

            encoded = [None]*msk.shape[0]
            for key in LABELS:
                for row, val in enumerate(msks[key]):
                    encoded[row] = im.encode_bin_img(val)
                text = " ".join(encoded)
                with open(f"{RESULT_DIR}/answer_map/{key}.txt", mode="a") as f:
                    print(f"{text}", file=f)
                
                accum[key] += int(np.sum(msks[key]))
            
            if not index:
                print("\n\n")
                index = 1
                
            remining_time = timecounter.predictTime(i+1)
            
            cp.cprint(f"\033[1A{i+1} / {length} \t{remining_time} \t ( {accum['before']}, {accum['just']} ){' '*30}", "green")
            
        with open(f"{RESULT_DIR}/answer_map/accum.txt", mode="w") as f:
            print(accum, file=f)
        cp.cprint(accum, "orange")
        
        
    ## 照度計測(AE)
    elif (mode == 3):
        Utils.makedir(f"{RESULT_DIR}/luminance_map_AE")
        
        length = Utils.datacounter(path(fold, "test"))
        timecounter = TimeCounter(length)
        index = 0
        
        with open(path(fold, "test")) as f:
            readlines = f.readlines()
        length = len(readlines)
        
        for key in KEYS:
            with open(f"{RESULT_DIR}/luminance_map_AE/{key}.txt", mode="w") : pass
            
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
            if (i >= length): break
            if i%10: continue
            
            linebuffer = line.split(" ")    
            
            img = Image.open(linebuffer[0])
            img = img.resize((576, 576))
            img = np.array(img, dtype=np.uint8)
            msk = Image.open(linebuffer[1])
            msk = msk.resize((576, 576))
            msk = np.array(msk, dtype=np.uint8)

            img = np.reshape(img, (1, *img.shape))
            msk = np.reshape(msk, (1, *msk.shape))
            
            img, *_ = im.adjust_data(
                np.copy(img), np.copy(msk), is_fullframe=True, is_use_average_image=False, size=(576, 576), classification="fourclasses", num_classes=4,
                use_AE_input=True
                )
            
            img = np.uint8(img[0]*255.)

            imgs = dict()
            imgs["max"] = np.max(img, axis=2)
            imgs["min"] = np.min(img, axis=2)
            imgs["avg"] = np.mean(img, axis=2)
            imgs["med"] = np.median(img, axis=2)
            imgs["dif"] = imgs["max"] - imgs["min"]

            for key in KEYS:
                text = ""
                for row, val in enumerate(imgs[key]):
                    text += "-".join(map(lambda x: hex(int(x))[2:], val))
                    text += " "
                text += "end"
                text = text.replace(" end", "")
                
                with open(f"{RESULT_DIR}/luminance_map_AE/{key}.txt", mode="a") as f:
                    print(f"{text}", file=f)
            
            if not index:
                print("\n\n")
                index = 1
                
            remining_time = timecounter.predictTime(i+1)
            
            cp.cprint(f"\033[1A{i+1} / {length} \t{remining_time}{' '*30}", "green")
        

def tally(fold, cmap, mode=0):
    """
    集計と、マップ画像の作成
    """
    
    ## 推論
    if (mode == 0):
        for section, name in enumerate(["", "_before", "_just"]):
            with open(f"{save_path(fold)}/correct_map{name}.txt", mode="r") as f:
                lines = f.readlines()
            length = len(lines)
            
            print()
            accum_map = np.float32(np.zeros((576, 576)))
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                image = np.array([im.decode_bin_img(row) for row in line.split(" ")], dtype=np.float32)
                accum_map += image
                cp.cprint(f"\033[1A{section+1} / 3\t{i+1} / {length}", "green")
            
            # accum_map -= np.min(accum_map)
            accum_map = accum_map / np.max(accum_map)
            accum_map *= 255.
            
            colored = cmap(np.uint8(accum_map))
            rgb_map = np.zeros((576, 576, 3))
            for i in range(3):
                rgb_map[:, :, i] += colored[:, :, i]*255.
            
            map_image = Image.fromarray(np.uint8(rgb_map))
            map_image.save(f"{save_path(fold)}/accum_map{name}.png")
    
    ## 照度
    elif (mode == 1):
        for key in KEYS:
            with open(f"{RESULT_DIR}/luminance_map/{key}.txt", mode="r") as f:
                lines = f.readlines()
            length = len(lines)
            
            print()
            accum_map = np.float32(np.zeros((576, 576)))
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                image = np.array([list(map(lambda x: int(x, 16), row.split("-"))) for row in line.split(" ")], dtype=np.float32)
                accum_map += image
                cp.cprint(f"\033[1A{key} : {i+1} / {length}", "green")
            
            accum_map -= np.min(accum_map)
            accum_map = accum_map / np.max(accum_map)
            accum_map *= 255.
            # colors = np.array([255, 0, 255], dtype=np.float32)

            colored = cmap(np.uint8(accum_map))
            rgb_map = np.zeros((576, 576, 3))
            for i in range(3):
                rgb_map[:, :, i] += colored[:, :, i]*255.
            # rgb_map[:, :, 0] += colors[0]*(accum_map)
            # rgb_map[:, :, 1] += colors[1]
            # rgb_map[:, :, 2] += colors[2]*(1-accum_map)
            
            map_image = Image.fromarray(np.uint8(rgb_map))
            map_image.save(f"{RESULT_DIR}/luminance_map/{key}.png")
            
            cp.cprint(f"\033[1Acompleted : {key}{' '*30}", "pink")
            
    ## ラベル集計
    elif (mode == 2):
        for key in LABELS:
            with open(f"{RESULT_DIR}/answer_map/{key}.txt", mode="r") as f:
                lines = f.readlines()
            length = len(lines)
            
            print()
            accum_map = np.float32(np.zeros((576, 576)))
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                image = np.array([im.decode_bin_img(row) for row in line.split(" ")], dtype=np.float32)
                accum_map += image
                cp.cprint(f"\033[1A{key} : {i+1} / {length}", "green")
            
            # accum_map -= np.min(accum_map)
            accum_map = accum_map / np.max(accum_map)
            accum_map *= 255.
            colored = cmap(np.uint8(accum_map))
            rgb_map = np.zeros((576, 576, 3))
            for i in range(3):
                rgb_map[:, :, i] += colored[:, :, i]*255.
            
            map_image = Image.fromarray(np.uint8(rgb_map))
            map_image.save(f"{RESULT_DIR}/answer_map/{key}.png")
            
            cp.cprint(f"\033[1Acompleted : {key}{' '*30}", "pink")
    
    ## 欠損有り推論
    if (mode == 3):
        
        col, row = 576, 576
        ## 重ねる円
        origin_x, origin_y = col/2, row/2
        radius_pow = col*row/(2*math.pi)
        circle_img = np.zeros((row, col))
        for y in range(row):
            for x in range(col):
                if ((x-origin_x)**2 + (y-origin_y)**2 <= radius_pow):
                    circle_img[y, x] = 1

        ## 重ねる斜め領域
        slant_img = np.zeros((row, col))
        dydx = row / col
        for y in range(row):
            for x in range(col):
                if (y <= dydx * x):
                    slant_img[y, x] = 1

        ## 重ねる半分割領域
        half_img = np.zeros((row, col))
        half_img[:, col//2:] += 1
        
        ## 反転
        circle_img = circle_img == 0
        slant_img = slant_img == 0
        half_img = half_img == 0
        
        cover_keys = ["circle", "slant", "half"]
        
        for section, name in enumerate(["", "_before", "_just"]):
            with open(f"{save_path(fold)}/correct_map{name}.txt", mode="r") as f:
                lines = f.readlines()
            length = len(lines)
            
            print()
            accum_map = np.uint64(np.zeros((576, 576)))
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                image = np.array([im.decode_bin_img(row) for row in line.split(" ")], dtype=np.uint64)
                accum_map += image
                cp.cprint(f"\033[1A{section+1} / 3\t{i+1} / {length}", "green")
                
            for key, cover_img in zip(cover_keys, [circle_img, slant_img, half_img]):
                covered = np.sum(accum_map*cover_img)
                with open(f"{save_path(fold)}/correct_map{name}_{key}.txt", mode="w") as f:
                    print(covered, file=f)
                cp.cprint(f"{key} : {covered}", "orange")
                
    ## LE
    elif (mode == 4):
        le_mode = ["all", "circle", "half", "slant"]
        le_const = [75, -75]
        
        for section, name in enumerate(["", "_before", "_just"]):
            for i, mode in enumerate(le_mode):
                for j, const in enumerate(le_const):
                    buf = f"_{mode}-{str(const).replace('-', 'in')}"
                    
                    with open(f"{save_path(fold)}/correct_map{name}{buf}.txt", mode="r") as f:
                        lines = f.readlines()
                    length = len(lines)
                    
                    print()
                    accum_map = np.float32(np.zeros((576, 576)))
                    for k, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                        image = np.array([im.decode_bin_img(row) for row in line.split(" ")], dtype=np.float32)
                        accum_map += image
                        cp.cprint(f"\033[1A{section+1} / 3, ( {i} - {j} ) \t{k+1} / {length}", "green")
                    
                    accum_map -= np.min(accum_map)
                    accum_map = accum_map / np.max(accum_map)
                    accum_map *= 255.
                    
                    colored = cmap(np.uint8(accum_map))
                    rgb_map = np.zeros((576, 576, 3))
                    for k in range(3):
                        rgb_map[:, :, k] += colored[:, :, k]*255.
                    
                    map_image = Image.fromarray(np.uint8(rgb_map))
                    map_image.save(f"{save_path(fold)}/accum_map{name}{buf}.png")
    
    
    ## 照度(AE)
    elif (mode == 5):
        for key in KEYS:
            with open(f"{RESULT_DIR}/luminance_map_AE/{key}.txt", mode="r") as f:
                lines = f.readlines()
            length = len(lines)
            
            print()
            accum_map = np.float32(np.zeros((576, 576)))
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                image = np.array([list(map(lambda x: int(x, 16), row.split("-"))) for row in line.split(" ")], dtype=np.float32)
                accum_map += image
                cp.cprint(f"\033[1A{key} : {i+1} / {length}", "green")
            
            accum_map -= np.min(accum_map)
            accum_map = accum_map / np.max(accum_map)
            accum_map *= 255.
            # colors = np.array([255, 0, 255], dtype=np.float32)

            colored = cmap(np.uint8(accum_map))
            rgb_map = np.zeros((576, 576, 3))
            for i in range(3):
                rgb_map[:, :, i] += colored[:, :, i]*255.
            # rgb_map[:, :, 0] += colors[0]*(accum_map)
            # rgb_map[:, :, 1] += colors[1]
            # rgb_map[:, :, 2] += colors[2]*(1-accum_map)
            
            map_image = Image.fromarray(np.uint8(rgb_map))
            map_image.save(f"{RESULT_DIR}/luminance_map_AE/{key}.png")
            
            cp.cprint(f"\033[1Acompleted : {key}{' '*30}", "pink")
            
            
def main():
    
    Utils.makedir(RESULT_DIR)
    
    ## 平均値大きめ
    # cmap = gen_cmap_name([
    #     "#000000",
    #     "#000044",
    #     "#000088",
    #     "#0000aa",
    #     "#0000ff",
    #     "#00ffff",
    #     "#00ff00",
    #     "#ffff00",
    #     "#ff0000",
    #     ])
    
    ## 普通
    cmap = gen_cmap_name([
        "#000000",
        "#ff00ff",
        "#0000ff",
        "#00ffff",
        "#00ff00",
        "#ffff00",
        "#ff0000",
        ])
    
    test(3, 0)
    tally(3, cmap, 0)
    # tally(3, cmap, 4)


main()
