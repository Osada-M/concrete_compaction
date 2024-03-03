from distutils.log import debug
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import umap
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from functools import lru_cache, partial
import random
import copy
import pickle
# import threading
# import multiprocessing as mp
from concurrent import futures

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


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel("ERROR")
ipt = IPT()
args = sys.argv


## ================ config ================


# DIR = "/workspace/fullframe/result/540x540"
DIR = "/workspace/mesh_encoder_result"
# DATASET_DIR = "/workspace/Dataset/fullframe"
DATASET_DIR = "/workspace/mesh_dataset"
STATE_TEXT = "/workspace/osada_ws/state.txt"
AVERAGE_IMAGE_PATH = "/workspace/osada_ws/average_image_0516.png"
SAVE_DIR = "/workspace/osada_ws/umap_result"

# MODEL = "e-unet_20220629_AutoLearning_fold4_576x576"
# MODEL = "e-unet_metric_classifier_20220629_AutoLearning_fold3_576x576"
# MODEL = "unet_metric_classifier_dropout_20220530_AutoLearning_fold4_540x540"
MODEL = "resnet_gray_fold5"
# FOLD = "each"
FOLD = [1, 2, 3, 4, 5]
# FOLD = "all"
MODE = "resnet"
# MODE = "eunet"

K_MEANS_INIT_POINTS = None

# END_LAYER = "add_50"
# END_LAYER = "model_3"
# END_LAYER = "classifier"
# END_LAYER = "output"
# END_LAYER = "activation_94" # fresh
END_LAYER = "activation_92" # image only (Best)
NORM = "batch_renorm"

# SIZE = (540, 540)
# SIZE = (576, 576)
# SIZE = (288, 288)
SIZE = (270, 270)
IS_GRAYSCALE = True

IS_USE_AVERAGE_IMAGE = True
IS_INCLUDE_FRESH = True
NUM_CLASSES = 2
BATCH_SIZE = 1
FRESH_KERNEL_SIZE = "auto"
LIMIT = None
USE_THREAD = False
BATCH_SIZE = 1000


## ========================================


if isinstance(FRESH_KERNEL_SIZE, str):
    if (FRESH_KERNEL_SIZE == "auto"):
        FRESH_KERNEL_SIZE = list(map(lambda x: x//16, SIZE))
    else:
        cp.cprint("\"FRESH_KERNEL_SIZE\" must be either [int, int] or \"auto\".")
        LIMIT = -1

if (MODE == "resnet"):
    path = lambda target : f"{DATASET_DIR}/fold{target[0]}/{target[1]}.txt"
    path_perent = lambda target : f"{DATASET_DIR}/{target}.txt"
else:
    path = lambda target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}{'_include_fresh' if IS_INCLUDE_FRESH else ''}.txt"
    path_perent = lambda target : f"{DATASET_DIR}/text_dataset/{target}{'_include_fresh' if IS_INCLUDE_FRESH else ''}.txt"


def get_data(resourcepath):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    images, masks, freshes = [], [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
    
    image_skip = 1
    # if (IMAGE_SKIP == "auto"):
        # image_skip = (line_length // SKIP_CONST) + 1
    # elif isinstance(IMAGE_SKIP, int):
        # image_skip = IMAGE_SKIP
        
    for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines)):
        if i%image_skip: continue
        linebuffer = line.split(" ")
        images.append(linebuffer[0])
        masks.append(linebuffer[1])
        if IS_INCLUDE_FRESH:
            freshes.append(list(map(float, linebuffer[2:])))
        else:
            freshes.append(None)
    length = len(images)
    
    return images, masks, freshes, length


def get_pickles():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    pickles = [None]*5
    for fold in range(1, 6):
        with open(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/umap_params{fold}.pickle", mode="rb") as f:
            pickles[fold-1] = pickle.load(f)
    
    return pickles


@partial
@Utils.ignore_unhashable
@lru_cache
def classdivide_with_umap(without_calc_umap:bool=False):
    
    if not without_calc_umap:
        if (len(args) > 2):
            with open(path_perent("all_4class"), mode="w"): pass
        
        ## スレッドの使用
        if USE_THREAD:
            # threads = [
            #     # threading.Thread(target=calc_umap, args=(fold,)) for fold in range(5)
            #     # mp.Process(target=calc_umap, args=(fold,)) for fold in range(5)
            # ]
            # for i, th in enumerate(threads):
            #     th.start()
            #     cp.cprint(f"start thread No.{i}", "cyan")
            # for i, th in enumerate(threads):
            #     th.join()
            #     cp.cprint(f"finished thread No.{i} !", "cyan")
            
            future_list = []
            with futures.ThreadPoolExecutor(max_workers=5) as executor:
                for fold in range(5):
                    future = executor.submit(calc_umap, fold)
                    future_list.append(future)
                _ = futures.as_completed(fs=future_list)
        
        else:
            calc_umap(int(args[1]))
            

@partial
@Utils.ignore_unhashable
@lru_cache
def calc_umap(thread_fold:int=None):
    """
    @機能：
    @引数：
    @戻値：
    """
        
    # with open(path([thread_fold, "4class"]), mode="w"): pass
    
    images, masks, freshes, length = get_data(path_perent("all"))
    avg_img = im.get_average_image(SIZE, is_grayscale=IS_GRAYSCALE)
    
    pickles = get_pickles()
    pickle_fold = dict()
    for i, p in enumerate(pickles):
        for key in p.fold_keys:
            pickle_fold[key] = i
            
    # classes, ipts = [0]*length, [0]*length
    fold_keys = set()
    classes_index = [iter(list(range(4))) for _ in range(5)]
    converted_class = [dict() for _ in range(5)]
    skip_count = 3
    
    count = 0
    if thread_fold is not None:
        color_args = [{"color" : "black", "background" : "white"},
                    {"color" : "black", "background" : "orange"},
                    {"color" : "black", "background" : "red"},
                    {"color" : "black", "background" : "green"},
                    {"color" : "black", "background" : "blue"}][thread_fold]
    else:
        color_args = {"color" : None, "background" : None}
    
    cp.cprint(f"reading model ...", **color_args)
    
    labeling_model = None
    inputs = []
        
    if (MODE == "resnet"):
        model = load_model(f"{DIR}/{MODEL}")
        for index, layer in enumerate(model.layers):
            if ("input" in layer.name):
                inputs.append(layer.input)
            if (layer.name == END_LAYER): break
        labeling_model = Model(inputs=inputs, outputs=model.layers[index].output)
    
    else:
        model = load_model(f"{DIR}/{MODEL}")
        # model.summary()
        
        for end_number, layer in enumerate(model.layers):
            print(layer.name)
            if (layer.name == END_LAYER): break
        try:
            labeling_model = Model(inputs=model.layers[0].input, outputs=model.layers[end_number].output)
        except:
            labeling_model = Model(inputs=model.layers[1].input, outputs=model.layers[end_number].output)
    del model
    
    cp.cprint(f"start calculation !", **color_args)
    
    batch = []
    img_batch = []
    frs_batch = []
    
    for i, (img, msk, frs) in enumerate(zip(images, masks, freshes)):
        
        if (i == skip_count):
            # if USE_THREAD:
            timecounter = TimeCounter((length-skip_count+1) // 5)
            # else:
            #     timecounter = TimeCounter(length-skip_count+1)

        key = im.get_image_key(img)
        
        fold_keys.add(key)
        fold = pickle_fold[key]
        
        ## スレッド使用時のデータの分割
        if (thread_fold is not None) and (fold != thread_fold): continue
        
        # if (count < 460000):
            # count += 1
            # continue
        
        mapper = pickles[fold]
        gravities = mapper.gravity
        
        # ipts[i] = ipt.ipt(img)
        gravity_dist = np.zeros((4))
        
        img_path = img
        
        img = Image.open(img)
        img = img.resize(SIZE)
        img = np.array(img, dtype=np.float32)
        if IS_GRAYSCALE:
            img = np.average(img, axis=2)
        img = np.reshape(img, (1, *SIZE, 3**(not IS_GRAYSCALE)))
        img -= avg_img
        img /= 255.
        
        if (MODE == "resnet"):
            msk = int(msk)
        else:
            msk = Image.open(msk)
            msk = msk.resize(SIZE)
            msk = np.array(msk, dtype=np.float32)
            msk = np.reshape(msk[:, :, 0], (1, *SIZE, 1))
            msk /= 255.
            
        if LIMIT and (i >= LIMIT): break
        if IS_INCLUDE_FRESH:
            frs = np.array(frs, dtype=np.float32)
            frs = np.reshape(frs, (1, 5))
            msk = np.array(msk)
            msk = np.reshape(msk, (1, 1))
            
            if (len(inputs) == 3):
                pred_buf = labeling_model.predict([img, frs, msk])[0]
            else:
                pred_buf = labeling_model.predict([img, frs])[0]
        else:
            pred_buf ,= labeling_model.predict([img])
            pred_buf = pred_buf[:, :, 0]
            
        batch.append(np.ravel(pred_buf).tolist())
        img_batch.append(img_path)
        frs_batch.append(frs)

        if (len(batch) >= BATCH_SIZE):
            
            print("calculating classes ...")
            coordinate = mapper.mapper.transform(batch)
            del batch
            batch = []
            
            for k, coord in enumerate(coordinate):
                x = coord[0] / mapper.x_max
                y = coord[1] / mapper.y_max
                z = coord[2] / mapper.z_max
                
                for class_index, anchor in enumerate(gravities):
                    gravity_dist[class_index] = np.sum([np.power(x - anchor[0], 2),
                                                        np.power(y - anchor[1], 2),
                                                        np.power(z - anchor[2], 2)])
                
                class_buf = np.argmin(np.array(gravity_dist, dtype=np.float32))
                if not class_buf in converted_class[fold].keys():
                    converted_class[fold][class_buf] = next(classes_index[fold])
                class_number = converted_class[fold][class_buf]
                
                with open(path_perent("all_4class"), mode="a") as f:
                    print(f"{img_batch[k]} {class_number} {' '.join(map(str, frs_batch[k][0]))}", file=f)

            img_batch = []
            frs_batch = []

        # with open(path([thread_fold, "4class"]), mode="a") as f:
            # print(f"{img_path} {class_number} {' '.join(map(str, frs[0]))}", file=f)
        
        if (count >= skip_count):
            remining_time = timecounter.predictTime(count-skip_count+2)
        else:
            remining_time = "--:--:--"

        cp.cprint(f"key       : {key}, pickle    : No.{fold}", **color_args)
        # cp.cprint(f"classes   : {converted_class}", **color_args)
        # cp.cprint(f"answer    : {msk}, class     : {class_number}", **color_args)
        # if USE_THREAD:
        cp.cprint(f"completed : {count-skip_count+2} / {length-skip_count+1} ({(length-skip_count+1) // 5}) | {remining_time}{' '*10}", **color_args)
        # else:
            # cp.cprint(f"completed : {count-skip_count+2} / {length-skip_count+1} | {remining_time}{' '*10}", **color_args)
        
        count += 1
    
    ## 余剰分を計算
    if len(batch):
        print("calculating classes ...")
        coordinate = mapper.mapper.transform(batch)

        for k, coord in enumerate(coordinate):
            x = coord[0] / mapper.x_max
            y = coord[1] / mapper.y_max
            z = coord[2] / mapper.z_max
            
            for class_index, anchor in enumerate(gravities):
                gravity_dist[class_index] = np.sum([np.power(x - anchor[0], 2),
                                                    np.power(y - anchor[1], 2),
                                                    np.power(z - anchor[2], 2)])
            
            class_buf = np.argmin(np.array(gravity_dist, dtype=np.float32))
            if not class_buf in converted_class[fold].keys():
                converted_class[fold][class_buf] = next(classes_index[fold])
            class_number = converted_class[fold][class_buf]
            
            with open(path_perent("all_4class"), mode="a") as f:
                print(f"{img_batch[k]} {class_number} {' '.join(map(str, frs_batch[k][0]))}", file=f)
    
    print("finished !")
        

def diside_classes(pickle_fold):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    # with open(path_perent("all_4class"), mode="w"): pass
    
    cp.cprint(f"\nread all data ...", "cyan")
    images, masks, freshes, length = get_data(path_perent("all"))
    cp.cprint(f"\nread 4class data ...", "cyan")
    images_, classes, freshes_, length_ = get_data(path_perent("all_4class"))
    
    cp.cprint(f"\nexcute k-means ...", "cyan")
    
    data = dict()
    for i, (img, msk, frs) in enumerate(zip(images, masks, freshes)):
        key = im.get_image_key(img)
        class_number = classes[i]
        ipt_number = im.get_image_key(img)
        if (class_number in [0, 1]) and (msk == 1):
            class_number = 2
        elif (class_number in [2, 3]) and (msk == 0):
            class_number = 1
        data[key] = data[key].append([ipt_number, class_number, msk]) if key in data.keys() else [ipt_number, class_number, msk]
    
    return data
    
    # with open(path_perent("all_4class"), mode="a") as f:
    #     print(f"{img_path} {classes[i]} {' '.join(map(str, frs[0]))}", file=f)


def main():
    
    pickles = get_pickles()
    for p in pickles:
        try:
            print(f"{p.fold = }\t[ {cp.colored('OK', 'green')} ]")
        except:
            print(f"p.fold = -\t[ {cp.colored('NG', 'red')} ]")
    
    # classes, ipts, pickle_fold = classdivide_with_umap(pickles)
    classdivide_with_umap()
    # diside_classes(classes, ipts, pickle_fold)
    
    
    ##### クラス定義後、こっちを実行する
    # pickle_fold = classdivide_with_umap(pickles, without_calc_umap=True)
    # data = diside_classes(None, None, pickle_fold)
    
    # print(data)


class UmapClasses:
    """
    @機能：パラメータ呼び出し用
    @引数：
    @戻値：
    """
    
    def __init__(self, mapper, gravity, fold, maxes, fold_keys):
        self.mapper = mapper
        self.gravity = gravity
        self.fold = fold
        self.fold_keys = fold_keys
        x, y, z = maxes
        self.x_max = x
        self.y_max = y
        self.z_max = z
        self.random_state = int(fold == 5)
        
        
if(__name__ == "__main__"):
    main()
