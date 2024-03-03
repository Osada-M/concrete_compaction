import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import umap
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from functools import lru_cache, partial
import random
import copy
import pickle

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

# K_MEANS_INIT_POINTS = None
## 大体正しいやつ
K_MEANS_INIT_POINTS = [[[0.5, 0.8, 0.6], [0.9, 1.0, 0.9], [0.7, 0.9, 0.7], [0.3, 0.7, 0.5]],
                       [[0.4, 0.3, 0.9], [0.9, 0.9, 0.7], [0.3, 0.5, 0.8], [0.9, 0.8, 0.7]],
                       [[-1.0, 1.0, 0.8], [-0.8, 0.9, 0.6], [0.1, 0.8, 0.5], [0.7, 0.6, 0.5]],
                       [[0.4, 0.8, 0.3], [0.7, 0.5, 0.5], [0.9, 0.0, 0.8], [1.0, -0.2, 0.9]],
                       [[-0.1, -1.2, 0.2], [0.5, 0.2, 0.5], [0.8, 0.7, 0.8], [0.3, -0.4, 0.2]]]

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
IS_GRAYSCALE = 1

IS_USE_AVERAGE_IMAGE = True
IS_INCLUDE_FRESH = True
NUM_CLASSES = 2
BATCH_SIZE = 1
FRESH_KERNEL_SIZE = "auto"
LIMIT = None
# LIMIT = 50

IMAGE_SKIP = "auto"
## IMAGE_SKIP = 画像枚数をSKIP_CONST以下の最大の約数で割った商を繰り上げた値 : int(length / SKIP_CONST) + 1
SKIP_CONST = 1000
# SKIP_CONST = 400
# SKIP_CONST = 50


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
    
    line_length = len(readlines)
    image_skip = 1
    if (IMAGE_SKIP == "auto"):
        image_skip = (line_length // SKIP_CONST) + 1
    elif isinstance(IMAGE_SKIP, int):
        image_skip = IMAGE_SKIP
        
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


@partial
@lru_cache
def execute_umap():
    
    avg_img = im.get_average_image(SIZE, is_grayscale=IS_GRAYSCALE)
    
    labeling_model = None
        
    if (MODE == "resnet"):
        model = load_model(f"{DIR}/{MODEL}")
        inputs = []
        for index, layer in enumerate(model.layers):
            if ("input" in layer.name):
                inputs.append(layer.input)
            if (layer.name == END_LAYER): break
            # labeling_model.add(layer)
        labeling_model = Model(inputs=inputs, outputs=model.layers[index].output)
        # createmodel = CreateModel(None)
        # labeling_model = createmodel.create_model("mlt_resnet18_sphereface", isMlt=True)
        # labeling_model.load_weights(f"{DIR}/{MODEL}.h5")
        # labeling_model.trainable = False
        # labeling_model.compile(optimizer=Adam(lr=0.001),
        #                 loss="categorical_crossentropy")
    
    else:
        model = load_model(f"{DIR}/{MODEL}")
        model.summary()
        
        # labeling_model = Sequential()
        for end_number, layer in enumerate(model.layers):
            print(layer.name)
            # labeling_model.add(layer)
            if (layer.name == END_LAYER): break
        try:
            labeling_model = Model(inputs=model.layers[0].input, outputs=model.layers[end_number].output)
        except:
            labeling_model = Model(inputs=model.layers[1].input, outputs=model.layers[end_number].output)
    
    del model
    
    labeling_model.summary()
    for_list = [[f, "test"] for f in FOLD]
    if (FOLD == "each"):
        for_list = [[i, "test"] for i in range(1, 6)]
    length = len(for_list)
    
    for target in for_list:
        images, masks, freshes, length = get_data(path(target)) ##### allに対応する
        preds, anses = [None]*(LIMIT if LIMIT else length), [None]*(LIMIT if LIMIT else length)
        ipts = []
        fold_keys = set()
        
        print("\n")
        
        for i, (img, msk, frs) in enumerate(zip(images, masks, freshes)):
            
            path_elements = img.split("/")
            _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
            element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
            date, place, time_id, mesh_id = element.split("_")
            key = f"{date}_{place}_{mesh_id}"
            fold_keys.add(key)
            
            ipts.append(ipt.ipt(img))
            
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
                # pred_buf = pred_buf[0]
            else:
                pred_buf ,= labeling_model.predict([img])
                pred_buf = pred_buf[:, :, 0]
            preds[i] = np.ravel(pred_buf).tolist()
            print(f"\033[1Aumap\t: {i+1} / {length}")
            
            del img, msk, pred_buf
        
        preds = np.array(preds)
        
        mapper = umap.UMAP(random_state=int(target[0] == 5), n_components=3)
        embedding = mapper.fit_transform(preds)
        
        yield mapper, embedding, ipts, length, fold_keys


def draw_embedding(embedding, ipts, fold:int):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    Utils.makedir(f"{SAVE_DIR}/{MODEL}_{END_LAYER}")
        
    x = embedding[:, 0]
    y = embedding[:, 1]
    z = embedding[:, 2]
    
    x_m, y_m, z_m = np.max(x), np.max(y), np.max(z)
    
    x /= x_m
    y /= y_m
    z /= z_m
    
    with open(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/{fold}.txt", mode="w") as f:
        f.write("ipt,x,y,z\n")
        for ipt, x_val, y_val, z_val in zip(ipts, x, y, z):
            f.write(f"{ipt},{x_val},{y_val},{z_val}\n")
    
    colorlist = IPT.ipt_color(ipts)
    
    ## 2次元直交座標
    fig = plt.figure()
    ax = fig.add_subplot(projection="rectilinear")
    ax.scatter(x, y, s=5, c=colorlist)
    plt.savefig(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/{fold}.png")
    plt.cla()
    
    ## 3次元直交座標
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, s=5, c=colorlist)
    plt.savefig(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/{fold}_3d.png")
    plt.cla()


def kmeans(embedding, fold:int="all"):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    x = embedding[:, 0]
    y = embedding[:, 1]
    z = embedding[:, 2]
    
    x_m, y_m, z_m = np.max(x), np.max(y), np.max(z)
    
    x /= x_m
    y /= y_m
    z /= z_m
    
    sets = [x, y, z]
    length = len(x)
    points = list(zip(*sets))
    
    if K_MEANS_INIT_POINTS is None:    
        gravity_points = np.array([list(points.pop(random.randint(0, length-1-class_num))) for class_num in range(4)], dtype=np.float32)
    else:
        gravity_points = copy.deepcopy(K_MEANS_INIT_POINTS[fold-1])
    old = np.array([[1e10 for vals in sets] for class_ in range(4)], dtype=np.float32)
    limit = 1e3
    del points
    
    print()
    
    while limit:
        
        is_fixed = False
        for (g, o) in zip(gravity_points, old):
            is_fixed = (g[0] == o[0]) and (g[1] == o[1]) and (g[2] == o[2])
        if is_fixed: break
        old = np.copy(gravity_points)
        
        gravity_dist = np.zeros((length, 4))
        classes = np.array([0]*length)
        
        for point_index, (x_set, y_set, z_set) in enumerate(zip(*sets)):
            for class_index, anchor in enumerate(gravity_points):
                gravity_dist[point_index, class_index] = np.sum([np.power(x_set - anchor[0], 2),
                                                                 np.power(y_set - anchor[1], 2),
                                                                 np.power(z_set - anchor[2], 2)])
        
            classes[point_index] = np.argmin(np.array(gravity_dist[point_index], dtype=np.float32))
        gravity_points = np.array([np.average(np.array(list(zip(*sets)), dtype=np.float32)[classes == class_index], axis=0) for class_index in range(4)], dtype=np.float32)

        limit -= 1
        print(f"\033[1Akmeans\t: {limit}{' '*10}")
    
    with open(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/classes_{fold}.txt", mode="w") as f:
        for x_val, y_val, z_val in zip(gravity_points[:, 0], gravity_points[:, 1], gravity_points[:, 2]):
            f.write(f"{x_val},{y_val},{z_val}\n")
            
    ## 2次元直交座標
    fig = plt.figure()
    ax = fig.add_subplot(projection="rectilinear")
    ax.scatter(gravity_points[:, 0], gravity_points[:, 1], c="#ff2222", s=20)
    ax.scatter(x, y, c=classes, s=5)
    plt.savefig(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/classes_{fold}.png")
    plt.cla()
    
    ## 3次元直交座標
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(gravity_points[:, 0], gravity_points[:, 1], gravity_points[:, 2], c="#ff2222", s=20)
    ax.scatter(x, y, z, c=classes, s=5)
    plt.savefig(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/classes_{fold}_3d.png")
    plt.cla()
        
    return gravity_points, x_m, y_m, z_m


class UmapClasses:
    """
    @機能：パラメータ保存用
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


def save_params(mapper, gravity, fold, maxes, fold_keys):
    with open(f"{SAVE_DIR}/{MODEL}_{END_LAYER}/umap_params{fold}.pickle", mode="wb") as f:
        pickle.dump(UmapClasses(mapper, gravity, fold, maxes, fold_keys), f)
    
    
def main():
    for i, (mapper, embedding, ipts, length, fold_keys) in enumerate(execute_umap()):
        draw_embedding(embedding, ipts, FOLD[i])
        gravity, x, y, z = kmeans(embedding, FOLD[i])
        print(gravity)
        
        save_params(mapper, gravity, FOLD[i], (x, y, z), fold_keys)
        
        print(f"iteration : {i+1} / {5**(FOLD != 'all') if (FOLD in ['each', 'all']) else len(FOLD)}")
        if (length-1 == i): break

main()
