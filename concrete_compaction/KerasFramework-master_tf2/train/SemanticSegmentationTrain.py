print("Train")


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.distribute import MirroredStrategy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import sys
import traceback

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from BetweenclassLearning import BCL
from line_sender import send_master
from my_loss_function import MyLosses


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")

## ================ config ===================


## make_learning_plan.pyを使ってください。
## 直接実行した場合は強制終了します。
## 詳しくはREADMEを読んでください。


#### ディレクトリの設定など(実行環境が変わった時以外、いじらない)
WORKSPACE_DIR = "/workspace/semanticSegmentation"
DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
# DATASET_DIR = "/workspace/mesh_dataset"
FULLFRAME_DIR = "/workspace/Dataset/fullframe"
LOAD_WEIGHT_DIR = "/workspace/semanticSegmentation/result"
FULLFRAME_RESULT_DIR = "/workspace/fullframe/result/540x540"
CALL_BACK_DIR = "/workspace/osada_ws/cp"
# ONETIME_SAVE_DIR = "/workspace/osada_ws/onetime"
STATE_TEXT = "/workspace/osada_ws/state.txt"
AVERAGE_IMAGE_PATH = "/workspace/osada_ws/average_image_0516.png"
IS_OUTPUT_LOG = True
IS_USE_COMMAND_INPUT = True
IS_HIDE_SUMMARY = True
SUDO = False
IS_FOURCLASS = False
IS_AFTERCLASS = False
####


#### 重みの読み込に関する設定
IS_LOAD_WEIGHT = True
LOAD_WEIGHT_ID = "unet_20220302_b8_e10_fold1"
# LOAD_WEIGHT_ID = "unet_fresh_20220309_b8_e10_fold1"
IS_LOAD_FULLFRAME_WEIGHT = False
OUTPUT_LAYER_NAME = ""
####


#### エポック数の自動探索に関する設定
## 探索するエポック数の上限
EPOCHS_LIMIT = 30
AUTO_TRAIN_EPOCHS = 1
## 探索するエポック数の下限
EPOCHS_UNDER_LIMIT = 0
## 検証精度の上限
ACCURACY_SUP = 95
####


#### 学習の基本設定達
BATCH_SIZE = 2
## EPOCH = "auto" でエポック数の自動探索の有効化。数値ならその数値通りのエポック数で動く
EPOCHS = "auto"
## k分割交差検証の番号(1~k)
FOLD = 1
## モデルの名前
MODEL_NAME = "unet"
## メッシュ領域画像のサイズ
SIZE = [270, 270]
## 正解ラベルに関する設定
NUM_CLASSES = 2
CLASSES = {"before-just" : ["before", "just"],
           "fourclasses" : ["before", "b-before", "b-just", "just"],
           "before-just-after" : ["before", "just", "after"]}
## 入力画像をグレースケールにするか否か
IS_GRAYSCALE = False
COLOR_TYPE = "rgb"
NORMALIZATION = "default"
## 入力画像から平均画像を引くか否か
IS_USE_AVERAGE_IMAGE = True
## 損失関数
# LOSS = "kullback_leibler_divergence"
LOSS = None

## 乱数の初期値
RANDOM_SEED = 1
## 使用する画像枚数の上限
## LIMIT : None or int
LIMIT = None

## マルチGPUにするか否か(入力の分割のみ)
USE_MULTI_GPU = False
## 入力画像をリサイズするか否か(上述の配列SIZEの値にリサイズ)
IS_RESIZE = False

## 保存先の指定
IS_SAVE = True
# SAVE_PATH = "/workspace/semanticSegmentation/result"
SAVE_PATH = "/workspace/fullframe/result/540x540"
SAVE_ID = "missing"

## フルフレームで学習させるか否か
IS_FULLFRAME = True
## 画像サイズの設定 : [a, b] -> 縦/a, 横/b
RESIZE_COEF = [2, 3]
# RESIZE_COEF = [4, 6]

## 全ての画像サイズを指定したサイズに強制的にリサイズを行うか否か(あまり必要ない)
IS_ABSOLUTE_RESIZE = False
## リサイズするサイズ(あまり必要ない)
ABSOLUTE_SIZE = [256, 256]

## フレッシュ性状データを入力するか否か
IS_INCLUDE_FRESH = False
## フレッシュ性状データのカーネルサイズ(Encoderの出力部分と同じにする。"auto"を推奨)
FRESH_KERNEL_SIZE = "auto"

## Between-class Learningを適用するか否か
IS_USE_BCL = False

## Metric Learningを適用するか否か
IS_USE_METRIC = False
## 適用する関数の定義名
METRIC_FUNC = "sphereface"

## 輝度の拡張をするか否か
IS_EXTEND_LUMINANCE = False
## 
# LUMINANCE_THRESHOLD = 0
####

## クラス分類の設定
# CLASSIFICATION = "before-just"
CLASSIFICATION = "fourclasses"
MULTI_LOSSES = False


## ===========================================


## エポック数の自動探索をするか否かの条件分岐
IS_AUTO_LEARNING = EPOCHS == "auto"
 
## コマンドライン引数の受け付け
if IS_USE_COMMAND_INPUT:
    args = sys.argv
    if (len(args) > 1):
        print("args :")
        for i, a in enumerate(args):
            print(f" > {i:02d}\t{a}")
        if (args[1] == "auto"):
            ACCURACY_SUP = float(args[2])
            IS_AUTO_LEARNING = True
        else:
            EPOCHS = int(args[1])
            # BATCH_SIZE = int(args[2])
            IS_AUTO_LEARNING = False
        FOLD = int(args[3])
        MODEL_NAME = args[4]
        SAVE_PATH = args[5]
        SAVE_ID = args[6]
        IS_FULLFRAME = int(args[7])
        IS_INCLUDE_FRESH = int(args[8])
        IS_LOAD_WEIGHT = int(args[9])
        LOAD_WEIGHT_ID = args[10]
        IS_EXTEND_LUMINANCE = int(args[11])
        IS_GRAYSCALE = False#int(args[12])        # remove
        IS_USE_BCL = int(args[13])
        LOSS = args[14]
        OPTIMIZER = args[15]
        IS_H5 = int(args[16])
        IS_USE_METRIC = int(args[17])
        METRIC_FUNC = args[18]
        IS_LOAD_FULLFRAME_WEIGHT = int(args[19])
        OUTPUT_LAYER_NAME = args[20]
        IS_FUSION_FACE = int(args[21])
        NULLFICATION_METRIC = int(args[22])
        DROPOUT_CONST = float(args[23])
        LABEL_SMOOTHING = float(args[24])
        NORM = args[25]
        USE_ATTENTION = int(args[26])
        CLASSIFICATION = args[27]
        MULTI_LOSSES = int(args[28])
        FOURCLASSES_TYPE = args[29]
        EUNET_METRIC_MODE = args[30]
        EUNET_METRIC_SUBCONTEXT = args[31]
        COLOR_TYPE = args[32]
        NORMALIZATION = args[33]
        USE_AE_INPUT = int(args[34])
        NOISE_TYPE = args[35]
        AUTOENCODER_LOSS = args[36]
        AE_MODEL_ID = args[37]
        IS_FLIP = int(args[38])
        FLIP_LIST = list(map(int, args[39].split(",")))
        IS_ROTATE = int(args[40])
        IS_ENLARGE = int(args[41])
        REDUCE_CONST = float(args[42])
        LEARNING_RATE = float(args[43])
        ROTATE_RATE = float(args[44])
        ROTATE_DEGREES = list(map(lambda x: list(map(int, x.split(","))), args[45].split("-")))
        ALL_IN_ONE = int(args[46])
        REDUCE = args[47]
    
        cp.cprint("@ Accepted command argments.", "orange")
else:
    ## 直接このプログラムを実行した場合は強制的にプロセス終了
    cp.cprint("Your method of execution is deprecated.", "red")
    cp.cprint("If you have read README.md, I recommend using \"make_learning_plan.py\".", "red")
    if not SUDO: exit()
    
im.fold = FOLD
if USE_AE_INPUT:
    im.AE_model_id = AE_MODEL_ID

## クラス分類の設定
IS_AFTERCLASS = CLASSIFICATION == "before-just-after"
IS_FOURCLASS = CLASSIFICATION == "fourclasses" and not IS_AFTERCLASS
NUM_CLASSES += 2*IS_FOURCLASS + IS_AFTERCLASS
IS_MYSELFCLASSES = FOURCLASSES_TYPE == "myself"
IS_USE_RECTIFIED_DATASET = FOURCLASSES_TYPE == "rectified" and IS_FULLFRAME

## 損失関数の設定
if (LOSS == "iou"):
    LOSS = MyLosses.iou_loss
elif (LOSS == "cross_entropy_iou"):
    LOSS = MyLosses.crossentropy_iou_loss
elif (LOSS == "cross_entropy_ssim"):
    LOSS = MyLosses.crossentropy_ssim_loss

## メッシュ領域ごとの学習に関する設定
if not IS_FULLFRAME:
    BATCH_SIZE = 8
    EPOCHS_LIMIT = 10 #10
    EPOCHS_UNDER_LIMIT = 0 #5
    
if ("e-unet" in MODEL_NAME):
    # BATCH_SIZE = 4 if IS_FULLFRAME else 48
    BATCH_SIZE = 6 if IS_FULLFRAME else 32
    EPOCHS_LIMIT = 30 if IS_FULLFRAME else 3
    EPOCHS_UNDER_LIMIT = 0 #5
    if MULTI_LOSSES:
        BATCH_SIZE = 3 if IS_FULLFRAME else 40
        # BATCH_SIZE = 4 if IS_FULLFRAME else 32
        EPOCHS_LIMIT = 20 if IS_FULLFRAME else 3
        EPOCHS_UNDER_LIMIT = 0 #5
    if not IS_AUTO_LEARNING:
        EPOCHS_LIMIT = EPOCHS
    # if IS_FOURCLASS:
        # BATCH_SIZE = 2 if IS_FULLFRAME else 10
        # EPOCHS_LIMIT = 1 if IS_FULLFRAME else 1
        # EPOCHS_UNDER_LIMIT = 0 #5

if ("espnet" in MODEL_NAME):
    # BATCH_SIZE = 4 if IS_FULLFRAME else 48
    BATCH_SIZE = 4 if IS_FULLFRAME else 32
    EPOCHS_LIMIT = 30 if IS_FULLFRAME else 10
    EPOCHS_UNDER_LIMIT = 0 #5

## 大いなる力
if IS_AFTERCLASS:
    EPOCHS_LIMIT = 5
elif ALL_IN_ONE:
    EPOCHS_LIMIT = 7
    BATCH_SIZE = 6
else:
    # EPOCHS_LIMIT = 30
    # BATCH_SIZE = 6
    pass

## 距離学習に関する設定
if IS_USE_METRIC:
    EPOCHS_UNDER_LIMIT = 0
    if ("classifier" in MODEL_NAME):
        IS_USE_METRIC = False
    elif ("e-unet" in MODEL_NAME):
        BATCH_SIZE = 4 if IS_FULLFRAME else 48
        EPOCHS_LIMIT = 20 if IS_FULLFRAME else 10
        EPOCHS_UNDER_LIMIT = 0 #5
    
## 入力画像サイズの調整
if ("e-unet" in MODEL_NAME): # or ("espnet" in MODEL_NAME):
    if IS_FULLFRAME:
        DATASET_DIR = FULLFRAME_DIR
        SIZE = [32*3*6, 32*3*6] # 576
    else:
        SIZE = [32*3*3, 32*3*3] # 280
        
elif ("espnet" in MODEL_NAME):
    if IS_FULLFRAME:
        DATASET_DIR = FULLFRAME_DIR
        # SIZE = [32*3*6, 32*3*6] # 576
        SIZE = [128, 128]
        # SIZE = [512, 512]
    else:
        # SIZE = [32*3*3, 32*3*3] # 280
        SIZE = [64, 64]
        # SIZE = [256, 256]
        
elif IS_FULLFRAME:
    DATASET_DIR = FULLFRAME_DIR
    if IS_ABSOLUTE_RESIZE:
        SIZE = ABSOLUTE_SIZE.copy()
    else:
        SIZE[0] = int(SIZE[0]*4/RESIZE_COEF[0])
        SIZE[1] = int(SIZE[1]*6/RESIZE_COEF[1])
elif IS_ABSOLUTE_RESIZE:
    SIZE = ABSOLUTE_SIZE.copy()

## 画像の形式に関する設定
IS_GRAYSCALE = COLOR_TYPE == "grayscale"

## 保存するフォルダ名・ファイル名の定義
if IS_AUTO_LEARNING:
    SAVE_ID = SAVE_ID.replace("_AutoLearning", "")
    SAVE_ID = f"{MODEL_NAME}{'_after' if IS_AFTERCLASS else ''}{'_4class' if IS_FOURCLASS else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}_{SAVE_ID}_AutoLearning_fold{FOLD}{f'_{SIZE[0]}x{SIZE[1]}' if IS_FULLFRAME else ''}{'_luminance' if IS_EXTEND_LUMINANCE else ''}{'_multiGPU' if USE_MULTI_GPU else ''}"
else:
    SAVE_ID = f"{MODEL_NAME}{'_after' if IS_AFTERCLASS else ''}{'_4class' if IS_FOURCLASS else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}_{SAVE_ID}_b{BATCH_SIZE}_e{EPOCHS}_fold{FOLD}{f'_{SIZE[0]}x{SIZE[1]}' if IS_FULLFRAME else ''}{'_luminance' if IS_EXTEND_LUMINANCE else ''}{'_multiGPU' if USE_MULTI_GPU else ''}"

## fullframeの重みを読み込む設定
if IS_LOAD_FULLFRAME_WEIGHT:
    LOAD_WEIGHT_DIR = FULLFRAME_RESULT_DIR

## 作業ディレクトリ
directory = lambda target : f"{DATASET_DIR}/{target}"
## 読み込む画像のパスが書かれたテキストファイル
path = lambda target : f"{DATASET_DIR}/text_dataset/fold{FOLD}/{target}{'_include_fresh' if IS_INCLUDE_FRESH else ''}{'_luminance' if IS_EXTEND_LUMINANCE else ''}{'_after_wide_argment' if IS_AFTERCLASS else ''}{'_4class' if IS_FOURCLASS else ''}{'_myself' if IS_MYSELFCLASSES else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}.txt"
path_multilosses = lambda target : f"{DATASET_DIR}/text_dataset/fold{FOLD}/{target[0]}{'_include_fresh' if IS_INCLUDE_FRESH else ''}{'_luminance' if IS_EXTEND_LUMINANCE else ''}{'_after_wide_argment' if IS_AFTERCLASS else ''}{'_4class' if target[1] else ''}{'_myself' if IS_MYSELFCLASSES else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}.txt"
## エラー表示
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")

## データ数の調整
if LIMIT is None:
    with open(path("train")) as f:
        data_length = len(f.readlines())
        # if IS_EXTEND_LUMINANCE: data_length //= 10
        LIMIT = data_length - (data_length%BATCH_SIZE)

## フレッシュ性状データのカーネルサイズの調節
if isinstance(FRESH_KERNEL_SIZE, str):
    if (FRESH_KERNEL_SIZE == "auto"):
        FRESH_KERNEL_SIZE = list(map(lambda x: x//16, SIZE))
    else:
        cp.cprint("\"FRESH_KERNEL_SIZE\" must be either [int, int] or \"auto\".")
        LIMIT = -1

## ログファイルの新規作成
if IS_OUTPUT_LOG:
    Utils.makedir(f"{SAVE_PATH}/{SAVE_ID}")
    with open(f"{SAVE_PATH}/{SAVE_ID}/log.txt", mode="w") as logoutput: pass


def output_state(color:str, string):
    """
    @機能：ステータスファイルに文字列を出力(get_state.pyでこの文字列を取得)
    @引数：color:str = 文字色, string = 出力する文字列
    @戻値：None
    """
    with open(STATE_TEXT, "w") as f:
        f.write(f"{color}\n{string}")


def dataGenerator(resourcepath:str, without_mixup:bool=False):
    """
    @機能：データセットの画像とラベルを学習用に整形
    @引数：resourcepath = 画像のパスが列挙されたテキストファイルのパス, without_mixup = Mixupを"行わない"か否か
    @戻値：画像(shape : *SIZE, 3)、ラベルのone-hot(shape : *SIZE, 2)
    """
    
    ## テキストファイルから、データの読み取り
    if IS_USE_BCL and not without_mixup:
        image_path, mask_path, fresh_data, luminance_data, mix_path = BCL.data_division(
            resourcepath=resourcepath,
            random_seed=RANDOM_SEED,
            limit=LIMIT,
            is_include_fresh=IS_INCLUDE_FRESH,
            is_extend_luminance=IS_EXTEND_LUMINANCE
            )
    else:
        image_path, mask_path, mask_path_multilosses, fresh_data, luminance_data = [], [], [], [], []
        with open(resourcepath) as f:
            readlines = f.readlines()
            random.seed(RANDOM_SEED)
            random.shuffle(readlines)
            if LIMIT: readlines = readlines[:LIMIT]
        for line in map(lambda x: x.rstrip("\n"), readlines):
            linebuffer = line.split(" ")    # linebuffer = [コンクリ画像のパス, マスク画像のパス]
            image_path.append(linebuffer[0])
            mask_path.append(linebuffer[1])
            if IS_INCLUDE_FRESH:
                fresh_data.append(list(map(float, linebuffer[2:])))
            if IS_EXTEND_LUMINANCE:
                luminance_data.append(float(linebuffer[-1]))
    length = len(image_path)
    data_gen_args = dict(
        rescale=None
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
    mask_dataframe = pd.DataFrame(mask_path, index=None, columns=["mask"])

    # size = (576, 576) if USE_AE_INPUT else SIZE.copy()
    size = SIZE.copy()

    image_generator = image_datagen.flow_from_dataframe(image_dataframe,
                                                        x_col="image",
                                                        target_size=size,
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        # classes=CLASSES[CLASSIFICATION].copy(),
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=size,
                                                      color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                    #   classes=CLASSES[CLASSIFICATION].copy(),
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)
    if IS_USE_BCL and not without_mixup:
        mix_datagen = ImageDataGenerator(**data_gen_args)
        mix_dataframe = pd.DataFrame(mix_path, index=None, columns=["mix"])
        mix_generator = mix_datagen.flow_from_dataframe(mix_dataframe,
                                                        x_col="mix",
                                                        target_size=size,
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        # classes=CLASSES[CLASSIFICATION].copy(),
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    
    ## 画像を一枚ずつ取り出し、出力
    fresh_index = 0
    if not IS_USE_BCL or without_mixup:
        for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
            image, mask = im.adjust_data(image, mask, IS_FULLFRAME, IS_ABSOLUTE_RESIZE,
                                        IS_USE_AVERAGE_IMAGE, SIZE, IS_GRAYSCALE, NUM_CLASSES,
                                        AVERAGE_IMAGE_PATH, classification=CLASSIFICATION,
                                        color_type=COLOR_TYPE, normalization=NORMALIZATION,
                                        use_AE_input=USE_AE_INPUT, noise_type=NOISE_TYPE,
                                        autoencoder_loss=AUTOENCODER_LOSS, is_flip=IS_FLIP, flip_list=FLIP_LIST,
                                        is_rotate=IS_ROTATE, rotate_rate=ROTATE_RATE, rotate_degrees=ROTATE_DEGREES,
                                        is_enlarge=IS_ENLARGE)
            
            ## 輝度の拡張(古い実装。使わない)
            if IS_EXTEND_LUMINANCE:
                pass
                # image *= luminance_data[i]
                # img = np.clip(img, 0, 1)
            
            ## フレッシュ性状データの入力あり
            if IS_INCLUDE_FRESH:
                fresh_buffer = []
                ## フレッシュ性状データをバッチサイズごとに切り出し
                for index in range(fresh_index, fresh_index+BATCH_SIZE):
                    if (index < length): fresh_buffer.append(fresh_data[index])
                    else: break
                if len(fresh_buffer):
                    fresh_index += BATCH_SIZE
                else:
                    fresh_index = 0
                    fresh_buffer = fresh_data[0:BATCH_SIZE]
                ## フレッシュ性状データを学習用に整形
                fresh = Calc.make_fresh_tensor(fresh=fresh_buffer,
                                            kernel_size=FRESH_KERNEL_SIZE.copy(),
                                            batch_size=len(fresh_buffer),
                                            isStr=False)
                yield [image, fresh], mask
            
            ## Metric Learning
            if IS_USE_METRIC:
                yield [image, mask], mask
            
            ## フレッシュ性状データの入力なし
            else:
                yield image, mask
    
    else:
        for i, (image, mask, mix) in enumerate(zip(image_generator, mask_generator, mix_generator)):
            image, mask = im.adjust_data(image, mask, IS_FULLFRAME, IS_ABSOLUTE_RESIZE,
                                         IS_USE_AVERAGE_IMAGE, SIZE, IS_GRAYSCALE, NUM_CLASSES,
                                         AVERAGE_IMAGE_PATH, IS_USE_BCL, mix)
            
            ## 輝度の拡張(古い実装。使わない)
            if IS_EXTEND_LUMINANCE:
                pass
                # image *= luminance_data[i]
                # img = np.clip(img, 0, 1)
            
            ## フレッシュ性状データの入力あり
            if IS_INCLUDE_FRESH:
                fresh_buffer = []
                ## フレッシュ性状データをバッチサイズごとに切り出し
                for index in range(fresh_index, fresh_index+BATCH_SIZE):
                    if (index < length): fresh_buffer.append(fresh_data[index])
                    else: break
                if len(fresh_buffer):
                    fresh_index += BATCH_SIZE
                else:
                    fresh_index = 0
                    fresh_buffer = fresh_data[0:BATCH_SIZE]
                ## フレッシュ性状データを学習用に整形
                fresh = Calc.make_fresh_tensor(fresh=fresh_buffer,
                                               kernel_size=FRESH_KERNEL_SIZE.copy(),
                                               batch_size=len(fresh_buffer),
                                               isStr=False)
                yield [image, fresh], mask
                
            ## Metric Learning
            if IS_USE_METRIC:
                yield [image, mask], mask
                
            ## フレッシュ性状データの入力なし
            else:
                yield image, mask
                
                
def dataGenerator_multiloss(target:str):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    ## テキストファイルから、データの読み取り
    image_path, mask_path = [], []
    with open(path_multilosses([target, 1])) as f:
        readlines = f.readlines()
        random.seed(RANDOM_SEED)
        random.shuffle(readlines)
        if LIMIT: readlines = readlines[:LIMIT]
    for line in map(lambda x: x.rstrip("\n"), readlines):
        linebuffer = line.split(" ")    # linebuffer = [コンクリ画像のパス, マスク画像のパス]
        image_path.append(linebuffer[0])
        mask_path.append(linebuffer[1])
    data_gen_args = dict(
        rescale=None
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # mask_ml_datagen = ImageDataGenerator(**data_gen_args)

    image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
    mask_dataframe = pd.DataFrame(mask_path, index=None, columns=["mask"])
    # mask_ml_dataframe = pd.DataFrame(mask_ml_path, index=None, columns=["mask_ml"])

    image_generator = image_datagen.flow_from_dataframe(image_dataframe,
                                                        x_col="image",
                                                        target_size=SIZE.copy(),
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        classes=CLASSES[CLASSIFICATION].copy(),
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                      classes=CLASSES[CLASSIFICATION].copy(),
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)
    # mask_ml_generator = mask_ml_datagen.flow_from_dataframe(mask_ml_dataframe,
    #                                                   x_col="mask_ml",
    #                                                   target_size=SIZE.copy(),
    #                                                   color_mode="grayscale" if IS_GRAYSCALE else "rgb",
    #                                                   classes=CLASSES[CLASSIFICATION].copy(),
    #                                                   class_mode=None,
    #                                                   batch_size=BATCH_SIZE,
    #                                                   shuffle=False)
    
    ## 画像を一枚ずつ取り出し、出力
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        image, mask, mask_ml = im.adjust_data_multilosses(image, mask, IS_FULLFRAME,
                                                          IS_USE_AVERAGE_IMAGE, SIZE, IS_GRAYSCALE,
                                                          AVERAGE_IMAGE_PATH)
        yield image, [mask, mask_ml]
    

def callbacks(cp_dir):
    """
    @機能：コールバックの定義
    @引数：( *入力不要 cp_dir = コールバックを出力するディレクトリのパス )
    @戻値：コールバックの定義が格納されたリスト
    """
    
    cp_dir = f"{cp_dir}/cp"
    Utils.makedir(cp_dir)
    cp_path = f"{cp_dir}/{SAVE_ID}_" + "{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5"
    # cp_path = f"{cp_dir}/" + SAVE_ID + "_cp.h5"
    callbacks_list = [
        ModelCheckpoint(
            filepath=cp_path,
            monitor="val_loss",
            save_best_only=True)]
    
    return callbacks_list
    

def write_graph(history, validation=False):
    """
    @機能：AcuuracyとLossの推移についてのグラフの描画
    @引数：history = Model.history.history, validation = 検証データに対する結果も描画するか否か
    @戻値：None
    """
    
    ## Accuracyの推移を描画
    plt.figure(figsize=(6,4))
    plt.plot(history.history['acc'], label='traindata')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if validation:
        plt.plot(history.history['val_acc'], label='valdata')
        plt.legend(loc='upper left')
    else:
        plt.legend(['traindata'], loc='upper left')
    plt.savefig(f"{SAVE_PATH}/{SAVE_ID}/{SAVE_ID}_accuracy.png")
    
    ## Lossの推移を描画
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='traindata')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validation:
        plt.plot(history.history['val_loss'], label='valdata')
        plt.legend(loc='upper left')
    else:
        plt.legend(['traindata'], loc='upper left')
    plt.savefig(f"{SAVE_PATH}/{SAVE_ID}/{SAVE_ID}_loss.png")
        

def train(model_name:str=None, sudo:bool=False):
    """
    @機能：モデルの訓練
    @引数：model_name = モデルの定義名
    @戻値：学習の数値推移の一覧(辞書型)
    """
    
    cp.cprint("This method is deplecated to use.", "red")
    cp.cprint("Please use \"auto_train()\" method.", "red")
    
    # if not sudo: return
    
    if model_name is None:
        blank("model_name")
        return None

    ## 複数のGPUを使用する場合
    # if USE_MULTI_GPU:
    #     strategy = MirroredStrategy()
    #     with strategy.scope():
    #         ## with文内は、一つのGPUに対する操作
    #         model = mm.create_model(model_name, f"{LOAD_WEIGHT_DIR}/{LOAD_WEIGHT_ID}/{LOAD_WEIGHT_ID}.h5", SAVE_ID, SIZE,
    #                                 IS_GRAYSCALE, FRESH_KERNEL_SIZE, IS_LOAD_WEIGHT, IS_USE_BCL, LOSS, IS_FUSION_FACE)
    #         if not IS_HIDE_SUMMARY: model.summary()
    #         train_gen = dataGenerator(path("train"))
    #         validation_gen = dataGenerator(path("validation"), True)
    #         callbacks_list = callbacks(directory("cp"))
    #         datacount = Utils.datacounter(path("train"), LIMIT)
    #         history = model.fit_generator(
    #             generator=train_gen,
    #             steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
    #             epochs=EPOCHS,
    #             validation_data=validation_gen,
    #             validation_steps=100,
    #             shuffle=False,
    #             # callbacks=callbacks_list
    #         )
    
    ## GPUをひとつだけ使用する場合
    # else:
    model = mm.create_model(model_name=model_name,
                            weights=f"{LOAD_WEIGHT_DIR}/{LOAD_WEIGHT_ID}/{LOAD_WEIGHT_ID}.h5",
                            save_id=SAVE_ID,
                            size=SIZE,
                            is_grayscale=IS_GRAYSCALE,
                            fresh_kernel_size=FRESH_KERNEL_SIZE,
                            is_load_weight=IS_LOAD_WEIGHT,
                            is_use_bcl=IS_USE_BCL,
                            loss=LOSS,
                            is_fusion_face=IS_FUSION_FACE,
                            metric_func=METRIC_FUNC,
                            nullfication_metric=NULLFICATION_METRIC,
                            is_h5=IS_H5,
                            dropout_const=DROPOUT_CONST,
                            label_smoothing=LABEL_SMOOTHING,
                            norm=NORM,
                            use_attention=USE_ATTENTION,
                            classification=CLASSIFICATION,
                            optimizer=OPTIMIZER,
                            multi_losses=MULTI_LOSSES,
                            eunet_metric_mode=EUNET_METRIC_MODE,
                            eunet_metric_subcontext=EUNET_METRIC_SUBCONTEXT,
                            is_HoG=COLOR_TYPE=="hog")
    if not IS_HIDE_SUMMARY: model.summary()
    train_gen = dataGenerator(path("train"))
    validation_gen = dataGenerator(path("validation"), True)
    datacount = Utils.datacounter(path("train"), LIMIT)
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=100,
        shuffle=False,
    )
    
    if IS_OUTPUT_LOG: Utils.log(f"{SAVE_PATH}/{SAVE_ID}", history.history)
    
    ## モデル・重みの保存    
    Utils.makedir(f"{SAVE_PATH}/{SAVE_ID}")
    model.save(f"{SAVE_PATH}/{SAVE_ID}")
    # model.save_weights(f"{SAVE_PATH}/{SAVE_ID}/{SAVE_ID}_weights.h5", True)
    # write_graph(history, True)
    
    return history.history
    

def auto_train(model_name:str=None):
    """
    @機能：モデルの訓練(適切なエポック数などの探索をするモード)
    @引数：model_name = モデルの定義名
    @戻値：検証精度の最高値, 実行したエポック数
    """
    
    if model_name is None:
        blank("model_name")
        return None

    accuracy_max = 0
    loss_min = 1e10
    is_beyoud = False
    
    ## 複数のGPUを使用する場合
    # if USE_MULTI_GPU:
    #     strategy = MirroredStrategy()
    #     with strategy.scope():
    #         ## with文内は、一つのGPUに対する操作
    #         model = mm.create_model(model_name, f"{LOAD_WEIGHT_DIR}/{LOAD_WEIGHT_ID}/{LOAD_WEIGHT_ID}.h5", SAVE_ID, SIZE,
    #                                 IS_GRAYSCALE, FRESH_KERNEL_SIZE, IS_LOAD_WEIGHT, IS_USE_BCL, LOSS, IS_FUSION_FACE)
    #         if not IS_HIDE_SUMMARY: model.summary()
    #         for itr in range(EPOCHS_LIMIT):
    #             cp.cprint(f"auto learing | iteration : {itr+1} | accurasy : {accuracy_max}", "green")
    #             train_gen = dataGenerator(path("train"))
    #             validation_gen = dataGenerator(path("validation"), True)
    #             callbacks_list = callbacks(f"{SAVE_PATH}/{SAVE_ID}")
    #             datacount = Utils.datacounter(path("train"), LIMIT)
    #             history = model.fit_generator(
    #                 generator=train_gen,
    #                 steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
    #                 epochs=1,
    #                 validation_data=validation_gen,
    #                 validation_steps=100,
    #                 shuffle=False,
    #                 callbacks=callbacks_list
    #             )
    #             if IS_OUTPUT_LOG: Utils.log(f"{SAVE_PATH}/{SAVE_ID}", history.history)
                
    #             try:
    #                 acc = history.history[f"val_{OUTPUT_LAYER_NAME}_acc"][-1]*100
    #             except:
    #                 acc = history.history["val_acc"][-1]*100
    #             ## 精度が更新される場合、モデルの保存を行う
    #             if ((acc >= accuracy_max) and (itr >= EPOCHS_UNDER_LIMIT-1)):
    #                 accuracy_max = acc
    #                 model.save(f"{SAVE_PATH}/{SAVE_ID}")
    #                 # model.save_weights(f"{SAVE_PATH}/{SAVE_ID}/{SAVE_ID}_weights.h5", True)
    #                 if (accuracy_max >= ACCURACY_SUP):
    #                     cp.cprint(f"@ The accuracy exceeded ACCURACY_SUP. | iteration : {itr+1} | accurasy : {accuracy_max}", "cyan")
    #                     is_beyoud = True
    #                     break
    
    ## GPUをひとつだけ使用する場合
    # else:
    model = mm.create_model(model_name=model_name,
                            weights=f"{LOAD_WEIGHT_DIR}/{LOAD_WEIGHT_ID}/{LOAD_WEIGHT_ID}.h5",
                            save_id=SAVE_ID,
                            size=SIZE,
                            is_grayscale=IS_GRAYSCALE,
                            fresh_kernel_size=FRESH_KERNEL_SIZE,
                            is_load_weight=IS_LOAD_WEIGHT,
                            is_use_bcl=IS_USE_BCL,
                            loss=LOSS,
                            is_fusion_face=IS_FUSION_FACE,
                            metric_func=METRIC_FUNC,
                            nullfication_metric=NULLFICATION_METRIC,
                            is_h5=IS_H5,
                            dropout_const=DROPOUT_CONST,
                            label_smoothing=LABEL_SMOOTHING,
                            norm=NORM,
                            use_attention=USE_ATTENTION,
                            classification=CLASSIFICATION,
                            optimizer=OPTIMIZER,
                            multi_losses=MULTI_LOSSES,
                            eunet_metric_mode=EUNET_METRIC_MODE,
                            eunet_metric_subcontext=EUNET_METRIC_SUBCONTEXT,
                            is_HoG=COLOR_TYPE=="hog",
                            reduce_const=REDUCE_CONST,
                            learning_rate=LEARNING_RATE,
                            fold=FOLD,
                            reduce=REDUCE,)
    if not IS_HIDE_SUMMARY: model.summary()
    for itr in range(EPOCHS_LIMIT):
        if IS_USE_METRIC:
            cp.cprint(f"auto learing (DeepML) | iteration : {itr+1} / {EPOCHS_LIMIT} | loss : {loss_min}", "green")
        else:
            cp.cprint(f"auto learing | iteration : {itr+1} / {EPOCHS_LIMIT} | accurasy : {accuracy_max}", "green")
        if MULTI_LOSSES:
            train_gen = dataGenerator_multiloss("train")
            validation_gen = dataGenerator_multiloss("validation")
            datacount = Utils.datacounter(path_multilosses(["train", 0]), LIMIT)
        else:
            train_gen = dataGenerator(path("train"))
            validation_gen = dataGenerator(path("validation"), True)
            datacount = Utils.datacounter(path("train"), LIMIT)
        # callbacks_list = callbacks(f"{SAVE_PATH}/{SAVE_ID}")
        history = model.fit_generator(
            generator=train_gen,
            steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
            epochs=AUTO_TRAIN_EPOCHS,
            validation_data=validation_gen,
            validation_steps=100,
            shuffle=False,
            # callbacks=callbacks_list
        )
        if IS_OUTPUT_LOG: Utils.log(f"{SAVE_PATH}/{SAVE_ID}", history.history)
        
        ## 距離学習のときは損失を元に更新する
        if IS_USE_METRIC:
            try:
                loss = history.history[f"val_face_layer_loss"][-1]
            except:
                loss = history.history["val_loss"][-1]
            ## 精度が更新される場合、モデルの保存を行う
            if (loss <= loss_min):
                cp.cprint("@ Update loss.", "orange")
                loss_min = loss
                model.save(f"{SAVE_PATH}/{SAVE_ID}")
                
        ## それ以外の時は正解率を元に更新する
        else:
            try:
                acc = history.history[f"val_{OUTPUT_LAYER_NAME}_acc"][-1]*100
            except:
                acc = history.history["val_acc"][-1]*100
            ## 精度が更新される場合、モデルの保存を行う
            if ((acc >= accuracy_max) and (itr >= EPOCHS_UNDER_LIMIT-1)):
                cp.cprint("@ Update accuracy.", "orange")
                accuracy_max = acc
                model.save(f"{SAVE_PATH}/{SAVE_ID}")
                if (accuracy_max >= ACCURACY_SUP):
                    cp.cprint(f"@ The accuracy exceeded ACCURACY_SUP. | iteration : {itr+1} | accurasy : {accuracy_max}", "cyan")
                    is_beyoud = True
                    break
    
    if IS_USE_METRIC:    
        final = loss_min
    else:
        final = accuracy_max
    if not is_beyoud: 
        if IS_USE_METRIC:
            cp.cprint(f"@ This learing has come to the end. | loss : {loss_min}", "orange")
        else:
            cp.cprint(f"@ This learing did not exceed ACCURAY_SUP. | max accuracy : {accuracy_max}", "orange")
    
    return final, itr+1


def main():
    if (LIMIT < 0):
        cp.cprint("[!] \"LIMIT\" is too short or \"BATCH_SIZE\" is too large", "red")
    else:
        if MODEL_NAME is None:
            blank("MODEL_NAME")
        else:
            if IS_FULLFRAME: cp.cprint("@ Execute fullframe prediction.", "orange")
            if IS_INCLUDE_FRESH: cp.cprint("@ Include fresh data.", "orange")
            if IS_GRAYSCALE: cp.cprint("@ Execute grayscale operation.", "orange")
            if IS_USE_BCL: cp.cprint("@ Execute Between-class Mixup.", "orange")
            if IS_USE_METRIC: cp.cprint("@ Execute Metric Learning.", "orange")
            if IS_FLIP: cp.cprint("@ Execute image fliping.", "orange")
            if IS_ROTATE: cp.cprint("@ Execute image rotation.", "orange")
            if IS_ENLARGE: cp.cprint("@ Execute image enlargement.", "orange")
            if (REDUCE_CONST < 1): cp.cprint(f"@ Reduce type : {REDUCE}", "orange")
            cp.cprint(f"classification : {CLASSIFICATION}", "pink")
            cp.cprint(f"classes_type   : {FOURCLASSES_TYPE}", "pink")
            if IS_AUTO_LEARNING:
                result = auto_train(MODEL_NAME)
            else:
                result = train(MODEL_NAME, SUDO)
    
    if IS_AUTO_LEARNING:
        output_state("cyan", f"\"{SAVE_ID}\" AutoLearning finished.\nThe validation max accuracy is {round(result[0], 3)} [%]\n( {result[1]+1} epochs )")
    else:        
        output_state("cyan", f"\"{SAVE_ID}\" learning finished.\nThe validation accuracy is {round(((result['val_acc'])[-1])*100, 3)} [%]")
    
    # send_master.send(f"train: {SAVE_ID}\n{result}")
    
    cp.cprint("- finished ! -", "cyan")


if (__name__ == "__main__"):
    try:
        main()
    except Exception as e:
        # print(e)
        # message = getattr(e, 'message', str(e)).replace('\n', '\\n').replace('\r', '\\r')[:989]
        message = traceback.format_exc()
        send_master.send(f"""\
< Error Report >
save_id :
{SAVE_ID}

type :
{type(e)}

message :
{message}

error :
{e}\
""")
