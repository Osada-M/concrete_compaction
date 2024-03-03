print("Test")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random
from PIL import Image
import sys
import time
import matplotlib.pyplot as plt
import os

## my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
import MetricLearning_for_semseg as metric_semseg
from line_sender import send_master
from mymodel.SemSegLight import E_UNet
from luminance_extender import LuminanceExtender
from my_loss_function import MyLosses


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ===================


#### ディレクトリの設定など(実行環境が変わった時以外、いじらない)
# WORKSPACE_DIR = "/workspace/semanticSegmentation"
# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
WORKSPACE_DIR = "/workspace/fullframe"
DATASET_DIR = "/workspace/Dataset/fullframe"
STATE_TEXT = "/workspace/osada_ws/state.txt"
AVERAGE_IMAGE_PATH = '/workspace/osada_ws/average_image_0516.png'
IS_USE_COMMAND_INPUT = True
IS_HIDE_SUMMARY = 1
IS_FOURCLASSES = False
####


#### 学習の基本設定達
## BATCH_SIZEは1のまま
BATCH_SIZE = 1
FOLD = 2
## MODEL_NAME : "unet", "unet_fresh", "pspnet"
MODEL_NAME = "unet"
## 入力画像サイズ
SIZE = [270*2, 270*2]
# SIZE = [270, 270]
## 正解ラベルに関する設定
NUM_CLASSES = 2
CLASSES = {"before-just" : ["before", "just"],
           "fourclasses" : ["before", "b-before", "b-just", "just"],
        #    "before-just-after" : ["before", "just", "after"]}
           "before-just-after" : ["just", "after"],}
## 入力画像をグレースケールにするか否か
IS_GRAYSCALE = False
## 入力画像から平均画像を引くか否か
IS_USE_AVERAGE_IMAGE = True
## メッシュ領域ごとに正否判定を行うか否か
IS_JUDGE_BY_MESH = True
## 正誤判定に、BeforeJust入れ替わり時の閾値を設けるか否か
IS_USE_THRESHOLD = True
## 閾値(Just->Before, Before->Just)
THRESHOLD = [0.75, 0.75]

RANDOM_SEED = 1
## LIMIT : None or int
LIMIT = None
SKIP = 1

## マルチGPUにするか否か(入力の分割のみ)
USE_MULTI_GPU = False
## 入力画像をリサイズするか否か(上述の配列SIZEの値にリサイズ)
IS_RESIZE = True

## 結果を保存するか否か
IS_OUTPUT_RESULT = True
## 詳細なテスト結果(画像単位での、全ての推論結果)を保存するか否か(実行時間は３倍増程度)
IS_OUTPUT_DETAIL_RESULT = False

## 読み込むモデルの設定
# LOAD_PATH = "/workspace/semanticSegmentation/result"
LOAD_PATH = "/workspace/fullframe/result/540x540"
LOAD_ID = "unet_20220319-1_AutoLearning_fold2_540x540"
# LOAD_ID = "unet_20220302_prefullframe_finetuning_b2_e10_fold4_540x540"
## 保存名の設定
RESULT_ID = f"{LOAD_ID}_testResult{'_judgeMesh' if IS_JUDGE_BY_MESH else ''}"

## フルフレームでテストを回すか否か
IS_FULLFRAME = True

## フレッシュ性状データを入力するか否か
IS_INCLUDE_FRESH = False
## フレッシュ性状データのカーネルサイズ(Encoderの出力部分と同じにする)
FRESH_KERNEL_SIZE = "auto"
####

## クラス分類の設定
CLASSIFICATION = "before-just"
DO_FOURCLASSES_TEST = False
CLASS_COLOR = ["red", "pink", "cyan", "green"]

## 量子化に関する設定
IS_QUANTIZED = False
QUANTIZED_FILE = "quantized_model_f32.bin"
QUANTIZED_BUF = "_f32"

## 輝度の拡張に関する設定
IS_USE_LE = False
LE_MODE = "circle"
LE_CONST = 50
LE_TEST_IMAGE_SKIP = 25

## ===========================================


## コマンドライン引数の受け付け
if IS_USE_COMMAND_INPUT:
    args = sys.argv
    if (len(args) > 1):
        # print("args :")
        # for i, a in enumerate(args):
        #     print(f" > {i:02d}\t{a}")
        FOLD = int(args[1])
        MODEL_NAME = args[2]
        LOAD_PATH = args[3]
        LOAD_ID = args[4]
        SIZE = [int(args[5]), int(args[6])]
        IS_FULLFRAME = int(args[7])
        IS_INCLUDE_FRESH = int(args[8])
        IS_JUDGE_BY_MESH = int(args[9])
        IS_GRAYSCALE = int(args[10])
        IS_FUSION_FACE = int(args[11])
        NORM = args[12]
        IS_H5 = int(args[13])
        IS_USE_AVERAGE_IMAGE = int(args[14])
        USE_ATTENTION = int(args[15])
        CLASSIFICATION = args[16]
        IS_QUANTIZED = int(args[17])            # FlatBuffers, float32
        DO_FOURCLASSES_TEST = int(args[18])     # 4classes
        MULTI_LOSSES = int(args[19])
        FOURCLASSES_TYPE = args[20]             # classtering or myself
        IS_USE_LE = int(args[21])
        LE_MODE = args[22]
        LE_CONST = int(args[23])
        USE_CUSTOM_LOSS = int(args[24])
        LOSS = args[25]
        COLOR_TYPE = args[26]
        NORMALIZATION = args[27]
        USE_AE_INPUT = int(args[28])
        NOISE_TYPE = args[29]
        AUTOENCODER_LOSS = args[30]
        AE_MODEL_ID = args[31]
        DO_THREECLASSES_TEST = int(args[32])    # 3classes
            
        ## 分類クラス数の矯正
        if DO_THREECLASSES_TEST:
            DO_FOURCLASSES_TEST = 0
            
        ## 損失関数の設定
        if (LOSS == "iou"):
            LOSS = MyLosses.iou_loss
        elif (LOSS == "cross_entropy_iou"):
            LOSS = MyLosses.crossentropy_iou_loss
            
        cp.cprint("@ Accepted command argments.", "orange")

im.fold = FOLD
if USE_AE_INPUT:
    im.AE_model_id = AE_MODEL_ID

## クラス分類の設定
IS_AFTERCLASS = CLASSIFICATION == "before-just-after"
IS_FOURCLASSES = CLASSIFICATION == "fourclasses" and not IS_AFTERCLASS
NUM_CLASSES += 2*IS_FOURCLASSES + IS_AFTERCLASS
IS_MYSELFCLASSES = FOURCLASSES_TYPE == "myself"
IS_USE_RECTIFIED_DATASET = FOURCLASSES_TYPE == "rectified"

## 保存名に関する設定
if IS_USE_LE:
    LE_const_str = str(LE_CONST).replace("-", "in")
    RESULT_ID = f"{LOAD_ID}{f'_quantized{QUANTIZED_BUF}' if IS_QUANTIZED else ''}_{LE_MODE}-{NOISE_TYPE}-{LE_const_str}_testResult{'_judgeMesh' if IS_JUDGE_BY_MESH else ''}{'_after' if IS_AFTERCLASS else ''}{'_4classes' if DO_FOURCLASSES_TEST else ''}{'_3classes' if DO_THREECLASSES_TEST else ''}{'_rectified' if FOURCLASSES_TYPE == 'rectified' else ''}"
else:
    RESULT_ID = f"{LOAD_ID}{f'_quantized{QUANTIZED_BUF}' if IS_QUANTIZED else ''}_testResult{'_judgeMesh' if IS_JUDGE_BY_MESH else ''}{'_after' if IS_AFTERCLASS else ''}{'_4classes' if DO_FOURCLASSES_TEST else ''}{'_3classes' if DO_THREECLASSES_TEST else ''}{'_rectified' if FOURCLASSES_TYPE == 'rectified' else ''}"

## Metric Learningを含んだネットワークか否か
IS_USE_METRIC = 0
for target in ["arcface", "cosface", "sphereface"]:
    IS_USE_METRIC += int(target in MODEL_NAME)
IS_USE_METRIC = bool(IS_USE_METRIC)

## 読み込む画像のパスが書かれたテキストファイル
path = lambda target : f"{DATASET_DIR}/text_dataset/fold{FOLD}/{target}{'_after_wide' if IS_AFTERCLASS else ''}{'_4class' if IS_FOURCLASSES else ''}{'_myself' if IS_MYSELFCLASSES else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}.txt"
path_multilosses = lambda target : f"{DATASET_DIR}/text_dataset/fold{FOLD}/{target[0]}{'_after_wide' if IS_AFTERCLASS else ''}{'_4class' if target[1] else ''}{'_myself' if IS_MYSELFCLASSES else ''}{'_rectified' if IS_USE_RECTIFIED_DATASET else ''}.txt"
## エラー表示
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")

## 保存先のファイルの新規作成
if IS_OUTPUT_RESULT:
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}.txt", mode="w") as logoutput: pass
if IS_OUTPUT_DETAIL_RESULT:
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_correct.txt", mode="w") as logoutput: pass
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_uncorrect.txt", mode="w") as logoutput: pass
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_combined.txt", mode="w") as logoutput: pass

## データ数の調整
with open(path("train")) as f:
    data_length = len(f.readlines())
    LIMIT = data_length - (data_length%BATCH_SIZE)

## フレッシュ性状データのカーネルサイズの調節
if isinstance(FRESH_KERNEL_SIZE, str):
    if (FRESH_KERNEL_SIZE == "auto"):
        FRESH_KERNEL_SIZE = list(map(lambda x: x//16, SIZE))
    else:
        cp.cprint("\"FRESH_KERNEL_SIZE\" must be either [int, int] or \"auto\".")
        LIMIT = -1
        
        
def output(contents:list):
    """
    @機能：保存先のファイルに書き込む
    @引数：contents = 書き込みたい内容(str型でも、int型でも、リスト型でも、辞書型でも、なんでもOK)
    @戻値：None
    """
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}.txt", mode="a") as logoutput:
        print(contents, file=logoutput)


def output_multilosses(contents:list, fourcls:bool=False):
    """
    @機能：保存先のファイルに書き込む
    @引数：contents = 書き込みたい内容(str型でも、int型でも、リスト型でも、辞書型でも、なんでもOK)
    @戻値：None
    """
    
    with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID.replace('_4classes', '')}{'_4classes' if fourcls else ''}.txt", mode="a") as logoutput:
        print(contents, file=logoutput)


def output_anyfile(contents:list, filename:str):
    """
    @機能：保存先のファイルに書き込む(こっちはファイル名を引数で指定できる)
    @引数：contents = 書き込みたい内容(型の指定なし), filename:str = ファイル名
    @戻値：None
    """
    if IS_OUTPUT_DETAIL_RESULT:
        with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_{filename}.txt", mode="a") as logoutput:
            print(contents, file=logoutput)


def output_state(color:str, string):
    """
    @機能：ステータスファイルに文字列を出力(get_state.pyでこの文字列を取得)
    @引数：color:str = 文字色, string = 出力する文字列
    @戻値：None
    """
    with open(STATE_TEXT, "w") as f:
        f.write(f"{color}\n{string}")


def dataGenerator(resourcepath, user_fourclasses_bool=False):
    """
    @機能：データセットの画像とラベルを学習用に整形
    @引数：resourcepath = 画像のパスが列挙されたテキストファイルのパス
    @戻値：画像(shape : *SIZE, 3)、ラベルのone-hot(shape : *SIZE, 2)
    """
    
    ## テキストファイルから、データの読み取り
    image_path, mask_path, fresh_data = [], [], []
    with open(resourcepath) as f:
        readlines = (f.readlines())[::SKIP]
        # random.seed(RANDOM_SEED)
        # random.shuffle(readlines)
        if LIMIT: readlines = readlines[:LIMIT]
    for line in map(lambda x: x.rstrip("\n"), readlines[::LE_TEST_IMAGE_SKIP**int(IS_USE_LE)]):
        linebuffer = line.split(" ")    # linebuffer = [コンクリ画像のパス, マスク画像のパス]
        image_path.append(linebuffer[0])
        mask_path.append(linebuffer[1])
        if IS_INCLUDE_FRESH:
            fresh_data.append(list(map(float, linebuffer[2:])))
    length = len(image_path)
    data_gen_args = dict(
        rescale=None
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
    mask_dataframe = pd.DataFrame(mask_path, index=None, columns=["mask"])

    image_generator = image_datagen.flow_from_dataframe(image_dataframe,
                                                        x_col="image",
                                                        target_size=SIZE.copy(),
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        # classes=CLASSES[CLASSIFICATION].copy(),
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                    #   classes=CLASSES[CLASSIFICATION].copy(),
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)
    
    ## 画像を一枚ずつ取り出し、出力
    fresh_index = 0
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        image, mask = im.adjust_data(image, mask, IS_FULLFRAME, False,
                                     IS_USE_AVERAGE_IMAGE, SIZE, IS_GRAYSCALE, NUM_CLASSES,
                                     AVERAGE_IMAGE_PATH, classification=CLASSIFICATION,
                                     to_two_classes=(not user_fourclasses_bool and MULTI_LOSSES),
                                     is_use_LE=IS_USE_LE, LE_mode=LE_MODE, LE_const=LE_CONST,
                                     color_type=COLOR_TYPE, normalization=NORMALIZATION,
                                     use_AE_input=USE_AE_INPUT, noise_type=NOISE_TYPE,
                                     autoencoder_loss=AUTOENCODER_LOSS)
        
        ## フレッシュ性状データの入力あり
        if IS_INCLUDE_FRESH:
            fresh_buffer = []
            ## フレッシュ性状データをバッチサイズごとに切り出し
            for index in range(fresh_index, fresh_index+BATCH_SIZE):
                if (index < length):
                    fresh_buffer.append(fresh_data[index])
                else:
                    break
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
            yield [image, fresh], mask, im.get_image_key(image_path[i%length], True, True)
    
        ## フレッシュ性状データの入力なし
        else:
            yield image, mask, im.get_image_key(image_path[i%length], True, True)


def datacounter(datapath):
    """
    @機能：テキストファイルの行数を数えるだけ
    @引数：数えたいテキストファイルのパス
    @戻値：行数
    """
    with open(datapath) as f:
        readlines = (f.readlines())[::SKIP]
        if LIMIT : readlines = readlines[:LIMIT]
        if IS_USE_LE : readlines = readlines[::LE_TEST_IMAGE_SKIP]
    return len(readlines)


def createModel(model_name:str=None):
    """
    @機能：モデルの定義
    @引数：model_name = モデルの定義名, weights = 読み込む重みファイルのパス
    @戻値：モデル
    """
    model = None
    if model_name is None:
        blank("model_name")
            
    else:
        cp.cprint(f"- model : {model_name} -", "cyan")
        cp.cprint(f"- LOAD_ID : {LOAD_ID} -", "cyan")
        
        ## モデルの振り分け
        if (model_name in ["unet", "unet_arcface", "unet_cosface", "unet_sphereface"]):
            is_pure_unet = model_name == "unet"
            model = SemanticSegmentation.unet([*SIZE, 1 if IS_GRAYSCALE else 3], is_compile=is_pure_unet, norm=NORM)
        
        elif (model_name == "unet_fresh"):
            model = SemanticSegmentation.unet_include_fresh([*SIZE, 1 if IS_GRAYSCALE else 3], [*FRESH_KERNEL_SIZE, 5])
        
        elif (model_name == "pspnet"):
            model = SemanticSegmentation.pspnet([*SIZE, 1 if IS_GRAYSCALE else 3])
        
        elif (model_name == "unet_metric_classifier"):
            model = SemanticSegmentation.unet([*SIZE, 1 if IS_GRAYSCALE else 3], is_compile=False, norm=NORM)
            model = SemanticSegmentation.unet_available_metric(model, "sphereface",
                                                                [*SIZE, 1 if IS_GRAYSCALE else 3], 2, is_fusion_face=IS_FUSION_FACE, dropout_const=0.)
            model = SemanticSegmentation.unet_learning_classifier(model)
        
        elif (model_name == "unet_only_classifier"):
            model = SemanticSegmentation.unet([*SIZE, 1 if IS_GRAYSCALE else 3],is_compile=False, norm=NORM)
            model = SemanticSegmentation.unet_learning_classifier(model)
            
        elif (model_name == "e-unet"):
            model  = E_UNet.run([*SIZE, 1 if IS_GRAYSCALE else 3], use_attention=USE_ATTENTION)
        
        elif (model_name == "e-unet_metric_classifier"):
            model = E_UNet.run([*SIZE, 1 if IS_GRAYSCALE else 3], dropout_const=0., is_compile=False, use_attention=USE_ATTENTION)
            model = SemanticSegmentation.unet_available_metric(model, "sphereface",
                                                               [*SIZE, 1 if IS_GRAYSCALE else 3], 2, is_fusion_face=IS_FUSION_FACE, dropout_const=0.,
                                                               is_eunet=True, is_test=True)
            model = SemanticSegmentation.unet_learning_classifier(model)
        
        else: cp.cprint(f"[!] {model_name} is not defined.", "red")
        
    return model


def test():
    """
    @機能：テストの実行
    @引数：mpdel_name = モデルの定義名
    @戻値：テストの結果(AccracyやF1値などの数値)
    """
    
    ## 量子化されたモデルの読み込み
    if IS_QUANTIZED:
        interpreter = tf.lite.Interpreter(model_path=f"{LOAD_PATH}/{LOAD_ID}/{QUANTIZED_FILE}")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    ## それ以外のH5またはPB形式のモデルの読み込み
    else:
        if USE_CUSTOM_LOSS:
        # if True: ## 一旦ね
            if ("e-unet" in MODEL_NAME):
                model = E_UNet.run(input_shape=(*SIZE, 3), num_classes=2+2*IS_FOURCLASSES, loss=LOSS)
                model.load_weights(f"{LOAD_PATH}/{LOAD_ID}")
        else:
            if IS_H5:
                model = load_model(f"{LOAD_PATH}/{LOAD_ID}/{LOAD_ID}.h5")
            else:
                model = load_model(f"{LOAD_PATH}/{LOAD_ID}")
    
    if not IS_HIDE_SUMMARY: model.summary()
    
    for multi_losses_itr in range(1 + int(MULTI_LOSSES)):
        
        # if MULTI_LOSSES:
        is_fourclasses = bool(multi_losses_itr)*MULTI_LOSSES + IS_FOURCLASSES*(1-MULTI_LOSSES)
        do_fourclasses_test = bool(multi_losses_itr)*MULTI_LOSSES + DO_FOURCLASSES_TEST*(1-MULTI_LOSSES)
        
        correct = 0    
        datacount = datacounter(path("test"))
        
        ## 各種データの初期化
        if do_fourclasses_test:
            true_positive = {"before":0, "b-before":0, "b-just":0, "just":0}
            true_negative = {"before":0, "b-before":0, "b-just":0, "just":0}
            false_positive = {"before":0, "b-before":0, "b-just":0, "just":0}
            false_negative = {"before":0, "b-before":0, "b-just":0, "just":0}
        elif DO_THREECLASSES_TEST:
            # true_positive = {"before":0, "boundaly":0, "just":0}
            # true_negative = {"before":0, "boundaly":0, "just":0}
            # false_positive = {"before":0, "boundaly":0, "just":0}
            # false_negative = {"before":0, "boundaly":0, "just":0}
            true_positive = {"just" : 0, "after" : 0}
            true_negative = {"just" : 0, "after" : 0}
            false_positive = {"just" : 0, "after" : 0}
            false_negative = {"just" : 0, "after" : 0}
        else:
            true_positive = {"before":0, "just":0}
            true_negative = {"before":0, "just":0}
            false_positive = {"before":0, "just":0}
            false_negative = {"before":0, "just":0}
        
        ## 保存先のテキストに項目名を入力
        output_anyfile("inputdata, answer, predict", "correct")
        output_anyfile("inputdata, answer, predict", "uncorrect")
        output_anyfile("inputdata, answer, predict", "combined")
        
        if IS_JUDGE_BY_MESH: cp.cprint("\n@ Judgment by mesh area.", "orange")
        
        ## テストの進行状況のリアルタイム表示の初期化
        cp.cprint(f"\n\ncompleted : 0 / {datacount} ( 0.00 [%] ) \t--:--:--", "green")
        cp.cprint(f"accuracy  : --.--- [%]", "cyan")
        cp.cprint("\n")
        
        ## 開始時刻の保存
        timecounter = TimeCounter(datacount)
        
        ## 時間毎に正誤判定を行うための変数たち（メッシュ領域限定）
        days = 0
        meshes = [[[] for _ in range(24)]]
        mesh_anses = [[[] for _ in range(24)]]
        old_key = ""
        if do_fourclasses_test:
            iou = {"before" : 0, "b-before" : 0, "b-just" : 0, "just" : 0}
            iou_accum = {"before" : 0, "b-before" : 0, "b-just" : 0, "just" : 0}
        elif DO_THREECLASSES_TEST:
            # iou = {"before" : 0, "boundaly" : 0, "just" : 0}
            # iou_accum = {"before" : 0, "boundaly" : 0, "just" : 0}
            iou = {"just" : 0, "after" : 0}
            iou_accum = {"just" : 0, "after" : 0}
            key_0, key_1 = "just", "after"
        else:
            iou = {"before" : 0, "just" : 0}
            iou_accum = {"before" : 0, "just" : 0}
            key_0, key_1 = "before", "just"
        
        ## 閾値を含んだ判定マップ
        if IS_USE_THRESHOLD:
            calc_map = np.zeros(SIZE)
        
        ## 正誤判定の閾値計算をリセットして良いか否か
        available_reset = False
        
        ## 正誤判定を行うピクセル数
        pixels = 24 if IS_JUDGE_BY_MESH else SIZE[0]*SIZE[1]
        # if do_fourclasses_test:
        #     pixels *= 4
        
        ## テストの実行
        for i, (img, msk, img_key) in enumerate(dataGenerator(path_multilosses(["test", is_fourclasses]), is_fourclasses)):
            if (i >= datacount): break
            if not i: old_key = img_key
            
            ## skip
            # i *= 3
            
            # data = [img, msk] if IS_USE_METRIC else img
            data = img
            ## 推論("a ,= c" は "a, *b = c" と同等。消しちゃだめ)
            if IS_QUANTIZED:
                interpreter.set_tensor(input_details[0]['index'], data)
                interpreter.invoke()
                predict ,= interpreter.get_tensor(output_details[0]['index'])
            else:
                if MULTI_LOSSES:
                    predict ,= model.predict(data, batch_size=BATCH_SIZE, verbose=0)[multi_losses_itr]
                else:
                    predict ,= model.predict(data, batch_size=BATCH_SIZE, verbose=0)
            
            if not do_fourclasses_test:# and not DO_THREECLASSES_TEST:
                #### 予測ラベルと正解ラベルを計算用に整形
                ## 正解ラベルの整形
                ans_label = np.argmax(msk[0], axis=2)
            
                ## ４クラス分類から２クラスへ変換する
                if is_fourclasses:
                    ans_label = ans_label >= 2
                    four_pred_buf = np.zeros((*SIZE, 2))
                    ## Softmax通してるから足してもOK!
                    four_pred_buf[:, :, 0] += predict[:, :, 0] + predict[:, :, 1]
                    four_pred_buf[:, :, 1] += predict[:, :, 2] + predict[:, :, 3]
                    del predict
                    predict = np.copy(four_pred_buf)
                    del four_pred_buf
                
                ## 判定の入れ替わりに閾値を設ける
                ## (点滅を避けるためだが、SoftmaxCrossEntropyの制約が強くて意味ない)
                if IS_USE_THRESHOLD:
                    if not np.sum(ans_label) and available_reset:
                        calc_map = np.zeros(SIZE)
                    positive = calc_map == 1
                    negative = positive^1
                    calc_map -= ((predict[:, :, 0] > THRESHOLD[0]) & positive) * 2
                    calc_map += ((predict[:, :, 1] > THRESHOLD[1]) & negative) * 2
                    calc_map = np.clip(calc_map, 0, 1)
                    pred_label = np.copy(calc_map)
                else:
                    pred_label = np.argmax(predict, axis=2)
            
            ## メッシュ領域ごとに値をまとめる
            if IS_JUDGE_BY_MESH:
                if do_fourclasses_test:
                    ans_label = np.argmax(msk[0], axis=2)
                    pred_label = np.argmax(predict, axis=2)
                    pred_buf = np.zeros((4, 6, 4))
                    ans_buf = np.zeros((4, 6, 4))
                    for row in range(4):
                        for col in range(6):
                            for class_number in range(4):
                                pred_buf[row][col][class_number] = np.sum(pred_label[SIZE[0]//4*row:SIZE[0]//4*(row+1), SIZE[1]//6*col:SIZE[1]//6*(col+1)] == class_number)
                                ans_buf[row][col][class_number] = ans_label[SIZE[0]//4*row][SIZE[1]//6*col] == class_number
                    pred_buf = pred_buf >= (SIZE[0]*SIZE[1]/24)/4
                # elif IS_AFTERCLASS:
                #     ans_label = np.argmax(msk[0], axis=2)
                #     pred_label = np.argmax(predict, axis=2)
                #     pred_buf = np.zeros((4, 6, 3))
                #     ans_buf = np.zeros((4, 6, 3))
                #     for row in range(4):
                #         for col in range(6):
                #             for class_number in range(3):
                #                 pred_buf[row][col][class_number] = np.sum(pred_label[SIZE[0]//4*row:SIZE[0]//4*(row+1), SIZE[1]//6*col:SIZE[1]//6*(col+1)] == class_number)
                #                 ans_buf[row][col][class_number] = ans_label[SIZE[0]//4*row][SIZE[1]//6*col] == class_number
                #     pred_buf = pred_buf >= (SIZE[0]*SIZE[1]/24)/3
                else:
                    pred_buf = np.zeros((4, 6))
                    ans_buf = np.zeros((4, 6))
                    for row in range(4):
                        for col in range(6):
                            pred_buf[row][col] = np.sum(pred_label[SIZE[0]//4*row:SIZE[0]//4*(row+1), SIZE[1]//6*col:SIZE[1]//6*(col+1)])
                            # if IS_AFTERCLASS:
                                # ans_buf[row][col] = ans_label[SIZE[0]//4*row][SIZE[1]//6*col] < 2
                            # else:
                            ans_buf[row][col] = ans_label[SIZE[0]//4*row][SIZE[1]//6*col] == 1
                    pred_buf = pred_buf >= (SIZE[0]*SIZE[1]/24)/2
                
                pred_label = np.copy(pred_buf)
                ans_label = np.copy(ans_buf)
                
                ## 時間毎の正誤判定のリストを初期化
                if (old_key != img_key):
                    days += 1
                    meshes += [[[] for _ in range(24)]]
                    mesh_anses += [[[] for _ in range(24)]]
                    old_key = img_key
                
                ## 時間毎の正誤判定を記録
                for j, (p, a) in enumerate(zip(pred_label, ans_label)):
                    for k, (p_val, a_val) in enumerate(zip(p, a)):
                        if do_fourclasses_test:
                            p_val = np.argmax(p_val); a_val = np.argmax(a_val)
                        # anses[j*6+k] = a_val
                        meshes[days][j*6+k].append(p_val == a_val)
                        mesh_anses[days][j*6+k].append(a_val)
            
            elif do_fourclasses_test:
                ans_label = np.copy(msk[0])
                pred_label = np.identity(4)[np.argmax(predict, axis=(2))]

            # elif DO_THREECLASSES_TEST:
            #     # three_ans_buf = np.zeros((*SIZE, 3))
            #     ans_label = np.copy(msk[0])
            #     # three_ans_buf[:, :, 0] += msk[0, :, :, 0]
            #     # three_ans_buf[:, :, 1] += np.clip(msk[0, :, :, 1] + msk[0, :, :, 2], 0, 1)
            #     # three_ans_buf[:, :, 2] += msk[0, :, :, 3]
            #     # ans_label = np.copy(three_ans_buf)
            #     # del three_ans_buf
                
            #     # three_pred_buf = np.zeros((*SIZE, 3))
            #     ## Softmax通してるから足してもOK!
            #     # three_pred_buf[:, :, 1] += predict[:, :, 1] + predict[:, :, 2]
            #     # three_pred_buf[:, :, 2] += predict[:, :, 3]
            #     # del predict
            #     # predict = np.copy(three_pred_buf)
            #     # del three_pred_buf
            #     pred_label = np.identity(3)[np.argmax(predict, axis=(2))]

            available_reset = np.min(ans_label)
            
            pred_label = np.uint8(pred_label)
            ans_label = np.uint8(ans_label)
            
            # print(ans_label)
            # print(pred_label)
            # exit()
            
            #### 精度の算出
            ## 4クラスでの精度の算出
            if do_fourclasses_test:
                ## 不正解数を集計
                incorrect = np.sum(np.sum(ans_label ^ pred_label, axis=(2)) >= 1)
                ## 正解数を逆算
                correct += pixels - np.sum(incorrect)
                
                #### 各種の値を行列演算(a : answer, p : prediction)
                # for class_number in range(4):
                ## a & p
                a_and_p = np.sum(ans_label & pred_label, axis=(0, 1))
                ## ~a & ~p
                not_a_and_not_p = np.sum(Calc.inverse_matrix(ans_label) & Calc.inverse_matrix(pred_label), axis=(0, 1))
                ## ~(a & p)
                not_1_a_and_p_1 = np.sum(Calc.inverse_matrix(ans_label & pred_label), axis=(0, 1))
                ## ~p & a
                not_p_or_a = np.sum(Calc.inverse_matrix(pred_label) & ans_label, axis=(0, 1))
                ## p & ~a
                p_or_not_a = np.sum(pred_label & Calc.inverse_matrix(ans_label), axis=(0, 1))
                ## a | p
                a_or_p = np.sum(pred_label | ans_label, axis=(0, 1))
                ## ~a | ~p
                not_a_or_not_p = np.sum(Calc.inverse_matrix(pred_label) | Calc.inverse_matrix(ans_label), axis=(0, 1))
                
                for class_number, key in enumerate(true_positive.keys()):
                    # True Positive
                    true_positive[key] += a_and_p[class_number]
                    ## True Negative
                    true_negative[key] += not_1_a_and_p_1[class_number]
                    ## False Positive
                    false_positive[key] += p_or_not_a[class_number]
                    ## False Negative
                    false_negative[key] += not_p_or_a[class_number]
                
                    ## IoUの計算
                    if a_or_p[class_number]:
                        iou[key] += np.nan_to_num(a_and_p[class_number] / a_or_p[class_number])
                        iou_accum[key] += 1

                if not i:
                    print("\n\n\n\n\n\n")
                    if IS_JUDGE_BY_MESH:
                        print("\n\n\n\n\n\n\n\n\n\n")
                
                if IS_JUDGE_BY_MESH:
                    print(f"\033[15Aprediction : \n" + "\n".join(map(lambda p: "".join(map(lambda x: cp.colored("##", color=CLASS_COLOR[x], background=CLASS_COLOR[x]), p)), np.argmax(pred_label, axis=(2)))))
                    print(f"answer : \n" + "\n".join(map(lambda a: "".join(map(lambda x: cp.colored("##", color=CLASS_COLOR[x], background=CLASS_COLOR[x]), a)), np.argmax(ans_label, axis=(2)))))
                    print("\n\n\n\n\n", end="")

                ## 終了時刻の予測
                remining_time = timecounter.predictTime(i+1)
                
                ## テストの進行状況のリアルタイム表示
                cp.cprint(f"\033[5Acompleted : {i+1} / {datacount} ( {round((i+1)/datacount*100, 2)} [%] )\t|  {img_key}  |  {remining_time}{' '*30}", "green")
                cp.cprint(f"accuracy  : {round(correct/(pixels*(i+1))*100, 3)} [%]{' '*50}", "cyan")
                cp.cprint(f"TP : {true_positive}, TN : {true_negative}{' '*10}", "cyan")
                cp.cprint(f"FP : {false_positive}, FN : {false_negative}{' '*10}", "cyan")
                # cp.cprint(f"IoU : Before {round(iou[key_0]/iou_accum[key_0], 3)}, Just {round(iou['just']/iou_accum['just'], 3)}{' '*10}", "cyan")
                if all(iou.values()):
                    cp.cprint(f"IoU : Before {round(iou['before']/iou_accum['before'], 3)}, B-Before {round(iou['b-before']/iou_accum['b-before'], 3)}, B-Just {round(iou['b-just']/iou_accum['b-just'], 3)}, Just {round(iou['just']/iou_accum['just'], 3)}{' '*50}", "cyan")
                else:
                    cp.cprint(f"IoU : It's incalculable. Wait a minute. (;~;)", "cyan")
                    
                    
            ## 3クラスでの精度の算出
            # elif DO_THREECLASSES_TEST:
            #     ## 不正解数を集計
            #     incorrect = np.sum(np.sum(ans_label ^ pred_label, axis=(2)) >= 1)
            #     ## 正解数を逆算
            #     correct += pixels - np.sum(incorrect)
                
            #     #### 各種の値を行列演算(a : answer, p : prediction)
            #     ## a & p
            #     a_and_p = np.sum(ans_label & pred_label, axis=(0, 1))
            #     ## ~a & ~p
            #     not_a_and_not_p = np.sum(Calc.inverse_matrix(ans_label) & Calc.inverse_matrix(pred_label), axis=(0, 1))
            #     ## ~(a & p)
            #     not_1_a_and_p_1 = np.sum(Calc.inverse_matrix(ans_label & pred_label), axis=(0, 1))
            #     ## ~p & a
            #     not_p_or_a = np.sum(Calc.inverse_matrix(pred_label) & ans_label, axis=(0, 1))
            #     ## p & ~a
            #     p_or_not_a = np.sum(pred_label & Calc.inverse_matrix(ans_label), axis=(0, 1))
            #     ## a | p
            #     a_or_p = np.sum(pred_label | ans_label, axis=(0, 1))
            #     ## ~a | ~p
            #     not_a_or_not_p = np.sum(Calc.inverse_matrix(pred_label) | Calc.inverse_matrix(ans_label), axis=(0, 1))
                
            #     for class_number, key in enumerate(true_positive.keys()):
            #         # True Positive
            #         true_positive[key] += a_and_p[class_number]
            #         ## True Negative
            #         true_negative[key] += not_1_a_and_p_1[class_number]
            #         ## False Positive
            #         false_positive[key] += p_or_not_a[class_number]
            #         ## False Negative
            #         false_negative[key] += not_p_or_a[class_number]
                
            #         ## IoUの計算
            #         if a_or_p[class_number]:
            #             iou[key] += np.nan_to_num(a_and_p[class_number] / a_or_p[class_number])
            #             iou_accum[key] += 1

            #     if not i:
            #         print("\n\n\n\n\n\n")
            #         if IS_JUDGE_BY_MESH:
            #             print("\n\n\n\n\n\n\n\n\n\n")
                
            #     if IS_JUDGE_BY_MESH:
            #         print(f"\033[15Aprediction : \n" + "\n".join(map(lambda p: "".join(map(lambda x: cp.colored("##", color=CLASS_COLOR[x], background=CLASS_COLOR[x]), p)), np.argmax(pred_label, axis=(2)))))
            #         print(f"answer : \n" + "\n".join(map(lambda a: "".join(map(lambda x: cp.colored("##", color=CLASS_COLOR[x], background=CLASS_COLOR[x]), a)), np.argmax(ans_label, axis=(2)))))
            #         print("\n\n\n\n\n", end="")

            #     ## 終了時刻の予測
            #     remining_time = timecounter.predictTime(i+1)
                
            #     ## テストの進行状況のリアルタイム表示
            #     cp.cprint(f"\033[5Acompleted : {i+1} / {datacount} ( {round((i+1)/datacount*100, 2)} [%] )\t|  {img_key}  |  {remining_time}{' '*30}", "green")
            #     cp.cprint(f"accuracy  : {round(correct/(pixels*(i+1))*100, 3)} [%]{' '*50}", "cyan")
            #     cp.cprint(f"TP : {true_positive}, TN : {true_negative}{' '*10}", "cyan")
            #     cp.cprint(f"FP : {false_positive}, FN : {false_negative}{' '*10}", "cyan")
            #     # cp.cprint(f"IoU : Before {round(iou['before']/iou_accum['before'], 3)}, Just {round(iou['just']/iou_accum['just'], 3)}{' '*10}", "cyan")
            #     if all(iou.values()):
            #         cp.cprint(f"IoU : Before {round(iou['before']/iou_accum['before'], 3)}, Boundaly {round(iou['boundaly']/iou_accum['boundaly'], 3)}, Just {round(iou['just']/iou_accum['just'], 3)}{' '*50}", "cyan")
            #     else:
            #         cp.cprint(f"IoU : It's incalculable. Wait a minute. (;~;)", "cyan")
            
            ## 通常の精度算出
            else:
                ## 不正解数を集計
                incorrect = ans_label ^ pred_label
                ## 正解数を逆算
                correct += pixels - np.sum(incorrect)
                
                #### 各種の値を行列演算(a : answer, p : prediction)
                ## a & p
                a_and_p = np.sum(ans_label & pred_label)
                ## ~a & ~p
                not_a_and_not_p = np.sum(Calc.inverse_matrix(ans_label) & Calc.inverse_matrix(pred_label))
                ## ~(a & p)
                not_1_a_and_p_1 = np.sum(Calc.inverse_matrix(ans_label & pred_label))
                ## ~p & a
                not_p_or_a = np.sum(Calc.inverse_matrix(pred_label) & ans_label)
                ## p & ~a
                p_or_not_a = np.sum(pred_label & Calc.inverse_matrix(ans_label))
                ## a | p
                a_or_p = np.sum(pred_label | ans_label)
                ## ~a | ~p
                not_a_or_not_p = np.sum(Calc.inverse_matrix(pred_label) | Calc.inverse_matrix(ans_label))
                
                # True Positive
                true_positive[key_0] += np.nan_to_num(not_1_a_and_p_1)
                true_positive[key_1] += np.nan_to_num(a_and_p)
                ## True Negative
                true_negative[key_0] += np.nan_to_num(a_and_p)
                true_negative[key_1] += np.nan_to_num(not_1_a_and_p_1)
                ## False Positive
                false_positive[key_0] += np.nan_to_num(not_p_or_a)
                false_positive[key_1] += np.nan_to_num(p_or_not_a)
                ## False Negative
                false_negative[key_0] += np.nan_to_num(p_or_not_a)
                false_negative[key_1] += np.nan_to_num(not_p_or_a)
                
                ## IoUの計算
                if a_or_p:
                    iou[key_1] += np.nan_to_num(a_and_p / a_or_p)
                    iou_accum[key_1] += 1
                if not_a_or_not_p:
                    iou[key_0] += np.nan_to_num(not_a_and_not_p / not_a_or_not_p)
                    iou_accum[key_0] += 1
            
                if not i:
                    print("\n\n\n\n\n")
                    if IS_JUDGE_BY_MESH:
                        print("\n\n\n\n\n\n\n\n\n\n")
                
                if IS_JUDGE_BY_MESH:
                    print(f"\033[15Aprediction : \n" + "\n".join(map(lambda p: "".join(map(lambda x: cp.colored("##", color="green", background="green") if x else cp.colored("##", color="red", background="red"), p)), pred_label)))
                    print(f"answer : \n" + "\n".join(map(lambda a: "".join(map(lambda x: cp.colored("##", color="green", background="green") if x else cp.colored("##", color="red", background="red"), a)), ans_label)))
                    print("\n\n\n\n\n", end="")

                ## 終了時刻の予測
                remining_time = timecounter.predictTime(i+1)
                
                ## テストの進行状況のリアルタイム表示
                cp.cprint(f"\033[5Acompleted : {i+1} / {datacount} ( {round((i+1)/datacount*100, 2)} [%] ) \t|  {img_key}  |  {remining_time}{' '*30}", "green")
                cp.cprint(f"accuracy  : {round(correct/(pixels*(i+1))*100, 3)} [%]{' '*50}", "cyan")
                cp.cprint(f"TP : {true_positive}, TN : {true_negative}{' '*10}", "cyan")
                cp.cprint(f"FP : {false_positive}, FN : {false_negative}{' '*10}", "cyan")
                # cp.cprint(f"IoU : Before {round(iou['before']/iou_accum['before'], 3)}, Just {round(iou['just']/iou_accum['just'], 3)}{' '*10}", "cyan")
                if iou_accum[key_0] and iou_accum[key_1]:
                    cp.cprint(f"IoU : {key_0} {round(iou[key_0]/iou_accum[key_0], 3)}, {key_1} {round(iou[key_1]/iou_accum[key_1], 3)}{' '*50}", "cyan")
                else:
                    cp.cprint(f"IoU : It's incalculable. Wait a minute. (;~;)", "cyan")
            
            # bar.update(1)
        
        print()
        
        denominator = pixels*datacount
        
        ## Accuracyの算出
        accuracy = correct/denominator
        
        ## 最終的なIoUの算出
        for key in iou.keys():
            iou[key] /= iou_accum[key]
        
        ## F1値の算出
        f1 = []
        for key in iou.keys():
            f1.append(Calc.f1_score(
                Calc.precision(
                    tp=true_positive[key],
                    fp=false_positive[key]
                ),
                Calc.recall(
                    tp=true_positive[key],
                    fn=false_negative[key]
                )
            ))
        
        ## テスト結果を辞書に格納
        if do_fourclasses_test:
            survey_keys = ["accuracy", "F1(Before)", "F1(B-Before)", "F1(B-Just)", "F1(Just)",
                        "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)",
                        "TP(B-Before)", "TN(B-Before)", "FP(B-Before)", "FN(B-Before)",
                        "TP(B-Just)", "TN(B-Just)", "FP(B-Just)", "FN(B-Just)",
                        "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)",
                        "IoU(Before)", "IoU(B-Before)", "IoU(B-Just)", "IoU(Just)"]
            survey_values = [accuracy, *f1,
                            true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
                            true_positive["b-before"], true_negative["b-before"], false_positive["b-before"], false_negative["b-before"],
                            true_positive["b-just"], true_negative["b-just"], false_positive["b-just"], false_negative["b-just"],
                            true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
                            *iou.values()]
            
            ## Mark Down用に整形
            markdown = f"mark down : |FOLD{FOLD}|{round(survey_values[0]*100, 3)}| and more ..."
        
        elif DO_THREECLASSES_TEST:
            # survey_keys = ["accuracy", "F1(Before)", "F1(Boundaly)", "F1(Just)",
            #             "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)",
            #             "TP(Boundaly)", "TN(Boundaly)", "FP(Boundaly)", "FN(Boundaly)",
            #             "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)",
            #             "IoU(Before)", "IoU(Boundaly)", "IoU(Just)"]
            # survey_values = [accuracy, *f1,
            #                 true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
            #                 true_positive["boundaly"], true_negative["boundaly"], false_positive["boundaly"], false_negative["boundaly"],
            #                 true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
            #                 *iou.values()]
            
            # ## Mark Down用に整形
            # markdown = f"mark down : |FOLD{FOLD}|{round(survey_values[0]*100, 3)}| and more ..."
            
            survey_keys = ["accuracy", "F1(Just)", "F1(After)", "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)", "TP(After)", "TN(After)", "FP(After)", "FN(After)", "IoU(Just)", "IoU(After)"]
            survey_values = [accuracy, *f1,
                            true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
                            true_positive["after"], true_negative["after"], false_positive["after"], false_negative["after"],
                            *iou.values()]
        
            ## Mark Down用に整形
            markdown = f"mark down : |FOLD{FOLD}|{round(survey_values[0]*100, 3)}|{round(survey_values[1], 5)}|{round(survey_values[2], 5)}|{round(iou[key_0], 5)}|{round(iou[key_1], 5)}|"

        else:
            survey_keys = ["accuracy", "F1(Before)", "F1(Just)", "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)", "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)", "IoU(Before)", "IoU(Just)"]
            survey_values = [accuracy, *f1,
                            true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
                            true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
                            *iou.values()]
        
            ## Mark Down用に整形
            markdown = f"mark down : |FOLD{FOLD}|{round(survey_values[0]*100, 3)}|{round(survey_values[1], 5)}|{round(survey_values[2], 5)}|{round(iou[key_0], 5)}|{round(iou[key_1], 5)}|"
        
        le_str = f"{LE_MODE}, {LE_CONST}" if IS_USE_LE else "default"
        send_master.send(f"test: {LOAD_ID}\n{le_str}\n{markdown}")
        
        ## 保存先のテキストに結果を書き込む
        if MULTI_LOSSES:
            output_multilosses(" ".join(survey_keys), do_fourclasses_test)
            output_multilosses(" ".join(map(str, survey_values)), do_fourclasses_test)
            output_multilosses(markdown, do_fourclasses_test)
        else:
            output(" ".join(survey_keys))
            output(" ".join(map(str, survey_values)))
            output(markdown)
        
        ## 時間毎の正誤判定を可視化し保存
        if IS_JUDGE_BY_MESH:
            for day in range(days+1):
                plt.figure(figsize=(10, 10))
                for i, (m, a) in enumerate(zip(meshes[day], mesh_anses[day])):
                    range_y_top = [i+0.5, 0.5]
                    range_y_bottom = [i+1, 0.5]
                    
                    ## 正誤関係の描画
                    m_positive_bar = [index for index, val in enumerate(m) if val]
                    m_positive_bar = list(zip(m_positive_bar, np.ones(len(m_positive_bar))))
                    
                    m_negative_bar = [index for index, val in enumerate(m) if not val]
                    m_negative_bar = list(zip(m_negative_bar, np.ones(len(m_negative_bar))))
                    
                    plt.broken_barh(xranges=m_positive_bar, yrange=range_y_top, facecolor="white")
                    plt.broken_barh(xranges=m_negative_bar, yrange=range_y_top, facecolor="black")
                    
                    ## 4クラスの色分け描画
                    if do_fourclasses_test:
                        for class_number, color in enumerate(CLASS_COLOR):
                            a_class_bar = [index for index, val in enumerate(a) if val == class_number]
                            a_class_bar = list(zip(a_class_bar, np.ones(len(a_class_bar))))
                            
                            plt.broken_barh(xranges=a_class_bar, yrange=range_y_bottom, facecolor=color)
                    
                    ## 3クラスの色分け描画
                    elif DO_THREECLASSES_TEST:
                        for class_number, color in enumerate(CLASS_COLOR[:3]):
                            a_class_bar = [index for index, val in enumerate(a) if val == class_number]
                            a_class_bar = list(zip(a_class_bar, np.ones(len(a_class_bar))))
                            
                            plt.broken_barh(xranges=a_class_bar, yrange=range_y_bottom, facecolor=color)
                    
                    ## 2クラスの色分け描画
                    else:
                        a_positive_bar = [index for index, val in enumerate(a) if val]
                        a_positive_bar = list(zip(a_positive_bar, np.ones(len(a_positive_bar))))
                        
                        a_negative_bar = [index for index, val in enumerate(a) if not val]
                        a_negative_bar = list(zip(a_negative_bar, np.ones(len(a_negative_bar))))
                    
                        plt.broken_barh(xranges=a_positive_bar, yrange=range_y_bottom, facecolor="green")
                        plt.broken_barh(xranges=a_negative_bar, yrange=range_y_bottom, facecolor="red")
                
                plt.title("Graph of The Hourly Correctness Relationships")
                plt.xlabel("frame")
                plt.ylabel("mesh number")
                
                plt.savefig(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID.replace('_4classes', '')}{'_4classes' if do_fourclasses_test else ''}{'_3classes' if DO_THREECLASSES_TEST else ''}_{day}.png")
                cp.cprint(f"@ Saved plot image. day : {day}", "orange")
                plt.cla()
    
    return dict(zip(survey_keys, survey_values))
    

def main():
    if (LIMIT < 0):
        cp.cprint("[!] \"LIMIT\" is too short or \"BATCH_SIZE\" is too large", "red")
    else:
        if MODEL_NAME is None:
            blank("MODEL_NAME")
        else:
            if IS_USE_LE:
                cp.cprint(f"@ Avalable luminance extend : {LE_MODE}, {NOISE_TYPE}, {LE_CONST}", "orange")
            
            result = test()
            cp.cprint(f"test result : {result}", "green")
    
    output_state("green", f"\"{LOAD_ID}\" test finished.\nThe accuracy is {round(result['accuracy']*100, 3)} [%]")
    
    # send_master.send(f"test: {LOAD_ID}")
    # send_master.send(result)
        
    cp.cprint("- finished ! -", "cyan")
    

if (__name__ == "__main__"):
    main()
