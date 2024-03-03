from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import random
from PIL import Image
import sys
import time
import cv2
import gc

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from ExplainableFuncs import ExplainableFuncs
from GifMaker import GifMaker


## ================ config ===================


#### ディレクトリの設定など(実行環境が変わった時以外、いじらない)
# WORKSPACE_DIR = "/workspace/semanticSegmentation"
# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
WORKSPACE_DIR = "/workspace/fullframe"
DATASET_DIR = "/workspace/Dataset/fullframe"
# STATE_TEXT = "/workspace/osada_ws/state.txt"
AVERAGE_IMAGE_PATH = "/workspace/osada_ws/average_image_0516.png"
# TEXT_PATH = "/workspace/osada_ws/images.txt"
TEXT_PATH = "/workspace/Dataset/fullframe/text_dataset/all_include_fresh.txt"
FLOW_TEXT_PATH = "/workspace/explain/flow/flow_images.txt"
IS_USE_COMMAND_INPUT = True
SAVE_PATH = "/workspace/explain"
FLOW_SAVE_PATH = "/workspace/explain/flow"
####


#### XAIの設定
## 実行する関数の名前
FUNCTION = "seg-grad-cam"
## 対象とする層の名前
OUTPUT_LAYER = "output"
# OUTPUT_LAYER = "conv9e_2"
# OUTPUT_LAYER = "conv5m_2"
## Seg-Grad-CAMで使用する、隠れ層の名前
HIDDEN_LAYERS = ["conv1c_2",
                 "conv2c_2",
                 "conv3c_2",
                 "conv4c_2",
                 "conv5m_2",
                 "conv6e_2",
                 "conv7e_2",
                 "conv8e_2",
                 "conv9e_2"]
## 計算する枚数(特に指定しない場合は、適当に大きい値を設定する)
EXPLAINED_NUMBER = 1e10
## ２クラス分類の出力部を対象とするか否か
IS_BINARY = False

## GIF画像を作成するか否か
IS_MAKE_GIF = True
## GIF画像の出力サイズ(横, 縦)
GIF_SIZE = (270*3, 270*2)
## GIF作成時に省く画像の間隔
GIF_IMAGE_SKIP = 100
## GIFの画像更新間隔
DURATION = 300.

## リアルタイムで画像を確認するか否か
IS_REALTIME_PREVIEW = True
## 出力する画像サイズ
DECODE_SIZE = (270*2, 270*2)
## 色の強調表現の指数
PLOT_MULTI = 1.5
####


#### 学習の基本設定達
## BATCH_SIZEは1のまま
BATCH_SIZE = 1
FOLD = 3
## MODEL_NAME : "unet", "unet_fresh", "pspnet"
MODEL_NAME = "unet"
## 入力画像サイズ
SIZE = [270*2, 270*2]
# SIZE = [270, 270]
## 正解ラベルに関する設定
NUM_CLASSES = 2
CLASSES = ["before", "just"]
## 入力画像をグレースケールにするか否か
IS_GRAYSCALE = False
## 入力画像から平均画像を引くか否か
## (Semantic Segmentationに着手する前のモデル(NINやResNetなど)を用いる時はTrueにする)
IS_USE_AVERAGE_IMAGE = False

RANDOM_SEED = 10
## LIMIT : None or int
LIMIT = None

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
LOAD_ID = "unet_20220319_AutoLearning_fold3_540x540"
# LOAD_ID = "unet_20220302_prefullframe_finetuning_b2_e10_fold4_540x540"
## 保存名の設定
SAVE_ID = f"{FUNCTION}_{LOAD_ID}"
SAVE_BUF = ""
# SAVE_BUF = "_final"

## フルフレームでテストを回すか否か
IS_FULLFRAME = True

## フレッシュ性状データを入力するか否か
IS_INCLUDE_FRESH = False
## フレッシュ性状データのカーネルサイズ(Encoderの出力部分と同じにする)
FRESH_KERNEL_SIZE = "auto"
####


## ===========================================


## GIF画像作成時の設定
if IS_MAKE_GIF:
    TEXT_PATH = FLOW_TEXT_PATH
    SAVE_PATH = FLOW_SAVE_PATH

## コマンドライン引数の受け付け
if IS_USE_COMMAND_INPUT:
    args = sys.argv
    if (len(args) > 1):
        FOLD = int(args[1])
        MODEL_NAME = args[2]
        LOAD_PATH = args[3]
        LOAD_ID = args[4]
        SIZE = [int(args[5]), int(args[6])]
        IS_FULLFRAME = int(args[7])
        IS_INCLUDE_FRESH = int(args[8])
        RESULT_ID = f"{LOAD_ID}_testResult"
        cp.cprint("@ Accepted command argments.", "orange")
        
## 読み込む画像のパスが書かれたテキストファイル
path = lambda target : f"{DATASET_DIR}/text_dataset/fold{FOLD}/{target}{'_include_fresh' if IS_INCLUDE_FRESH else ''}.txt"
## エラー表示
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")

# ## 保存先のファイルの新規作成
# if IS_OUTPUT_RESULT:
#     with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}.txt", mode="w") as logoutput: pass
# if IS_OUTPUT_DETAIL_RESULT:
#     with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_correct.txt", mode="w") as logoutput: pass
#     with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_uncorrect.txt", mode="w") as logoutput: pass
#     with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_combined.txt", mode="w") as logoutput: pass

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
        
        
# def output(contents:list):
#     """
#     @機能：保存先のファイルに書き込む
#     @引数：contents = 書き込みたい内容(str型でも、int型でも、リスト型でも、辞書型でも、なんでもOK)
#     @戻値：None
#     """
#     with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}.txt", mode="a") as logoutput:
#         print(contents, file=logoutput)


# def output_anyfile(contents:list, filename:str):
#     """
#     @機能：保存先のファイルに書き込む(こっちはファイル名を引数で指定できる)
#     @引数：contents = 書き込みたい内容(型の指定なし), filename:str = ファイル名
#     @戻値：None
#     """
#     if IS_OUTPUT_DETAIL_RESULT:
#         with open(f"{LOAD_PATH}/{LOAD_ID}/{RESULT_ID}_{filename}.txt", mode="a") as logoutput:
#             print(contents, file=logoutput)


def dataGenerator(resourcepath):
    """
    @機能：データセットの画像とラベルを学習用に整形
    @引数：resourcepath = 画像のパスが列挙されたテキストファイルのパス
    @戻値：画像(shape : *SIZE, 3)、ラベルのone-hot(shape : *SIZE, 2)
    """
    
    ## テキストファイルから、データの読み取り
    image_path, mask_path, fresh_data = [], [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        ## GIF画像を作成しない場合、画像のシャッフルを行う
        if not IS_MAKE_GIF:
            random.seed(RANDOM_SEED)
            random.shuffle(readlines)
        if LIMIT: readlines = readlines[:LIMIT]
    for line in map(lambda x: x.rstrip("\n"), readlines):
        linebuffer = line.split(" ")    # linebuffer = [コンクリ画像のパス, マスク画像のパス]
        image_path.append(linebuffer[0])
        mask_path.append(linebuffer[1])
        if IS_INCLUDE_FRESH:
            fresh_data.append(list(map(float, linebuffer[2:])))
    length = len(fresh_data)
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
                                                        classes=["before", "just"],
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                      classes=["before", "just"],
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)
    
    ## 画像を一枚ずつ取り出し、出力
    fresh_index = 0
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        image, mask = im.adjust_data(image, mask, IS_FULLFRAME, False,
                                     IS_USE_AVERAGE_IMAGE, SIZE, IS_GRAYSCALE, NUM_CLASSES,
                                     AVERAGE_IMAGE_PATH)
        
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
            yield [image, fresh], mask
        
        ## フレッシュ性状データの入力なし
        else:
            yield image, mask


def datacounter(datapath):
    """
    @機能：テキストファイルの行数を数えるだけ
    @引数：数えたいテキストファイルのパス
    @戻値：行数
    """
    with open(datapath) as f:
        readlines = f.readlines()
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


def explain(model_name:str=None):
    """
    @機能：
    @引数：mpdel_name = モデルの定義名
    @戻値：
    """
    exf = ExplainableFuncs()
    ## 関数の選択
    if (FUNCTION == "grad-cam"):
        func = exf.Grad_CAM
    elif (FUNCTION == "seg-grad-cam"):
        func = exf.Seg_Grad_CAM
    else:
        cp.cprint("\"FUNCTION\" is incorrect!", "red")
        return
    
    ## モデルの定義
    if model_name is None:
        blank("model_name")
        return None
    
    ## 重みの読み込み
    model = mm.create_model(model_name, f"{LOAD_PATH}/{LOAD_ID}/{LOAD_ID}.h5", SAVE_ID, SIZE,
                            IS_GRAYSCALE, FRESH_KERNEL_SIZE, True)
    model.summary()
    
    Utils.makedir(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}")
    datacount = int(min(datacounter(TEXT_PATH), EXPLAINED_NUMBER))
    
    if IS_MAKE_GIF:
        datacount //= GIF_IMAGE_SKIP
        # gifmaker = [GifMaker(size=GIF_SIZE) for _ in HIDDEN_LAYERS]
    
    ## 開始時刻の保存
    timecounter = TimeCounter(datacount)
        
    index = 0
    
    ## 計算開始
    for i, (img, msk) in enumerate(dataGenerator(TEXT_PATH)):
        if (index >= datacount): break
        if IS_MAKE_GIF and i%GIF_IMAGE_SKIP: continue
        
        ## 関数の選択
        if (FUNCTION == "grad-cam"):
            func = exf.Grad_CAM
        elif (FUNCTION == "seg-grad-cam"):
            func = exf.Seg_Grad_CAM
        else:
            cp.cprint("\"FUNCTION\" is incorrect!", "red")
            return
        
        ## モデルと画像から、目的の特徴マップを取得
        # classes, feature_map, explained, added_image, prediction, feature_map_buf = func(model, img[0], OUTPUT_LAYER, HIDDEN_LAYERS, DECODE_SIZE, IS_BINARY, PLOT_MULTI)
        feature_maps, explained, mixed, prediction = func(model, img[0], OUTPUT_LAYER, HIDDEN_LAYERS, DECODE_SIZE, IS_REALTIME_PREVIEW)
        
        
        ## 画像・テキストとして保存
        for j, (e, m) in enumerate(zip(explained, mixed)):
            
            ## 画像出力用にリサイズ
            e = cv2.resize(e, GIF_SIZE)
            m = cv2.resize(m, GIF_SIZE)
            origin = cv2.resize(img[0]*255./2., GIF_SIZE)
            covered_e = np.clip(origin + e, 0, 255)
            covered_m = np.clip(origin + m, 0, 255)
            
            ## 画像を保存
            cv2.imwrite(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_explained_{index:04d}_{j:02d}.png", e)
            cv2.imwrite(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_explained_cover_{index:04d}_{j:02d}.png", covered_e)
            cv2.imwrite(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_mixed_{index:04d}_{j:02d}.png", m)
            cv2.imwrite(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_mixed_cover_{index:04d}_{j:02d}.png", covered_m)
        
            # if IS_MAKE_GIF:
                ## GIF画像の配列に追加
                # gifmaker[j].add_image(m, is_cv2_image=True)
        
        # ## リアルタイムで可視化
        # if IS_REALTIME_PREVIEW:
        #     cv2.imshow("[q] : Quit", mixed[-2])
        #     key = cv2.waitKey(1) & 0xff
        #     ## "q" : プログラム終了
        #     if (key == ord("q")):
        #         cv2.destroyAllWindows()
        #         return
        
        for j, feature in enumerate(feature_maps):
            if (len(feature) <= 1): feature = [feature]
            for k, values in enumerate(feature):
                with open(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_{index:04d}_{j:02d}_{k:02d}.txt", "w") as f:
                    f.write("\n".join(map(lambda x: " ".join(map(str, x)), values)))
        
        with open(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_{index:04d}_image.txt", "w") as f:
            f.write("\n".join(map(lambda x: " ".join(map(str, map(lambda val: sum(val)/len(val), x))), img[0])))
            # with open(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_{index:04d}_{f:04d}_linear.txt", "w") as f:
                # f.write("\n".join(map(lambda x: " ".join(map(str, x)), feature)))
        # if not IS_BINARY:
            # with open(f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}/{SAVE_ID}_{index:04d}_otherinfo.txt", "w") as f:
                # f.write("classes " + " ".join(map(str, classes)))
        
        
        if not i: print("\n")
        
        ## 終了時刻の予測
        remining_time = timecounter.predictTime(index+1)
        cp.cprint(f"\033[1Acompleted : {index+1} / {datacount} - {remining_time}", "pink")
        
        index += 1
        
        # del func, feature_maps, explained, mixed, prediction
        
        gc.collect()
    
    if IS_REALTIME_PREVIEW: cv2.destroyAllWindows()
    
    ## GIF画像の保存
    # if IS_MAKE_GIF:
        # for i, g in enumerate(gifmaker):
            # g.save_gif(path=f"{SAVE_PATH}/{SAVE_ID}{SAVE_BUF}_{i:02d}.gif", duration=DURATION)
    
    return
    

def main():
    if (LIMIT < 0):
        cp.cprint("[!] \"LIMIT\" is too short or \"BATCH_SIZE\" is too large", "red")
    else:
        if MODEL_NAME is None:
            blank("MODEL_NAME")
        else:
            result = explain(MODEL_NAME)
            # cp.cprint(f"test result : {result}", "green")
            
    cp.cprint("- finished ! -", "cyan")
    

if (__name__ == "__main__"):
    main()
