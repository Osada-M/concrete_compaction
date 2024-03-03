from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import random
from PIL import Image
import os
import time
import cv2
import copy

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Calc, TimeCounter
from ExplainableFuncs.ExplainableFuncs import Grad_CAM


## ================ config ===================


## ディレクトリの設定など(実行環境が変わった時以外、いじらない)
# WORKSPACE_DIR = "/workspace/semanticSegmentation"
# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
WORKSPACE_DIR = "/workspace/fullframe"
DATASET_DIR = "/workspace/Dataset/fullframe"
TEXT_DATASET = "/workspace/hidden/text_dataset.txt"
SAVE_DIR = "/workspace/hidden"
STATE_TEXT = "/workspace/osada_ws/state.txt"
IS_OUTPUT_LOG = True

## 学習の基本設定達
## BATCH_SIZEは1のまま
BATCH_SIZE = 1
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

SPLIT = 5
RANDOM_SEED = 1
## LIMIT : None or int
LIMIT = None

## マルチGPUにするか否か(入力の分割のみ)
USE_MULTI_GPU = False
## 入力画像をリサイズするか否か(上述の配列SIZEの値にリサイズ)
IS_RESIZE = True

## 結果を保存するか否か
IS_OUTPUT_RESULT = True

## 読み込むモデルの設定
# LOAD_PATH = "/workspace/semanticSegmentation/result"
LOAD_PATH = "/workspace/fullframe/result/540x540"
# LOAD_ID = "unet_fresh_20220314_b2_e20_fold5_540x540"
LOAD_ID = "unet_20220319_AutoLearning_fold1_540x540"
SAVE_ID = ""

## 適用する機能(linear, Grad-CAM)
FUNC = "Grad-CAM"

# OUTPUT_LAYER_NAME = "fresh_conv4"
OUTPUT_LAYER_NAME = "conv5m_2"

## フルフレームでテストを回すか否か
IS_FULLFRAME = True

## フレッシュ性状データを入力するか否か
IS_INCLUDE_FRESH = False
## フレッシュ性状データのカーネルサイズ(Encoderの出力部分と同じにする)
FRESH_KERNEL_SIZE = "auto"


## ===========================================

        
## エラー表示
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")
## 保存先のパス
save_path = lambda index : f"{SAVE_DIR}/{LOAD_ID}{'_' + SAVE_ID if SAVE_ID else ''}/hidden_{index}.png"

## フレッシュ性状データのカーネルサイズの調節
if isinstance(FRESH_KERNEL_SIZE, str):
    if (FRESH_KERNEL_SIZE == "auto"):
        FRESH_KERNEL_SIZE = list(map(lambda x: x//16, SIZE))
    else:
        cp.cprint("\"FRESH_KERNEL_SIZE\" must be either [int, int] or \"auto\".")
        LIMIT = -1


def makedir(path):
    """
    @機能：ディレクトリの新規作成（既にあれば作らない、無ければ作る）
    @引数：作りたいパス
    @戻値：None
    """
    if not os.path.isdir(path):
        os.mkdir(path)


## ログファイルの新規作成
if IS_OUTPUT_LOG:
    makedir(f"{SAVE_DIR}/{LOAD_ID}{'_' + SAVE_ID if SAVE_ID else ''}")
    with open(f"{SAVE_DIR}/{LOAD_ID}{'_' + SAVE_ID if SAVE_ID else ''}/result.txt", mode="w") as logoutput: pass


def log(strings:list):
    """
    @機能：ログファイルにログを出力
    @引数：strings;list = 出力したい数値などのリスト
    @戻値：None
    """
    with open(f"{SAVE_DIR}/{LOAD_ID}{'_' + SAVE_ID if SAVE_ID else ''}/result.txt", mode="a") as logoutput:
        print(strings, file=logoutput)


def output_state(color:str, string):
    """
    @機能：ステータスファイルに文字列を出力(get_state.pyでこの文字列を取得)
    @引数：color:str = 文字色, string = 出力する文字列
    @戻値：None
    """
    with open(STATE_TEXT, "w") as f:
        f.write(f"{color}\n{string}")
        
            
def get_palette():
    """
    @機能：マスク画像から読み取る色の設定
    @引数：void
    @戻値：対象のRGBの値が入った配列
    """
    palette = [[0, 0, 0],
               [255, 255, 255]]
    return np.asarray(palette)


def get_average_image():
    """
    @機能：平均画像の読み込み
    @引数：isGrayScale:bool = グレースケールで読み込むか否か
    @戻値：平均画像のデータが格納された配列
    """
    avgimg_path = '/workspace/osada_ws/average_image_0516.png'

    if IS_GRAYSCALE:
        avg_img = img_to_array(load_img(avgimg_path, color_mode='grayscale', grayscale=True))
    else:
        avg_img = img_to_array(load_img(avgimg_path, color_mode='rgb'))

    return avg_img


def adjustData(img, mask):
    """
    @機能：コンクリ画像を0~1に正規化、マスク画像をone-hot表現に変換
    @引数：img = コンクリ画像一枚, mask = マスク画像一枚
    @戻値：変換し終わったコンクリ画像, one-hot表現にしたマスク画像
    """
    
    ## 画像サイズをconfigで設定した値に調節
    if IS_FULLFRAME:
        img_buf = [None]*(img.shape[0])
        mask_buf = [None]*(mask.shape[0])
        for index, (i, m) in enumerate(zip(img, mask)):
            img_buf[index] = Image.fromarray(np.uint8(i))
            mask_buf[index] = Image.fromarray(np.uint8(m))
            img_buf[index].resize(SIZE)
            mask_buf[index].runet_20220302_prefullframe_finetuning_b2_e20_fold5_540x540
        if IS_USE_AVERAGE_IMAGE:
            img -= get_average_image()
        img /= 255.
    ## 平均画像を引いた後、画像データを0~1に正規化
    elif IS_USE_AVERAGE_IMAGE:
        avg_img = get_average_image()
        avg_img /= 255.    
        img -= avg_img

    ## ラベルの設定色を取得
    palette = get_palette()

    ## マスク画像を元に、正解ラベル(ont-hot表現)を作成する
    onehot = np.zeros((mask.shape[0], *SIZE, NUM_CLASSES), dtype=np.uint8)
    for i in range(2):
        cat_color = palette[i]
        temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                        (mask[:, :, :, 1] == cat_color[1]) &
                        (mask[:, :, :, 2] == cat_color[2]), 1, 0)
        onehot[:, :, :, i] = temp

    ## グレールスケールに変換
    if IS_GRAYSCALE:
        img = np.mean(img, axis=1)
        mask = np.mean(mask, axis=1)
    
    return img, onehot


def make_fresh_tensor(fresh:list, kernel_size:int, batch_size:int=None, isStr:bool=False):
    """
    @機能：フレッシュ性状データを学習用に整形
    @引数：fresh = フレッシュ性状データ, kernel_size = 変換後のカーネルサイズ, batch_size = バッチサイズ, isStr = freshがstring型であるか否か(FalseでOK)
    @戻値：shape:(kernel_size, kernel_size, len(fresh))に整形されたnumpy配列
    """
    if isStr: fresh = list(map(lambda x : list(map(float, x.split(" "))), fresh))
    if batch_size is None:
        output = [[fresh for col in range(kernel_size[1])] for row in range(kernel_size[0])]
    else:
        output = [[[fresh[batch] for col in range(kernel_size[1])] for row in range(kernel_size[0])] for batch in range(batch_size)]
    
    return np.asarray(output, dtype=np.float32)


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
        image, mask = adjustData(image, mask)
        if IS_INCLUDE_FRESH:
            fresh_buffer = []
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
            fresh = make_fresh_tensor(fresh=fresh_buffer,
                                      kernel_size=FRESH_KERNEL_SIZE.copy(),
                                      batch_size=len(fresh_buffer),
                                      isStr=False)
            yield [image, fresh], mask
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
        if (model_name == "unet"):
            model = SemanticSegmentation.unet([*SIZE, 1 if IS_GRAYSCALE else 3])
        elif (model_name == "unet_fresh"):
            model = SemanticSegmentation.unet_include_fresh([*SIZE, 1 if IS_GRAYSCALE else 3], [*FRESH_KERNEL_SIZE, 5])
        elif (model_name == "pspnet"):
            model = SemanticSegmentation.pspnet([*SIZE, 1 if IS_GRAYSCALE else 3])
        else: cp.cprint(f"[!] {model_name} is not defined.", "red")
        
    return model


def make_RGB_prefix(dim):
    """
    @機能：
    @引数：
    @戻値：
    """
    random.seed(RANDOM_SEED)
    colors = [[None]*3 for _ in range(dim)]
    for i in range(dim):
        buffer = colors[i].copy()
        while buffer in colors:
            buffer = [random.randint(0, 255) for _ in range(3)]
        colors[i] = buffer.copy()
        
    return colors


def convert_for_RGB(array, colors, counter):
    """
    @機能：
    @引数：array:list = 
    @戻値：RGB
    """
    result = [[None]*(array.shape[1]) for _ in range(array.shape[0])]
    for i, row in enumerate(array):
        for j, pixel in enumerate(row):
            index = np.argmax(pixel)
            result[i][j] = colors[index].copy()
            counter[index] += 1
    
    return np.array(result), counter
    

def test(model_name:str=None):
    """
    @機能：テストの実行
    @引数：mpdel_name = モデルの定義名
    @戻値：
    """
    
    ## モデルの定義
    if model_name is None:
        blank("model_name")
        return None
    
    ## 重みの読み込み
    model = createModel(model_name)
    model.load_weights(f"{LOAD_PATH}/{LOAD_ID}/{LOAD_ID}.h5")
    
    ## モデルの切り出し
    try:
        model = Model(inputs=model.input, outputs=model.get_layer(OUTPUT_LAYER_NAME).output)
    except:
        cp.cprint(f"[!] {OUTPUT_LAYER_NAME} is not found.", "red")
        return
    
    model.summary()
    
    ## 色の設定
    colors = make_RGB_prefix(model.output.shape[-1])
    datacount = datacounter(TEXT_DATASET)
    ## テストの進行状況のリアルタイム表示の初期化
    cp.cprint(f"\ncompleted : 0 / {datacount}\t--:--:--", "green")
    ## 開始時刻の保存
    timecounter = TimeCounter(datacount)
    ## 保存先のディレクトリの作成
    makedir(f"{SAVE_DIR}/{LOAD_ID}")
    counter = [0]*len(colors)
    
    ## テストの実行
    for i, (img, msk) in enumerate(dataGenerator(TEXT_DATASET)):
        if (i >= datacount): break
        predict = model.predict(img, batch_size=BATCH_SIZE)
        for j, batch in enumerate(predict):
            image, counter = convert_for_RGB(batch, colors, counter)
            cv2.imwrite(save_path(i), image)
        if not i: print("\n")
        cp.cprint(f"\033[1Acomcpleted : {i+1} / {datacount}\t{timecounter.predictTime(i+1)}{' '*30}", "green")
    print()
    
    return counter
    

def main():
    if LIMIT is not None and (LIMIT < 0):
        cp.cprint("[!] \"LIMIT\" is too short or \"BATCH_SIZE\" is too large", "red")
    else:
        if MODEL_NAME is None:
            blank("MODEL_NAME")
        else:
            result = test(MODEL_NAME)
            cp.cprint(f"test result : {result}", "green")
    
    log(result)
    output_state("yellow", f"finished visualizing for \"{LOAD_ID}\"\'s hidden layer.")
        
    cp.cprint("- finished ! -", "cyan")
    

if (__name__ == "__main__"):
    main()
