from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image


# conc modules
from mymodel.SemanticSegmentation import SemanticSegmentation
# my module
from colorPrint import Cprint as cp


## ================ config ===================


## モデルの名前
MODEL_NAME = "unet"
## 入力画像のサイズ
# SIZE = [270, 270]
SIZE = [270*2, 270*3]
# SIZE = [256, 256]
## BATCH_SIZE : 1 only
BATCH_SIZE = 1
NUM_CLASSES = 2
CLASSES = ["before", "just"]
## 入力画像をグレースケールにするか否か
IS_GRAYSCALE = False
## 入力画像から平均画像を引くか否か
## (Semantic Segmentationに着手する前のモデル(NINやResNetなど)を用いる時はTrueにする)
IS_USE_AVERAGE_IMAGE = False

RANDOM_SEED = 1
## LIMIT : None or int
LIMIT = None

## マルチGPUにするか否か(入力の分割のみ)
USE_MULTI_GPU = False
## 入力画像をリサイズするか否か(上述の配列SIZEの値にリサイズ)
IS_RESIZE = False

## 読み込むモデルのパス
# LOAD_PATH = "/workspace/semanticSegmentation/result"
LOAD_PATH = "/workspace/fullframe/result"
LOAD_ID = "unet_20220222_prefullframe_finetuning_b2_e10_fold4"
# LOAD_ID = "pspnet_20220215_fullframe_b1_e10"

## 判定が反転する際の尤度の閾値：[ Just -> Before,  Before -> Just ]
THRESHOLD = [0.80, 0.75]

## 結果を保存するか否か
IS_OUTPUT_RESULT = True
## 保存先のパス
SAVE_ID = f"{LOAD_ID}_{int(THRESHOLD[0]*100)}-{int(THRESHOLD[1]*100)}"
SAVE_PATH = f"/workspace/semanticSegmentation/visualize/{SAVE_ID}"

## 画像に加算する色の値
BEFORE = [-50, -50, 100]
JUST = [-50, 100, -50]
## マスク画像のみを出力
IS_MASK_ONLY = False
SPLIT = 10

## フルフレームでの推論を可視化
IS_FULLFRAME = True
FULLFRAME_LIMIT = 405

## 取得する画像の時系列順での範囲(完了・未完了の境界付近の画像を取り出す目的)
TAKE_IMAGE_RANGE = [50, 250]

## データを２分割して実行
IS_SPLIT_2 = False
## 分割したうちのデータの番号(0 ro 1)
SPLIT_PART = 0

## ===========================================


def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def text_path(place:int) :
    if IS_FULLFRAME:
        if (place == 1):
            return "/workspace/Dataset/fullframe/text_dataset/all.txt"
        else:
            return ""
    else:
        return f"/workspace/Dataset/semanticSegmentation/text_dataset/visualize/take_image_{place:02d}.txt"
        
        
def save_dir(place:int):
    makedir(SAVE_PATH)
    if IS_FULLFRAME:
        return f"{SAVE_PATH}/{'mask_only/' if IS_MASK_ONLY else ''}"
    else:
        return f"{SAVE_PATH}/{'mask_only/' if IS_MASK_ONLY else ''}{place:02d}"
save_path = lambda index : f"{save_dir(index[0])}/{index[1]:04d}.png"
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")


def get_palette():
    """
    @機能：マスク画像から読み取る色の設定
    @引数：void
    @戻値：対象のRGBの値が入った配列
    """
    palette = [[0, 0, 0],       # Before
               [255, 255, 255]] # Just
    return np.asarray(palette)



def get_average_image():
    """
    @機能：平均画像の読み込み
    @引数：void
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
    if IS_FULLFRAME:
        img_buf = [None]*(img.shape[0])
        mask_buf = [None]*(mask.shape[0])
        for index, (i, m) in enumerate(zip(img, mask)):
            img_buf[index] = Image.fromarray(np.uint8(i))
            mask_buf[index] = Image.fromarray(np.uint8(m))
            img_buf[index].resize(SIZE)
            mask_buf[index].resize(SIZE)
            img[index] = np.asarray(img_buf[index])
            mask[index] = np.asarray(mask_buf[index])
            
    # if(np.max(img) > 1):
    #     img = img / 255.
    
    if IS_USE_AVERAGE_IMAGE:
        avg_img = get_average_image()
        img -= avg_img
    
    palette = get_palette()

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


def dataGenerator(resourcepath):
    
    image_path = []
    label = []
    with open(resourcepath) as f:
        readlines = f.readlines()
        if LIMIT: readlines = readlines[:LIMIT]
        if IS_FULLFRAME and FULLFRAME_LIMIT : readlines = readlines[:FULLFRAME_LIMIT]
    for line in readlines:
        buffer = line.split(" ")
        while not len(buffer[0]):
            buffer = buffer[1:]
        image_path.append(buffer[0])
        if IS_FULLFRAME:
            label.append(buffer[1].rstrip("\n"))
        else:
            label.append(int(buffer[1].rstrip("\n")))
    
    data_gen_args = dict(
        rescale=None
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
    image_generator = image_datagen.flow_from_dataframe(image_dataframe,
                                                        x_col="image",
                                                        target_size=SIZE.copy(),
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        classes=["before", "just"],
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    
    if IS_FULLFRAME:
        mask_datagen = ImageDataGenerator(**data_gen_args)
        mask_dataframe = pd.DataFrame(label, index=None, columns=["mask"])
        mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                        x_col="mask",
                                                        target_size=SIZE.copy(),
                                                        color_mode="grayscale" if IS_GRAYSCALE else "rgb",
                                                        classes=["before", "just"],
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
        
        for image, mask in zip(image_generator, mask_generator):
            image, mask = adjustData(image, mask)
            yield image, mask  
    
    for i, image in enumerate(image_generator):
        yield image, label[i]


def datacounter(datapath):
    with open(datapath) as f:
        readlines = f.readlines()
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


def createModel(model_name:str=None):
    
    model = None
    
    if model_name is None:
        blank("model_name")
            
    else:
        cp.cprint(f"- model : {model_name} -", "cyan")        
        if (model_name == "unet"):
            model = SemanticSegmentation.unet([*SIZE, 3])
        elif (model_name == "pspnet"):
            model = SemanticSegmentation.pspnet([*SIZE, 3])
        else:
            cp.cprint(f"[!] {model_name} is not defined.", "red")
        
    return model


def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def colorAdder(color, val):
    color += val
    for i, c in enumerate(color):
        if (c > 255): color[i] = 255
        elif (c < 0): color[i] = 0
    return color


def test(model_name:str=None):
    if model_name is None:
        blank("model_name")
        return None
    
    model = createModel(model_name)
    model.load_weights(f"{LOAD_PATH}/{LOAD_ID}/{LOAD_ID}.h5")
    
    print()
    
    collected = 0; accum = 0
    for place in range(1 if IS_FULLFRAME else 12 if IS_SPLIT_2 else 24):
        if IS_SPLIT_2 and SPLIT_PART: place += 12
        makedir(save_dir(place+1))
        datacount = datacounter(text_path(place+1))     
        cp.cprint(f"- place : {place+1} / {24} -\n", "cyan")
        if not place: print()
        index = 0
        # owatta arigatou!!!hai!!!
        for i, (img, label) in enumerate(dataGenerator(text_path(place+1))):
            if (i >= datacount): break
            if i%SPLIT: continue
            index += 1
            if IS_FULLFRAME:
                if (index > FULLFRAME_LIMIT): break
            else:
                if (index <= TAKE_IMAGE_RANGE[0]): continue
                if (index > TAKE_IMAGE_RANGE[1]): break
            predict_map = [[0]*SIZE[1] for _ in range(SIZE[0])]
            predict = model.predict(img/255., batch_size=BATCH_SIZE)
            for j, batch in enumerate(predict):
                for k, row in enumerate(batch):
                    for l, pixel in enumerate(row):
                        if (pixel[1] > THRESHOLD[1]): predict_map[k][l] = 1
                        elif (pixel[0] > THRESHOLD[0]): predict_map[k][l] = 0
                        if IS_MASK_ONLY: img[j][k][l] = np.array([0, 255, 0]) if predict_map[k][l] else np.array([0, 0, 255])
                        else: img[j][k][l] = colorAdder(img[j][k][l], JUST if predict_map[k][l] else BEFORE)
                        if IS_FULLFRAME: collected += int(predict_map[k][l] == label[j][k][l][1])
                        else: collected += int(predict_map[k][l] == label)
                        accum += 1
            cv2.imwrite(save_path([place+1, index-1]), *img)
            cp.cprint(f"\033[1Acomcpleted : {index-1} / {datacount//SPLIT}  -  threshold : ( {THRESHOLD[0]}, {THRESHOLD[1]} )  -  accuracy : {round(collected/accum, 2)}   ", "green")
        print()
    print()
    
    return collected/accum
    

def main():
    result = test(MODEL_NAME)
    cp.cprint(f"accuracy : {result}", "green")
    cp.cprint("- finished ! -", "cyan")
    

if (__name__ == "__main__"):
    main()
