import cv2
from datetime import datetime
from matplotlib.pyplot import axis
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time

# my module
from train.mymodel.SemanticSegmentation import SemanticSegmentation
from train.colorPrint import Cprint as cp

## ================ config ================

IS_HIDE_SUMMARY = True

MODEL_NAME = "unet"
INPUT = [270*2, 270*2, 3]
OUTPUT = [270*8, 270*6]
OUTPUT_COEF = [4, 4]
# SKIP_FRAME = 20
SKIP_FRAME = 0

## 確率をそのまま描写するか否か
IS_DIRECT_PREDICTION = False
AVERAGE_IMAGE_PATH = "/workspace/osada_ws/average_image_0516.png"

LOAD_DIR = "/workspace/fullframe/result/540x540"
LOAD_ID = "unet_20220324_realtime_AutoLearning_fold3_540x540"

SAVE_DIR = "/workspace/osada_ws/KerasFramework-master_tf2/saved_frame"

## 画像に加算する色の値
# BEFORE = [-50, -50, 100]
# JUST = [-50, 100, -50]

IS_DOCKER = True
IS_GRAYSCALE = False
IS_USE_AVERAGEIMAGE = True

## SLEEP_DELAY = None or float
SLEEP_DELAY = None

VIDEO_INPUT = 0
# VIDEO_INPUT = "/workspace/video/compaction_video_190731/CM190731_01ab.MTS"
# VIDEO_INPUT = "/workspace/video/compaction_video_220210/CM220210_12ab.MTS"


## ========================================


if not IS_DOCKER:
    LOAD_DIR = LOAD_DIR.replace("/workspace", "/media/nagalab/SSD1.7TB/nagalab/osada_ws")
    SAVE_DIR = SAVE_DIR.replace("/workspace", "/media/nagalab/SSD1.7TB/nagalab/osada_ws")

save_path = lambda tag : f"{SAVE_DIR}/{tag}.png"
# エラー表示
blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")


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

        ## モデルの振り分け
        if (model_name == "unet"):
            model = SemanticSegmentation.unet(INPUT)
        elif (model_name == "unet_fresh"):
            model = SemanticSegmentation.unet_include_fresh(INPUT)
        elif (model_name == "pspnet"):
            model = SemanticSegmentation.pspnet(INPUT)
        else: cp.cprint(f"[!] {model_name} is not defined.", "red")
        
    return model


def get_average_image():
    """
    @機能：平均画像の読み込み
    @引数：isGrayScale:bool = グレースケールで読み込むか否か
    @戻値：平均画像のデータが格納された配列
    """
    if IS_GRAYSCALE:
        avg_img = img_to_array(load_img(AVERAGE_IMAGE_PATH, color_mode='grayscale', grayscale=True))
    else:
        avg_img = img_to_array(load_img(AVERAGE_IMAGE_PATH, color_mode='rgb'))

    return avg_img[::-1]*IS_USE_AVERAGEIMAGE


def capture(model:str=None, weights:str=None):
    """
    @機能：モデルの読み込みと、リアルタイム判定
    @引数：model = 学習済のモデル, weights = 読み込む重みファイルのパス
    @戻値：None
    """
    try:
        if not IS_HIDE_SUMMARY: model.summary()
        ## 重みの読み込み
        model.load_weights(weights)
    except:
        cp.cprint("[!] \"model_name\" or \"weights\" is incorrect.", "red")
        return
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    
    cp.cprint(f"YYYY/MM/DD hh:mm:ss\tdelay : [s]\tfps [Hz]\tfps average [Hz]", "green")
    
    average_image = (cv2.resize(get_average_image(), dsize=(INPUT[0], INPUT[1])))/255.
    count = 0; fps_accum = 0; fps_avg = 0
    # old_pred
    
    ## キャプチャ開始
    while True:
        start = time.time()
        count += 1
        
        ret, frame = cap.read()
        for _ in range(SKIP_FRAME):
            ret, frame = cap.read()
        if not ret:
            cp.cprint("[!] Could not read a frame.", "red")
            break

        frame = cv2.resize(frame, dsize=(INPUT[0], INPUT[1]))/255.
        
        ## 推論
        prediction ,= model.predict(np.array([frame]) - average_image, batch_size=1)
        
        ## 尤度の値を色に直接加減算して、画面に出力する
        frame[:,:,0] = 0
        frame /= 2.
        
        # pred_label = np.argmax(prediction, axis=2)
        
        if IS_DIRECT_PREDICTION:
            ## Before
            frame[:,:,2] += (prediction[:,:,0])/2.
            frame[:,:,1] -= (prediction[:,:,0])/2.
            # frame[:,:,2] += ((prediction[:,:,0]*(pred_label^1))**3.)/2.
            # frame[:,:,1] -= (pred_label^1)/2.
            
            ## Just
            frame[:,:,2] -= (prediction[:,:,1])/2.
            frame[:,:,1] += (prediction[:,:,1])/2.
            # frame[:,:,1] += ((prediction[:,:,1]*pred_label)**3.)/2.
            # frame[:,:,2] -= (pred_label)/2.

        ## 尤度を0, 1で２極化させて、画面に出力する
        else:
            ## 尤度を0, 1に２極化
            prediction = np.argmax(prediction, axis=2)
            frame[:,:,1] += prediction-0.5
            frame[:,:,2] += (prediction^1)-0.5
            
            old_pred = np.copy(prediction)
        
        ## BGR値を0~1に丸め込む
        frame = np.clip(frame, 0, 1)
        
        ## FPS値の計算  
        end = time.time()
        delay = round(end-start, 2)
        fps = 1/delay
        if (count > 5):
            fps_accum += fps
            fps_avg = fps_accum/(count-5)
        
        ## 画像の表示
        cv2.imshow("quit : press \"q\"", cv2.resize(frame, dsize=OUTPUT))
        
        ## FPS値などの表示
        cp.cprint(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\tdelay : {delay}\tfps : {round(fps, 2)}\tfps average : {round(fps_avg, 2)}", "cyan")

        key = cv2.waitKey(1) & 0xff
        ## "q" : プログラム終了
        if (key == ord("q")):
            break
        ## "s" : 画面保存
        elif (key == ord("s")):
            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(save_path(date), cv2.resize(frame*255, dsize=OUTPUT))
            cp.cprint(f"Saved a frame! ( {save_path(date)} )", "pink")
        
        if SLEEP_DELAY is not None: time.sleep(SLEEP_DELAY)

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = createModel(MODEL_NAME)
    capture(model, f"{LOAD_DIR}/{LOAD_ID}/{LOAD_ID}.h5")


if(__name__ == "__main__"):
    main()
