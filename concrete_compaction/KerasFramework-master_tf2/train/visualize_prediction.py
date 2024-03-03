import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ================


DIR = "/workspace/fullframe/result/540x540"
TEXT = lambda fold: f"/workspace/Dataset/fullframe/text_dataset/fold{fold}/test.txt"
SAVE_DIR = "/workspace/visualization/fullframe"

## 読み込むモデルの設定
## [id, available, avg_img]
LOAD_ID = [
    # lambda fold: [f"e-unet_4class_adam_dropout_20220805_AutoLearning_fold{fold}_576x576", 0, 1],
    lambda fold: [f"e-unet_4class_adam_dropout_20220805_AutoLearning_fold{fold}_576x576", 1, 1],
    # lambda fold: [f"e-unet_20220629_AutoLearning_fold{fold}_576x576", 1, 1],
    # lambda fold: [f"unet_20220319_AutoLearning_fold{fold}_540x540", 0, 0]
    ]
FOLD = 1

## 画像の読込と保存に関する設定
# SIZE = int(LOAD_ID[-3:])
SKIP_IMAGE_INTERVAL = 25
GIF_DURATION_MSEC = 100
OUTPUT_IMAGE_SIZE = 180*4
IMAGE_FORMAT = "png"


## ========================================


def excute():
    """
    @機能：推論結果の可視化
    @引数：void
    @戻値：none
    """
    
    for load_id_func in LOAD_ID:
                
        load_id, available, use_avg_img = load_id_func(FOLD)
        
        if not available: continue
        
        size = int(load_id[-3:])
        
        cp.cprint(f"\n@ Load a model : {load_id}\n", "pink")
        
        Utils.makedir(f"{SAVE_DIR}/{load_id}")
        Utils.makedir(f"{SAVE_DIR}/{load_id}/image")
        Utils.makedir(f"{SAVE_DIR}/{load_id}/covered")
        Utils.makedir(f"{SAVE_DIR}/{load_id}/mask")
        Utils.makedir(f"{SAVE_DIR}/{load_id}/gif")
        
        try:
            model = load_model(f"{DIR}/{load_id}")
        except:
            model = load_model(f"{DIR}/{load_id}/{load_id}.h5")
        
        with open(TEXT(FOLD)) as f:
            readlines = f.readlines()
        length = len(readlines) // SKIP_IMAGE_INTERVAL
        
        key_tmp = None
        gif_images = []
        
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines[::SKIP_IMAGE_INTERVAL])):
            image, answer, *fresh = line.split(" ")
            
            img = Image.open(image)
            img = img.resize((size, size))
            img = np.array(img, dtype=np.float32)
            
            avg_img = im.get_average_image((size, size)) if use_avg_img else 0
            predict ,= model.predict([np.reshape(img-avg_img, (1, size, size, 3)) / 255.])
            
            ## 4クラス分類用の処理
            if (predict.shape[2] == 4):
                tmp = np.zeros((predict.shape[0], predict.shape[1], 2))
                tmp[:, :, 0] += predict[:, :, 0] + predict[:, :, 1]
                tmp[:, :, 1] += predict[:, :, 2] + predict[:, :, 3]
                del predict
                predict = np.copy(tmp)
                del tmp
            
            ## 推論結果を局所表現ベクトルに変換
            predict = np.uint8(np.argmax(predict, axis=2))
            
            
            img = np.uint8(img)
            origin = np.copy(img)
            
            img //= 2
            img[:, :, 0] += (predict^1)*100
            img[:, :, 1] += (predict)*100

            ## 画像の保存
            key = im.get_image_key(image, is_fullframe=True)

            img = Image.fromarray(img)
            img = img.resize((OUTPUT_IMAGE_SIZE//4*6, OUTPUT_IMAGE_SIZE))
            img.save(f"{SAVE_DIR}/{load_id}/covered/{key}.{IMAGE_FORMAT}")
            
            origin = Image.fromarray(origin)
            origin = origin.resize((OUTPUT_IMAGE_SIZE//4*6, OUTPUT_IMAGE_SIZE))
            origin.save(f"{SAVE_DIR}/{load_id}/image/{key}.{IMAGE_FORMAT}")
            
            ## 推論結果のみの保存
            pred_img = np.zeros((predict.shape[0], predict.shape[1], 3))
            pred_img[:, :, 0] += (predict^1)*224
            pred_img[:, :, 1] += predict*224
            pred_img[:, :, 2] += 32
            pred_img = Image.fromarray(np.uint8(pred_img))
            pred_img = pred_img.resize((OUTPUT_IMAGE_SIZE//4*6, OUTPUT_IMAGE_SIZE))
            pred_img.save(f"{SAVE_DIR}/{load_id}/mask/{key}.{IMAGE_FORMAT}")
            
            ## GIFの作成と保存
            master_key = im.get_image_key(image, without_mesh_id=True, is_fullframe=True)
            if key_tmp is None:
                key_tmp = master_key
            if (master_key != key_tmp):
                gif_images[0].save(f"{SAVE_DIR}/{load_id}/gif/{key_tmp}.gif",
                                save_all=True,
                                append_images=gif_images[1:],
                                duration=GIF_DURATION_MSEC,
                                loop=True)
                gif_images = []
                cp.cprint(f"@ Saved a GIF image. ( key : {key_tmp} )\n", "cyan")
            
            gif_images.append(img)
            key_tmp = master_key
        
            cp.cprint(f"\033[1A{i+1} / {length} ( key : {key} )   ", "orange")
        
        ## 剰余分のGIFを保存
        gif_images[0].save(f"{SAVE_DIR}/{load_id}/gif/{key_tmp}.gif",
                            save_all=True,
                            append_images=gif_images[1:],
                            duration=GIF_DURATION_MSEC,
                            loop=True)
        cp.cprint(f"@ Saved a GIF image. ( key : {key_tmp} )\n", "cyan")
        
        del model



def excute_answer_only():
    """
    @機能：正解ラベルのみ可視化
    @引数：void
    @戻値：none
    """
    
    for load_id_func in LOAD_ID:
        
        for fold in range(1, 6):
            
            load_id, available, use_avg_img = load_id_func(fold)
            
            if not available: continue
            
            size = int(load_id[-3:])
            
            cp.cprint(f"\n@ Load a model : {load_id}\n", "pink")
            
            Utils.makedir(f"{SAVE_DIR}/")
            Utils.makedir(f"{SAVE_DIR}/answer")
            Utils.makedir(f"{SAVE_DIR}/answer_gif")
            
            with open(TEXT(fold)) as f:
                readlines = f.readlines()
            length = len(readlines) // SKIP_IMAGE_INTERVAL
            
            key_tmp = None
            gif_images = []
            
            for i, line in enumerate(map(lambda x: x.rstrip("\n"), readlines[::SKIP_IMAGE_INTERVAL])):
                image, answer, *fresh = line.split(" ")
                
                img = Image.open(image)
                img = img.resize((size, size))
                img = np.array(img, dtype=np.float32)
                
                ans = Image.open(answer)
                ans = ans.resize((size, size))
                ans = np.uint8(np.copy(np.array(ans, dtype=np.float32)[:, :, 1]))
                ans = np.clip(ans, 0, 1)
                
                img = np.uint8(img)
                img //= 2
                img[:, :, 0] += (ans^1)*100
                img[:, :, 1] += (ans)*100
                
                img = np.clip(img, 0, 255)
                
                img = Image.fromarray(img)
                img = img.resize((OUTPUT_IMAGE_SIZE//4*6, OUTPUT_IMAGE_SIZE))
                                
                ## 画像の保存
                key = im.get_image_key(image, is_fullframe=True)
                master_key = im.get_image_key(image, without_mesh_id=True, is_fullframe=True)
                Utils.makedir(f"{SAVE_DIR}/answer/{master_key}")
                img.save(f"{SAVE_DIR}/answer/{master_key}/{key}.{IMAGE_FORMAT}")
                
                ## GIFの作成と保存
                if key_tmp is None:
                    key_tmp = master_key
                if (master_key != key_tmp):
                    Utils.makedir(f"{SAVE_DIR}/answer_gif/{master_key}")
                    gif_images[0].save(f"{SAVE_DIR}/answer_gif/{master_key}/{key_tmp}.gif",
                                    save_all=True,
                                    append_images=gif_images[1:],
                                    duration=GIF_DURATION_MSEC,
                                    loop=True)
                    gif_images = []
                    cp.cprint(f"@ Saved a GIF image. ( key : {key_tmp} )\n", "cyan")
                
                gif_images.append(img)
                key_tmp = master_key
            
                cp.cprint(f"\033[1A{i+1} / {length} ( key : {key} )   ", "orange")
            
            ## 剰余分のGIFを保存
            Utils.makedir(f"{SAVE_DIR}/answer_gif/{master_key}")
            gif_images[0].save(f"{SAVE_DIR}/answer_gif/{master_key}/{key_tmp}.gif",
                                save_all=True,
                                append_images=gif_images[1:],
                                duration=GIF_DURATION_MSEC,
                                loop=True)
            cp.cprint(f"@ Saved a GIF image. ( key : {key_tmp} )\n", "cyan")
            

def main():
    excute()
    # excute_answer_only()


main()
