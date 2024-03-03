import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from BetweenclassLearning import BCL
from line_sender import send_master
from mymodel.ClassDivision import ClassDivider
from mymodel.MyModel import MyModel
from mymodel.CreateModel import CreateModel

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel("ERROR")


## ================ config ================


DIR = "/workspace/fullframe/result/class_divide"
# TEXT_DIR = "/workspace/osada_ws/text_dataset/ngc_docker"
TEXT_DIR = "/workspace/Dataset/fullframe/text_dataset/fold4"
AVERAGE_IMAGE_PATH = '/workspace/osada_ws/average_image_0516.png'
SIZE = [270, 270]
# SIZE = [576, 576]
# RESULT_PATH = "/workspace/fullframe/result/class_divide/four_class_time_20220704"
RESULT_PATH = "/workspace/fullframe/result/class_divide/four_class_time_e-unet_20220705"
RESNET = "/workspace/osada_ws/gray_resnet18_initBC_metric_lr5_f5_39.h5"
EUNET = "/workspace/fullframe/result/540x540/e-unet_20220629_AutoLearning_fold4_576x576"
EUNET_END_LAYER = "add_50"
# BATCH_SIZE = 1<<15
BATCH_SIZE = 4
EPOCHS = 100
CLASSES = 4
RANDOM_SEED = 1
IS_SHUFFLE = False

MODE = "resnet"

BORDER_COEF = .1


## ========================================


text = lambda target: f"{TEXT_DIR}/{target}_include_fresh.txt"


def make_input(text_path):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    image_path, ans_data, fresh_data, image_time = [], [], [], []
    length = 0; data = dict(); border = dict()    
    labeling_model = None
    pca_model = PCA(n_components=4)
    
    if (MODE in ["eunet"]):
        labeling_model = load_model(EUNET)
        for end_number, layer in enumerate(labeling_model.layers):
            if (layer.name == EUNET_END_LAYER): break
        labeling_model = Model(inputs=labeling_model.input, outputs=labeling_model.layers[end_number].output)
        cp.cprint("model : E-Unet", "orange")
    
    elif (MODE in ["resnet"]):
        createmodel = CreateModel(None)
        labeling_model = createmodel.create_model("resnet18", isMlt=True)
        labeling_model = Model(inputs=labeling_model.layers[1].input, outputs=labeling_model.layers[-1].output)
        labeling_model.trainable = False
        labeling_model.compile(optimizer=Adam(lr=0.001),
                        loss="categorical_crossentropy")
        labeling_model.load_weights(RESNET)
        labeling_model.summary()
        cp.cprint("model : ResNer18", "orange")
    
    with open(text_path, mode="r") as f:
        readlines = list(sorted(f.readlines()))
        for line in map(lambda x: x.rstrip("\n"), readlines):
            if len(line):
                ## 読み込んだテキストの解釈
                image, ans, *fresh = line.split(" ")
                image_path.append(image)
                # ans_data.append(int(ans))
                ans_data.append(ans)
                fresh_data.append(list(map(float, fresh)))
                
                length += 1
                
                ## 画像のパスからIDを抽出
                if (MODE in ["resnset"]):
                    img_info = ((image.split("/"))[7]).replace(".jpg", "").replace("CM", "")
                    img_day, img_place, img_id, img_num = img_info.split("_")
                    ans = int(ans)
                    key = f"{img_day}-{img_place}-{img_num}"
                
                elif (MODE in ["eunet"]):                
                    img_info_list = image.split("/")
                    img_info = "_".join([img_info_list[4], img_info_list[5], (img_info_list[6]).replace(".png", "")])
                    img_day, img_val, img_place, img_id = img_info.split("_")
                    key = f"{img_day}-{img_val}-{img_place}"
                
                ## Before, Boundry, Justのアンカーとなる画像を探索
                # if not key in data.keys(): data[key] = [[1e6, -1], [1e6, -1]]
                # if int(img_id) < int(data[key][int(ans)][0]):
                #     data[key][int(ans)][0] = img_id
                # if int(img_id) > int(data[key][int(ans)][1]):
                #     data[key][int(ans)][1] = img_id
        
        #### =================================================================================
        
        if False:
            for key, val in data.items():
                ## 4クラスの閾値を算出
                before_range, just_range = val
                video_length = int(just_range[1]) - int(before_range[0])
                boundry_length = video_length / 4
                center = (int(before_range[1]) + int(just_range[0])) / 2
                
                # border[key] = [0]*4
                # border[key][0] = (int(before_range[0])+int(before_range[1]))/2
                # border[key][1] = (int(before_range[1])+int(just_range[0]))/2
                # border[key][2] = (int(just_range[0])+int(just_range[1]))/2
                # border[key][3] = int(just_range[1])*1.1
                
                border[key] = [(center - boundry_length),
                            center,
                            (center + boundry_length),
                            int(just_range[1])*1.1]
        
        for i, (path, ans) in enumerate(zip(image_path, ans_data)):
            ## 画像のパスからIDを抽出
            img_info_list = path.split("/")
            
            ## E-Unet
            if (MODE in ["eunet"]):
                img_info ="_".join([img_info_list[4], img_info_list[5], (img_info_list[6]).replace(".png", "")])
                img_day, img_val, img_place, img_id = img_info.split("_")
                key = f"{img_day}-{img_val}-{img_place}"
                
            ## ResNet
            elif (MODE in ["resnet"]):
                img_info = (img_info_list[7]).replace(".jpg", "").replace("CM", "")
                img_day, img_place, img_id, img_num = img_info.split("_")
                ans = int(ans)
                key = f"{img_day}-{img_place}-{img_num}"
                
            ## 画像の時間的座標を取得
            # time_buf = 0
            # if (int(img_id) <= int(data[key][1][0])):
            #     now = int(img_id) - int(data[key][0][0])
            #     time_range = int(data[key][0][1]) - int(data[key][0][0])
            #     time_buf = 1 - (now / time_range)
            # else:
            #     now = int(img_id) - int(data[key][1][0])
            #     time_range = int(data[key][1][1]) - int(data[key][1][0])
            #     time_buf = now / time_range
            # image_time.append(time_buf)
            
            ## 学習済みE-Unetを使用するバージョン(k-Means)
            if True:
                eunet_image = Image.open(path)
                eunet_image = eunet_image.resize(SIZE)
                eunet_image = np.array(eunet_image, dtype=np.float32)
                eunet_image = np.reshape(eunet_image, (1, *eunet_image.shape))
                pred ,= labeling_model.predict([eunet_image])
                pred_range = list(zip(list(np.min(pred, axis=(0, 1))), list(np.max(pred, axis=(0, 1)))))
                vectors = pca_model.fit_transform(pred)
                cp.cprint(vectors, "random", bloom=2)
                # points = [[None]*len(pred_range)]
                # for dim, minmax in enumerate(pred_range):
                #     inf, sup = minmax
                #     x, y = random.random()*(sup-inf), random.random()*(sup-inf)
                #     while (x == y): y = random.random()*(sup-inf)
                #     points[dim] = [x, y]
                
            ## 学習済みResNetを使用するバージョン
            if False:
                # model = load_model(RESNET)
                
                # model = MyModel.resnet18((270, 270, 1), 2)
                # for layer in model.layers:
                    # print(layer.name)
                print()
                for i, path in enumerate(image_path):
                    resnet_image = np.array(Image.open(path).convert("L"), dtype=np.float32)
                    pred = model.predict([resnet_image])
                    cp.cprint(f"\033[1A{i} / {length}, {pred}")
            
            ## 自分で決めた閾値を使うバージョン(学習の呈をなさない可能性あり)
            if False:
                for label, id_ in enumerate(border[key]):
                    if (int(img_id) < id_): break
                ans_data.append(label)
                # if not i%100:
                    # print(f"{cp.colored(' '*10, background=['red', 'green'][ans])}{cp.colored(' '*10, background=['red', 'pink', 'yellow', 'green'][label])}")
            
            ## 内積を使うバージョン
            if False:
                ## 各画像のIDを取得
                before_key = data[key][0][0]
                boundry_key = data[key][0][1]
                just_key = data[key][1][1]

                ## 各画像のパスを取得
                before_path = f"{('/'.join(img_info_list[:7]).replace('just', 'before'))}/CM{img_day}_{img_place}_{before_key}_{img_num}.jpg"
                boundry_path = f"{('/'.join(img_info_list[:7]).replace('just', 'before'))}/CM{img_day}_{img_place}_{boundry_key}_{img_num}.jpg"
                just_path = f"{('/'.join(img_info_list[:7]).replace('before', 'just'))}/CM{img_day}_{img_place}_{just_key}_{img_num}.jpg"

                ## 各画像の値を取得
                img = np.array(Image.open(path), dtype=np.float32)
                before_img = np.array(Image.open(before_path), dtype=np.float32)
                boundry_img = np.array(Image.open(boundry_path), dtype=np.float32)
                just_img = np.array(Image.open(just_path), dtype=np.float32)
                
                ## 一次元化(平坦化)
                img = np.ravel(img)
                before_img = np.ravel(before_img)
                boundry_img = np.ravel(boundry_img)
                just_img = np.ravel(just_img)
                
                ## 正規化(適当。厳密さ必要？)
                maximum = max([np.max(img), np.max(before_img), np.max(boundry_img), np.max(just_img)])            
                img /= maximum; before_img /= maximum; boundry_img /= maximum; just_img /= maximum

                ## cos類似度を算出
                before_dot = Calc.cos_sim(img, before_img)
                boundry_dot = Calc.cos_sim(img, boundry_img)
                just_dot = Calc.cos_sim(img, just_img)
                
                ## ラベルを決定
                label = np.argmax(np.array([before_dot, boundry_dot, just_dot]))
            
                # print(f"{cp.colored(' '*10, background=['red', 'green'][ans])}{cp.colored(' '*10, background=['red', 'white', 'green'][label])}\t\t{before_dot}\t{boundry_dot}\t{just_dot}")
            
    
    return image_time, ans_data, fresh_data, length


def data_generator(image_time, ans_data, fresh_data, length, batch_size=1, is_shuffle=False):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    for e in range(EPOCHS+1):
        
        ## 要素のシャッフル
        if IS_SHUFFLE and is_shuffle:
            random.seed(RANDOM_SEED+e)
            random.shuffle(image_time)
            
            random.seed(RANDOM_SEED+e)
            random.shuffle(ans_data)
            
            random.seed(RANDOM_SEED+e)
            random.shuffle(fresh_data)            
        
        index = -1 * batch_size
        for batch in range(length//batch_size):
            ## ミニバッチを切り出せるか否かの検証
            try:
                _ = image_time[batch+batch_size]
            except:
                batch = 0
                
            index += batch_size
            
            ## ミニバッチの切り出し
            ## 画像とフレッシュ性状データ
            inputs = [None]*batch_size
            for i, (image, fresh) in enumerate(zip(image_time[batch*batch_size:(batch+1)*batch_size], fresh_data[batch*batch_size:(batch+1)*batch_size])):
                inputs[i] = np.array([image, *fresh], dtype=np.float32)
            
            ## 正解ラベル(one-hot vector化もする)
            answers = [[0]*CLASSES for _ in range(batch_size)]
            for i, ans, in enumerate(ans_data[batch*batch_size:(batch+1)*batch_size]):
                answers[i][ans] = 1
            
            inputs = np.array(inputs, dtype=np.float32)
            answers = np.array(answers, dtype=np.float32)
            
            yield inputs, answers
        
        
def class_divide():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    # model = ClassDivider.divide_classes_difftime(classes=CLASSES, dropout_const=0.25)
    # model.summary()
    
    # image_time_all, ans_data_all, fresh_data_all, length_all = make_input(text("all"))
    # train_gen = data_generator(image_time_all, ans_data_all, fresh_data_all, length_all, BATCH_SIZE, True)
    
    image_time_train, ans_data_train, fresh_data_train, length_train = make_input(text("train"))
    train_gen = data_generator(image_time_train, ans_data_train, fresh_data_train, length_train, BATCH_SIZE, True)
    
    # image_time_validation, ans_data_validation, fresh_data_validation, length_validaton = make_input(text("validation"))
    # validation_gen = data_generator(image_time_validation, ans_data_validation, fresh_data_validation, length_validaton, BATCH_SIZE, False)
    
    image_time_test, ans_data_test, fresh_data_test, length_test = make_input(text("test"))
    test_gen = data_generator(image_time_test, ans_data_test, fresh_data_test, length_test, BATCH_SIZE, False)
    
    return
    
    history = model.fit_generator(
                generator=train_gen,
                steps_per_epoch=length_train//BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=test_gen,
                validation_steps=length_test//BATCH_SIZE,
                shuffle=False,
            )
    
    model.save(f"{RESULT_PATH}")
    
    Utils.log(f"{RESULT_PATH}", history.history)
    
    acc = history.history["val_acc"][-1]*100
    cp.cprint(acc, "cyan")
    
    return history.history


def main():
    result = class_divide()
    print(result)


main()
