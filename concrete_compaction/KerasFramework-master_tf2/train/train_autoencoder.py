print("Train or Test : AutoEncoder")


from tabnanny import verbose
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import time

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from BetweenclassLearning import BCL
from line_sender import send_master
from my_loss_function import MyLosses
from mymodel.SemanticSegmentation import SemanticSegmentation
from mymodel.SemSegLight import E_UNet, ESPNet
from colorPrint import Cprint as cp
from luminance_extender import LuminanceExtender
from fine_tuner import HoG_Model


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")

DATASET_DIR = "/workspace/Dataset/fullframe"
RESULT_DIR = "/workspace/fullframe/result/autoencoder"
# FOLD = 3
BATCH_SIZE = 8
EPOCHS = [5, 50] # 50
# SIZE = [576, 576]
SIZE = [128, 128]
LIMIT = None
# LIMIT = 48*4
RANDOM_SEED = 1

# SAVE_ID = lambda skip: "20221014_ssim_2x"
# SAVE_ID = lambda skip: "20221014_ssim_mse"
# SAVE_ID = lambda skip: "20221014-1"
# SAVE_ID = lambda skip: f"20221025_rec-mse_skip-connection-{skip}"
# LOSS = MyLosses.rectified_mse_loss
# SAVE_ID = "rotate_20221014_ssim_mse"
# SAVE_ID = f"20221209_ssim_mse"
SAVE_ID = f"20230101_ssim_mse"
# SAVE_ID = lambda skip: "flip_rotate_20221014_ssim_mse"
# LOSS = MyLosses.ssim_loss
LOSS = MyLosses.ssim_mse_loss


# save_path = lambda *fold: f"{RESULT_DIR}/AE_e-unet-light_{SAVE_ID(fold[0])}_fold{fold[1]}"
# save_path = lambda *fold: f"{RESULT_DIR}/AE_e-unet_reduce-{str(fold[1]).replace('.', '-')}_{SAVE_ID}_fold{fold[0]}"
save_path = lambda *fold: f"{RESULT_DIR}/AE_espnet_{SAVE_ID}_fold{fold[0]}"
text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"

## データ数の調整
# if LIMIT is None:
#     with open(path("train")) as f:
#         data_length = len(f.readlines())
#         LIMIT = data_length - (data_length%BATCH_SIZE)
    
    
def dataGenerator(resourcepath:str, noise:bool=True, test:bool=False, noise_type:str="linear", is_flip:bool=False, is_rotate:bool=False):
    
    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        if not test:
            random.seed(RANDOM_SEED)
            random.shuffle(readlines)
        
        data_length = len(readlines)
        limit = data_length - (data_length%BATCH_SIZE)
        readlines = readlines[:limit]
        
    for line in map(lambda x: x.rstrip("\n"), readlines):
        linebuffer = line.split(" ")
        image_path.append(linebuffer[0])
        mask_path.append(linebuffer[1])
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
                                                        color_mode="rgb",
                                                        classes=["red", "green", "blue"],
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE**(not test),
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="rgb",
                                                      classes=["red", "green", "blue"],
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE**(not test),
                                                      shuffle=False)
    
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        noised, origin = im.adjust_data(
            image, mask, is_fullframe=True, size=SIZE.copy(), classification="fourclasses", num_classes=4,
            autoencoder=True, noise=noise, noise_type=noise_type, is_flip=is_flip, is_rotate=is_rotate
            )
        
        yield noised, origin
    
    
def check_noise(fold:int=1, noise_type:str="linear", limit:int=20):
    
    for i, (noised, image) in enumerate(dataGenerator(text_path(fold, "train"), noise_type=noise_type, is_flip=True, is_rotate=True)):
        if (i >= limit): break
        
        # noised = np.uint8(((noised+1.)/2.)*255.)[0]
        noised = np.uint8(noised*255.)[0]
        noised = Image.fromarray(noised)

        save_dir = f"/workspace/visualization/luminance/random_mesh/{noise_type}_{SAVE_ID}"
        Utils.makedir(save_dir)
        noised.save(f"{save_dir}/{i}.png")
        
        print(f"{i+1} / {limit}")
    
    
def train(fold, noise_type="linear", load_pre=False, skip_connection=3, reduce_const=1):
    
    datacount = Utils.datacounter(text_path(fold, "train"))

    if not load_pre:
        
        path = save_path(fold, reduce_const)
        
        Utils.makedir(path)
        Utils.makedir(f"{path}/pre-model")
        cp.cprint(f"@ save path : {path}", "orange")
        
        # model = E_UNet.run((576, 576, 3), num_classes=3, loss="binary_crossentropy", autoencoder=True)
        # model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True, skip_connection=skip_connection, reduce_const=reduce_const)
        model = ESPNet.run((128, 128, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True)
        # model.summary()
        
        # return
        
        cp.cprint("Pre Learn", "green")
        model.fit_generator(
            generator=dataGenerator(text_path(fold, "train"), noise=False, noise_type=noise_type),
            steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
            epochs=EPOCHS[0],
            validation_data=dataGenerator(text_path(fold, "validation"), noise=False, noise_type=noise_type),
            validation_steps=100,
            shuffle=False,
            )
        
        model.save(f"{path}/pre-model")
        cp.cprint("@ saved pre-model", "orange")
    
    else:
        model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True)
        model.load_weights(f"{path}/pre-model")
        cp.cprint("@ loaded pre-model", "orange")
        
    cp.cprint("Noise Learn", "green")
    history = model.fit_generator(
        generator=dataGenerator(text_path(fold, "train"), noise_type=noise_type),
        steps_per_epoch=int(np.ceil(datacount/BATCH_SIZE)),
        epochs=EPOCHS[1],
        validation_data=dataGenerator(text_path(fold, "validation"), noise_type=noise_type),
        validation_steps=100,
        shuffle=False,
        )
    
    print(history.history)
    model.save(path)
    
    with open(f"{path}/log.txt", mode="a") as f:
        print(history.history, file=f)


def test(fold, noise_type="linear", skip_connection=3, reduce_const=1):
    
    # for fold in [1, 2, 4, 5]:
            
    # model = load_model(save_path(fold))
    path = save_path(fold, reduce_const)
    # model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True, skip_connection=skip_connection, reduce_const=reduce_const)
    model = ESPNet.run((128, 128, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True)
    model.load_weights(path)
    cp.cprint(f"@ load path : {path}", "orange")
    
    length = Utils.datacounter(text_path(fold, "test"))
    result_1d = []
    result_2d = []
    
    Utils.makedir(f"{path}/test_noise_{noise_type}")
    Utils.makedir(f"{path}/test_pred_{noise_type}")
    
    timecounter = TimeCounter(length)
    index = 0
    
    for i, (noised, origin) in enumerate(dataGenerator(text_path(fold, "test"), test=True, noise_type=noise_type)):
        if (i >= length): break
        if i%50: continue
        
        pred ,= model.predict([noised], verbose=0)
        # pred_img = np.clip(np.uint8((pred+1)*(255/2)), 0, 255)
        pred_img = np.clip(np.uint8(pred*255), 0, 255)
        pred_img = Image.fromarray(pred_img)
        pred_img.save(f"{path}/test_pred_{noise_type}/{index}.png")
        
        nois = noised[0]
        # nois_img = Image.fromarray(np.uint8((nois+1)*(255/2)))
        nois_img = Image.fromarray(np.uint8(nois*255))
        nois_img.save(f"{path}/test_noise_{noise_type}/{index}.png")
        
        orig = origin[0]
        result_1d.append(np.sum(np.abs(pred - orig)))
        result_2d.append(np.sum((pred - orig)**2))
        
        if not index: print("\n\n")
        remining_time = timecounter.predictTime(i+1)
        
        cp.cprint(f"\033[1A{i+1} / {length} \t{remining_time}{' '*30}", "green")
        index += 1
        
    result_1d = np.array(result_1d)
    result_2d = np.array(result_2d)
    
    print("\n1D", np.average(result_1d))
    print("2D", np.average(result_2d))
    
    txt = """\
1D-Max, 1D-Min, 1D-Avg, 2D-Max, 2D-Min, 2D-Avg
%s, %s, %s, %s, %s, %s
"""%(np.max(result_1d), np.min(result_1d), np.average(result_1d),
np.max(result_2d), np.min(result_2d), np.average(result_2d))
    
    with open(f"{path}/testResult_{noise_type}.txt", mode="w") as f:
        f.write(txt)
        
 
def speed_measure(loop:int=50):
    
    esp_size = 576
    
    esp = ESPNet.run((esp_size, esp_size, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True)
    eun = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=LOSS, autoencoder=True)
    
    img_0 = np.ones((1, esp_size, esp_size, 3))
    img_1 = np.ones((1, 576, 576, 3))
    
    speed_0 = np.zeros(loop)
    speed_1 = np.zeros(loop)
    
    for i in range(loop + 5):
        if (i < 5):
            index = -1
        else:
            index = i - 5
            
        start = time.time()
        esp.predict(img_0, verbose=0)
        end = time.time()
        diff_0 = end-start
        
        print(f"{index}. + ESP : {diff_0}")
    
        start = time.time()
        eun.predict(img_1, verbose=0)
        end = time.time()
        diff_1 = end-start
        
        print(f"{index}. - EUN : {diff_1}")
        
        if (index >= 0):
            speed_0[index] += diff_0
            speed_1[index] += diff_1
    
    with open(f"{RESULT_DIR}/esp-{esp_size}-eun-576.txt", mode="w") as f:
         f.write(f"""\
ESPNet:
max min mean median
{np.max(speed_0)} {np.min(speed_0)} {np.mean(speed_0)} {np.median(speed_0)}

E-UNet:
max min mean median
{np.max(speed_1)} {np.min(speed_1)} {np.mean(speed_1)} {np.median(speed_1)}
""")
 
 
def main():
    
    # return
    
    # check_noise(3, "linear")
    # check_noise(3, "tanh")
    # check_noise(3, "mix")
    
    # speed_measure(); return
    
    for reduce_const in [1]:
        for fold in [3]:
            
            train(fold, "mix", reduce_const=reduce_const)
            test(fold, "mix", reduce_const=reduce_const)
        

main()
