import tensorflow as tf
from tensorflow.keras.models import Sequential
import os
import numpy as np
from PIL import Image

## my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from mymodel.SemSegLight import E_UNet
from luminance_extender import LuminanceExtender as LE


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ================


SAVE_DIR = "/workspace/visualization/fill_image"
AE_DIR = "/workspace/fullframe/result/autoencoder"
CNN_DIR = "/workspace/fullframe/result/540x540"

TARGET = [
    ["default", None, "e-unet_4class_adam_dropout_20220805_AutoLearning_fold5_576x576"],
    ["ae_mse", "AE_e-unet_20221014-1_fold5", "e-unet_4class_use-AE-input_20221014-1_flip_20221103_AutoLearning_fold5_576x576"],
    ["ae_ssim_mse", "AE_e-unet_20221014_ssim_mse_fold5", "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_20221103_AutoLearning_fold5_576x576"],
]

COLORS = np.array([
    [255, 0, 0],
    [255, 64, 255],
    [64, 255, 255],
    [0, 255, 0],
], dtype=np.uint8)


## ========================================


def make_fill_image(size=(576, 576, 3)):
    
    nega = np.zeros(size)
    posi = np.ones(size)
    
    le = LE(size[:-1])
    
    circle_nega = posi - le.circle_img
    half_nega = posi - le.half_img
    slant_nega = posi - le.slant_img
    circle_posi = nega + le.circle_img
    half_posi = nega + le.half_img
    slant_posi = nega + le.slant_img
    
    imgs = [[nega, posi], [circle_nega, circle_posi],
            [half_nega, half_posi], [slant_nega, slant_posi]]
    
    Utils.makedir(f"{SAVE_DIR}/images")
    for i, img in enumerate(imgs):
        n, p = img
        n_img = Image.fromarray(np.uint8(n) * 255)
        n_img.save(f"{SAVE_DIR}/images/nega_{i}.png")
        p_img = Image.fromarray(np.uint8(p) * 255)
        p_img.save(f"{SAVE_DIR}/images/posi_{i}.png")
    
    return imgs


def prediction(imgs):
    
    for target in TARGET:
        name, ae, cnn = target
        Utils.makedir(f"{SAVE_DIR}/{name}")
        
        if ae is not None:
            ae_model = E_UNet.run((576, 576, 3), num_classes=3, autoencoder=True)
            ae_model.load_weights(f"{AE_DIR}/{ae}")
            cp.cprint(f"> Autoencoder : {ae}", "cyan")
        
        cnn_model = E_UNet.run((576, 576, 3), num_classes=4)
        cnn_model.load_weights(f"{CNN_DIR}/{cnn}")
        cp.cprint(f"> CNN : {cnn}\n", "cyan")
        
        for i, img in enumerate(imgs):
            n, p = img
            n = np.reshape(n, (1, *n.shape))
            p = np.reshape(p, (1, *p.shape))
            
            if ae is not None:
                n = ae_model.predict([n])
                p = ae_model.predict([p])
            n ,= cnn_model.predict([n])
            p ,= cnn_model.predict([p])

            n = np.argmax(n, axis=2)
            p = np.argmax(p, axis=2)
            
            n_map = np.zeros((*n.shape, 3), dtype=np.uint8)
            p_map = np.zeros((*p.shape, 3), dtype=np.uint8)
            
            for label in range(4):
                n_map += COLORS[label] * np.repeat(np.reshape((n==label), (*n.shape, 1)), 3, axis=2)
                p_map += COLORS[label] * np.repeat(np.reshape((p==label), (*p.shape, 1)), 3, axis=2)
            
            n_img = Image.fromarray(n_map)
            p_img = Image.fromarray(p_map)
            n_img.save(f"{SAVE_DIR}/{name}/nega_{i}.png")
            p_img.save(f"{SAVE_DIR}/{name}/posi_{i}.png")
            
            cp.cprint(f"\033[1A{i+1} / 4   ", "green")


def main():
    
    Utils.makedir(SAVE_DIR)
    imgs = make_fill_image()
    prediction(imgs)


main()
