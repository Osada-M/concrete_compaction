import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import time

# my module
from colorPrint import Cprint as cp
from MyUtils import Utils
from MyUtils import ImageManager as im
from colorPrint import Cprint as cp


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ================

DATASET_DIR = "/workspace/Dataset/fullframe"
RESULT_DIR = "/workspace/visualization/AE_correct"
RANDOM_SEED = int(time.time())
SIZE = [576, 576]
LENGTH = 20

TARGET = [[
    # AE_id, flip, rotate
    "20221014_ssim_mse", True, True]
    ]


## ========================================

text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"

## データ数の調整
# if LIMIT is None:
#     with open(path("train")) as f:
#         data_length = len(f.readlines())
#         LIMIT = data_length - (data_length%BATCH_SIZE)
    
    
def dataGenerator(resourcepath:str, is_flip:bool=False, is_rotate:bool=False):
    
    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        random.seed(RANDOM_SEED)
        random.shuffle(readlines)
        
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
                                                        # classes=["red", "green", "blue"],
                                                        class_mode=None,
                                                        batch_size=1,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="rgb",
                                                    #   classes=["red", "green", "blue"],
                                                      class_mode=None,
                                                      batch_size=1,
                                                      shuffle=False)
    
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        corrected, _ = im.adjust_data(
            image, mask, is_fullframe=True, size=SIZE.copy(), classification="fourclasses", num_classes=4,
            use_AE_input=True, noise=True, noise_type="includeAE-noise", is_flip=is_flip, is_rotate=is_rotate
            )
        
        yield corrected, None#, origin
    
    
def correct():
    
    for target in TARGET:
        
        ae_id, is_flip, is_rotate = target

        for fold in range(1, 6):
            
            cp.cprint(f"{ae_id} : fold {fold}" ,"pink")   
            save_dir = f"/workspace/visualization/AE_correct/{ae_id}_fold{fold}"
            Utils.makedir(save_dir)
            # Utils.makedir(f"{save_dir}/noised")
            Utils.makedir(f"{save_dir}/corrected")
            
            im.FOLD = fold
            im.set_AE_id(ae_id)
            
            for i, (corrected, image) in enumerate(dataGenerator(text_path(fold, "train"), is_flip=is_flip, is_rotate=is_rotate)):
                if (i >= LENGTH): break
                
                corrected = np.uint8(corrected*255.)[0]
                corrected = Image.fromarray(corrected)
                corrected.save(f"{save_dir}/corrected/{i}.png")
                
                # image = np.uint8(image*255.)[0]
                # image = Image.fromarray(image)
                # image.save(f"{save_dir}/corrected/{i}.png")

                if not i:
                    print()
                    
                cp.cprint(f"\033[1A{i+1} / {LENGTH}", "orange")
    
 
def main():
    
    correct()
        

main()
