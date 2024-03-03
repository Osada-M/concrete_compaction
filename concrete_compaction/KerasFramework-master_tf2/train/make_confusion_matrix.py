from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# my module
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from line_sender import send_master
from mymodel.SemSegLight import E_UNet
from colorPrint import Cprint as cp


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DATASET_DIR = "/workspace/Dataset/fullframe"
DIR = "/workspace/fullframe/result/540x540"
SAVE_DIR = "/workspace/visualization/confusion_matrix"
MODEL = lambda fold: f"e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning_fold{fold}_576x576"

SIZE = [576, 576]
LIMIT = None
TEST_SKIP = 3
BATCH_SIZE = 20


text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}_4class.txt"


def create_model(fold):
    
    model = tf.keras.models.load_model(f"{DIR}/{MODEL(fold)}")
    
    return model


def datacounter(datapath):

    with open(datapath) as f:
        readlines = (f.readlines())
    return len(readlines) // (TEST_SKIP * BATCH_SIZE)


def dataGenerator(resourcepath:str):
    
    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        
        if LIMIT: readlines = readlines[:LIMIT]
        
        data_length = len(readlines)
        
        readlines = readlines[:data_length//TEST_SKIP]
        
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
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="rgb",
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False)
    
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        noised, origin = im.adjust_data(
            image, mask, is_fullframe=True, size=SIZE.copy(), classification="fourclasses", num_classes=4,
            autoencoder=False, noise=False, noise_type="includeAE-noise",
            is_flip=False, is_rotate=False, use_AE_input=True
            )
        
        yield noised, origin


def make_matrix(fold):
    
    model = create_model(fold)
    length = datacounter(text_path(fold, "test"))
    matrix = [[0]*4 for _ in range(4)]
    
    text = f"{SAVE_DIR}/{MODEL(fold)}/confusion_matrix.txt"
    with open(text, mode="w") as f:
        f.write("row : prediction, col : answer, index : (Before, B-Before, B-Just, Just)\n")
    
    for i, (noised, masks) in enumerate(dataGenerator(text_path(fold, "test"))):
        if (i >= length): break
        if not i: print("\n\n\n\n")
        
        preds = model.predict([noised], batch_size=BATCH_SIZE, verbose=0)
        print("\033[1A", end="")

        for pred, mask in zip(preds, masks):
            pred_label = np.uint32(np.identity(4)[np.argmax(pred, axis=(2))])
            ans_label = np.uint32(np.copy(mask))
            
            print("\033[4A", end="")
            for pred_index in range(4):
                for mask_index in range(4):
                    # if (mask_index == pred_index): continue
                    matrix[pred_index][mask_index] += np.sum(pred_label[:, :, pred_index] * ans_label[:, :, mask_index])
                cp.cprint("".join(map(lambda x: f"{int(x)}{' '*(20 - len(str(int(x))))}", matrix[pred_index])))
                
            with open(text, mode="a") as f:
                for pred_row in matrix:
                    row = " ".join(map(lambda x: str(int(x)), pred_row))
                    print(row, file=f)
                print("=", file=f)
                        
            matrix = [[0]*4 for _ in range(4)]
                
        cp.cprint(f"{i+1} / {length}", "orange")

    return matrix
        

def run_matrix_work():
    
    for fold in range(1, 6):
        
        Utils.makedir(f"{SAVE_DIR}/{MODEL(fold)}")
        
        matrix = make_matrix(fold)
        text = f"{SAVE_DIR}/{MODEL(fold)}/confusion_matrix.txt"
        
        # with open(text, mode="w") as f:
        #     f.write("row : prediction, col : answer, index : (Before, B-Before, B-Just, Just)\n")
        #     for pred_row in matrix:
        #         row = " ".join(map(str, pred_row))
        #         f.write(row + "\n")

    
def main():
    
    run_matrix_work()


main()
