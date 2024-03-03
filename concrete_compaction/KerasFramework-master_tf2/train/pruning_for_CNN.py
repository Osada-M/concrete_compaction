from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from PIL import Image

# my module
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from line_sender import send_master
from mymodel.SemSegLight import E_UNet
from colorPrint import Cprint as cp
from my_loss_function import MyLosses
from MyPruning import MyPruning
# import pruning_layer as PL


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
# ex = PL.prune_rectify()
# if ex: exit()

DATASET_DIR = "/workspace/Dataset/fullframe"
RESULT_DIR = "/workspace/fullframe/result/pruning_model"
AE_DIR = "/workspace/fullframe/result/autoencoder"
CNN_DIR = "/workspace/fullframe/result/540x540"
BATCH_SIZE = 6
EPOCHS = 25 #100
SIZE = [576, 576]
LIMIT = None
SKIP = 1
RANDOM_SEED = 1
TEST_SKIP = 10

# AE_TRAIN = True
# LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_TRAIN = False
LOAD_ID = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"
AE_LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_LOSS = MyLosses.ssim_mse_loss


# save_path = lambda *fold: f"{RESULT_DIR}/filter-reduce-trail/{fold[0]}_fold{fold[1]}"
save_path = lambda *fold: f"{RESULT_DIR}/filter-reduce/{fold[0]}_fold{fold[1]}_square-error"
# save_path = lambda *fold: f"{RESULT_DIR}/filter-reduce/AE_{fold[0]}_fold{fold[1]}_2"

ae_path = lambda fold: f"{AE_DIR}/{AE_LOAD_ID}_fold{fold}"
cnn_path = lambda fold: f"{CNN_DIR}/{LOAD_ID}_fold{fold}_576x576"

text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"


def create_model(fold:int=1, reduce_const:float=1., load:bool=False):
    
    if AE_TRAIN:
        model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=AE_LOSS, autoencoder=True, reduce_const=reduce_const)
        if load:
            if (reduce_const == 1.):
                model.load_weights(ae_path(fold))
            else:
                model.load_weights(f"{save_path(f'{int(100-reduce_const*100):02d}', fold)}/pruned_model.h5")
                
    else:
        model = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False, reduce_const=reduce_const)
        if load:
            if (reduce_const == 1.):
                model.load_weights(cnn_path(fold))
            else:
                model.load_weights(f"{save_path(f'{int(100-reduce_const*100):02d}', fold)}/pruned_model.h5")
    
    return model


def datacounter(datapath):

    with open(datapath) as f:
        readlines = (f.readlines())[::SKIP]
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


def dataGenerator(resourcepath:str, noise:bool=True, test:bool=False, is_flip:bool=True, is_rotate:bool=True,):
    
    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        random.seed(RANDOM_SEED)
        random.shuffle(readlines)
        
        if LIMIT: readlines = readlines[:LIMIT]
        
        data_length = len(readlines)
        
        if test:
            readlines = readlines[:data_length//TEST_SKIP]
        else:
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
                                                        class_mode=None,
                                                        batch_size=BATCH_SIZE**(not test),
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="rgb",
                                                      class_mode=None,
                                                      batch_size=BATCH_SIZE**(not test),
                                                      shuffle=False)
    
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        noised, origin = im.adjust_data(
            image, mask, is_fullframe=True, size=SIZE.copy(), classification="fourclasses", num_classes=4,
            autoencoder=AE_TRAIN, noise=noise, noise_type="mix" if AE_TRAIN else "includeAE-noise",
            is_flip=is_flip, flip_list=[0, 0, 1, 1, 2, 3], is_rotate=is_rotate, rotate_degrees=[[0, 359],],
            use_AE_input=not AE_TRAIN
            )
        
        yield noised, origin


def prune():
    """
    Plune weights and save quantized model
    """
    
    ## Prune rates
    # length = list(reversed(list(range(20, 68))[::4]))
    # length = list(reversed(list(range(20, 100))[::4]))
    length = [25]
    
    rc_old = 0.5
    # rc_old = 1.
    
    MyPruning.is_print_details = False
    
    for fold in range(3, 4):
        
        im.fold = fold
        im.AE_model = None
        
        for sparsity in length:
            cp.cprint(f"sparsity : {sparsity}", "pink")
            reduce_const = sparsity / 100
            
            path = save_path(f"{100-sparsity :02d}", fold)
            
            cp.cprint(f"path : {path}", "orange")
            Utils.makedir(path)
            
            train_length = datacounter(text_path(fold, "train"))

            model_be = create_model(fold, rc_old, True)
            model_af = create_model(fold, reduce_const, False)
            
            ## Reduce weights (filter-wise)
            # model_af = MyPruning.prune_tuning(model_before=model_be, model_after=model_af, reduce_const=reduce_const, reduce="ssim")
            model_af = MyPruning.prune_tuning(model_before=model_be, model_after=model_af, reduce="square-error")
            
            model_af.compile(optimizer='adam',
                        loss=AE_LOSS if AE_TRAIN else tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['acc'])

            cp.cprint("Train", "green")
            
            if AE_TRAIN:
                cp.cprint("[!] AE train", "pink")
                
                history = model_af.fit_generator(
                    generator=dataGenerator(text_path(fold, "train"), noise=True),#, interpreter=interpreter, input_index=input_index, output_index=output_index),
                    steps_per_epoch=int(np.ceil(train_length/BATCH_SIZE)),
                    epochs=EPOCHS,
                    validation_data=dataGenerator(text_path(fold, "validation"), noise=True),#, interpreter=interpreter, input_index=input_index, output_index=output_index),
                    validation_steps=100,
                    shuffle=False,
                    )
                
                with open(f"{path}/log.txt", mode="w") as f:
                    f.write(str(history.history))
                
                ## Freeze pruned weights
                model_af.save(f"{path}/pruned_model.h5")
                cp.cprint(f"Saved pruned weights.", "green")
            
            else:
                accuracy_max = 0
                
                with open(f"{path}/log.txt", mode="w"): pass
                
                for itr in range(EPOCHS):
                    cp.cprint(f"auto learing | iteration : {itr+1} / {EPOCHS} | accurasy : {accuracy_max}", "green")
                    
                    history = model_af.fit_generator(
                        generator=dataGenerator(text_path(fold, "train"), noise=True),
                        steps_per_epoch=int(np.ceil(train_length/BATCH_SIZE)),
                        epochs=1,
                        validation_data=dataGenerator(text_path(fold, "validation"), noise=True),
                        validation_steps=100,
                        shuffle=False,
                        )
                    
                    with open(f"{path}/log.txt", mode="a") as f:
                        f.write(str(history.history))
                    
                    try:
                        acc = history.history[f"val_classifier_acc"][-1]*100
                    except:
                        acc = history.history["val_acc"][-1]*100
                    ## 精度が更新される場合、モデルの保存を行う
                    # if (itr > 2) and (acc >= accuracy_max):
                    if (acc >= accuracy_max):
                        cp.cprint("@ Update accuracy.", "orange")
                        accuracy_max = acc
                        model_af.save(f"{path}/pruned_model.h5")
                        cp.cprint(f"@ Saved pruned weights.", "green")
            
            rc_old = reduce_const
            del model_be, model_af


def main():
    
    prune()


main()
