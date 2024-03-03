from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, save_model
import tensorflow_model_optimization as tfmot
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
import pluning_layer as PL


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
ex = PL.prune_rectify()
if ex: exit()

DATASET_DIR = "/workspace/Dataset/fullframe"
RESULT_DIR = "/workspace/fullframe/result/pluning_model"
AE_DIR = "/workspace/fullframe/result/autoencoder"
CNN_DIR = "/workspace/fullframe/result/540x540"
BATCH_SIZE = 5
EPOCHS = 20
SIZE = [576, 576]
LIMIT = None
SKIP = 1
RANDOM_SEED = 1
TEST_LENGTH = 200

AE_TRAIN = True
LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_LOSS = MyLosses.ssim_mse_loss


save_path = lambda *fold: f"{RESULT_DIR}/{LOAD_ID}_{fold[0]}_fold{fold[1]}"
ae_path = lambda fold: f"{AE_DIR}/{AE_LOAD_ID}_fold{fold}"
cnn_path = lambda fold: f"{CNN_DIR}/{AE_LOAD_ID}_fold{fold}"
text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"


def load(fold:int=1):
    
    # E_UNet.is_PL = True
    
    if AE_TRAIN:
        model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=AE_LOSS, autoencoder=True)
        model.load_weights(ae_path(fold))
    
    return model


def datacounter(datapath):

    with open(datapath) as f:
        readlines = (f.readlines())[::SKIP]
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


def dataGenerator(resourcepath:str, noise:bool=True, test:bool=False, is_flip:bool=False, is_rotate:bool=False):
    
    if not AE_TRAIN:
        im.set_AE_id(AE_LOAD_ID)
    
    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        # if not test:
        random.seed(RANDOM_SEED)
        random.shuffle(readlines)
        
        if test:
            readlines = readlines[:TEST_LENGTH]
        else:
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
            is_flip=is_flip, flip_list=[0, 0, 1, 1, 2, 3], is_rotate=is_rotate, rotate_degrees=[[0, 359]]
            )
        
        yield noised, origin


def plune():
    """
    Plune weights and save quantized model
    """
    
    for sparsity in [45]:
        cp.cprint(f"sparsity : {sparsity}", "pink")
        
        for fold in range(1, 6):
            
            if (fold != 1):
                path = save_path(sparsity, fold)
                cp.cprint(f"path : {path}", "orange")
                Utils.makedir(path)
                Utils.makedir(f"{path}/cp")
                        
                train_length = datacounter(text_path(fold, "train"))
                end_step = np.ceil(train_length / BATCH_SIZE).astype(np.int32) * EPOCHS

                model = load(fold)

                prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
                pruning_params = {'pruning_schedule':
                    tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.10, final_sparsity=sparsity/100, begin_step=0, end_step=end_step)
                    # tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsity/100, begin_step=0, frequency=100)
                    }
                model_for_pruning = prune_low_magnitude(model, **pruning_params)

                model_for_pruning.compile(optimizer='adam',
                                        loss=AE_LOSS,
                                        metrics=['acc'])

                callbacks = [
                    tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir=f"{path}/cp"),
                    ]
                
                cp.cprint("Pluning", "green")
                history = model_for_pruning.fit_generator(
                    generator=dataGenerator(text_path(fold, "train"), noise=True),
                    steps_per_epoch=int(np.ceil(train_length/BATCH_SIZE)),
                    epochs=EPOCHS,
                    validation_data=dataGenerator(text_path(fold, "validation"), noise=True),
                    validation_steps=100,
                    shuffle=False,
                    callbacks=callbacks
                    )
                
                with open(f"{path}/log.txt", mode="w") as f:
                    f.write(str(history.history))
                
                ## Freeze pluned weights
                model_for_export = tfmot.sparsity.keras.strip_pruning(model)
                save_model(model_for_export, f"{path}/pluned_model.h5", include_optimizer=False)
                cp.cprint(f"Saved pluned weights.", "green")
                
                ## Convert to quantized model
                converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
                pruned_tflite_model = converter.convert()
                cp.cprint(f"Converted to quantized model.", "green")

                with open(f"{path}/quantized_model.tflite", 'wb') as f:
                    f.write(pruned_tflite_model)
                cp.cprint(f"Saved quantized model.", "green")
            
            # model_for_pruning.save(path)
            
            if AE_TRAIN:
                test_AE(sparsity, fold)


def test_AE(sparsity, fold):
    """
    Test for illuminance collection AutoEncoder
    """
    
    path = save_path(sparsity, fold)
    # model = load_model(path, custom_objects={"ssim_mse_loss" : AE_LOSS})
    
    ## Load quantied model
    interpreter = tf.lite.Interpreter(model_path=f"{path}/quantized_model.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    cp.cprint(f"@ load path : {path}", "orange")
    
    length = TEST_LENGTH
    result_1d = []
    result_2d = []
    
    Utils.makedir(f"{path}/test_noise")
    Utils.makedir(f"{path}/test_pred")
    
    timecounter = TimeCounter(length)
    index = 0
    
    for i, (noised, origin) in enumerate(dataGenerator(text_path(fold, "test"), test=True)):
        if (i >= length): break
        
        # pred ,= model.predict([noised])
        
        interpreter.set_tensor(input_index, noised)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        pred = np.copy(output()[0])
        
        # pred_img = np.clip(np.uint8((pred+1)*(255/2)), 0, 255)
        pred_img = np.clip(np.uint8(pred*255), 0, 255)
        pred_img = Image.fromarray(pred_img)
        pred_img.save(f"{path}/test_pred/{index}.png")
        
        nois = noised[0]
        # nois_img = Image.fromarray(np.uint8((nois+1)*(255/2)))
        nois_img = Image.fromarray(np.uint8(nois*255))
        nois_img.save(f"{path}/test_noise/{index}.png")
        
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
    
    with open(f"{path}/testResult.txt", mode="w") as f:
        f.write(txt)
        
        
def main():
    
    plune()
    # test_AE(1)


main()
