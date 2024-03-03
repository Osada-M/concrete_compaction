from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
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
import pruning_layer as PL


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
ex = PL.prune_rectify()
if ex: exit()

DATASET_DIR = "/workspace/Dataset/fullframe"
RESULT_DIR = "/workspace/fullframe/result/pruning_model"
AE_DIR = "/workspace/fullframe/result/autoencoder"
CNN_DIR = "/workspace/fullframe/result/540x540"
BATCH_SIZE = 6
EPOCHS = 10
SIZE = [576, 576]
LIMIT = None
SKIP = 1
RANDOM_SEED = 1
TEST_SKIP = 10

# AE_TRAIN = True
AE_TRAIN = False
# LOAD_ID = "AE_e-unet_20221014_ssim_mse"
LOAD_ID = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"
AE_LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_LOSS = MyLosses.ssim_mse_loss


save_path = lambda *fold: f"{RESULT_DIR}/{LOAD_ID}_{fold[0]}_fold{fold[1]}"
pruned_ae_path = lambda *fold: f"{RESULT_DIR}/{AE_LOAD_ID}_45_fold{fold[1]}"

ae_path = lambda fold: f"{AE_DIR}/{AE_LOAD_ID}_fold{fold}"
cnn_path = lambda fold: f"{CNN_DIR}/{LOAD_ID}_fold{fold}_576x576"

text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"


# if not AE_TRAIN:
#     TEST_LENGTH = 2000


def load(fold:int=1):
    
    # E_UNet.is_PL = True
    
    if AE_TRAIN:
        model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=AE_LOSS, autoencoder=True)
        model.load_weights(ae_path(fold))
    else:
        model = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False)
        model.load_weights(cnn_path(fold))
    
    return model


def datacounter(datapath):

    with open(datapath) as f:
        readlines = (f.readlines())[::SKIP]
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


def dataGenerator(resourcepath:str, noise:bool=True, test:bool=False, is_flip:bool=True, is_rotate:bool=True,
                  sparsity:int=40, fold:int=1):
                #   interpreter=None, input_index=None, output_index=None):
    
    # if not AE_TRAIN:
    #     im.set_AE_id(AE_LOAD_ID)
    
    if test:
        is_flip = False
        is_rotate = False
    
    if not AE_TRAIN:
        p_ae_path = pruned_ae_path(sparsity, fold)
        
        ## Load quantied model
        interpreter = tf.lite.Interpreter(model_path=f"{p_ae_path}/quantized_model.tflite")
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
    
    
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
            all_in_one=not AE_TRAIN
            )
        
        if not AE_TRAIN:
            
            pred = [0]*(BATCH_SIZE**(not test))
            for index, noi in enumerate(noised):
                interpreter.set_tensor(input_index, [noi])
                # interpreter.allocate_tensors()
                interpreter.invoke()
                output = interpreter.tensor(output_index)
                pred[index] = np.copy(output()[0])
            pred = np.array(pred)

            yield pred, origin
        
        yield noised, origin


def prune():
    """
    Plune weights and save quantized model
    """
    
    for sparsity in [55]:
    # for sparsity in [30, 25, 20, 15, 10]:
        cp.cprint(f"sparsity : {sparsity}", "pink")
        
        for fold in range(1, 2):
        # for fold in range(1, 6):
            
            # if (fold >= 5):
            # if (sparsity != 20):
            if True:
                path = save_path(sparsity, fold)
                
                ## 
                cp.cprint(f"path : {path}", "orange")
                Utils.makedir(path)
                Utils.makedir(f"{path}/cp")
                        
                ## Create changing permission script
                create_permission_sh(path)
                
                train_length = datacounter(text_path(fold, "train"))
                end_step = np.ceil(train_length / BATCH_SIZE).astype(np.int32) * EPOCHS

                model = load(fold)

                prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
                pruning_params = {'pruning_schedule':
                    # tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.10, final_sparsity=sparsity/100, begin_step=0, end_step=end_step)# \
                    # if not AE_TRAIN else \
                    tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsity/100, begin_step=0, frequency=100)
                    }
                model_for_pruning = prune_low_magnitude(model, **pruning_params)

                model_for_pruning.compile(optimizer='adam',
                                        loss=AE_LOSS if AE_TRAIN else tf.keras.losses.CategoricalCrossentropy(),
                                        metrics=['acc'])

                callbacks = [
                    tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir=f"{path}/cp"),
                    ]
                
                cp.cprint("Pruning", "green")
                history = model_for_pruning.fit_generator(
                    generator=dataGenerator(text_path(fold, "train"), noise=True),#, interpreter=interpreter, input_index=input_index, output_index=output_index),
                    steps_per_epoch=int(np.ceil(train_length/BATCH_SIZE)),
                    epochs=EPOCHS,
                    validation_data=dataGenerator(text_path(fold, "validation"), noise=True),#, interpreter=interpreter, input_index=input_index, output_index=output_index),
                    validation_steps=100,
                    shuffle=False,
                    callbacks=callbacks
                    )
                
                with open(f"{path}/log.txt", mode="w") as f:
                    f.write(str(history.history))
                
                ## Freeze pruned weights
                model_for_export = tfmot.sparsity.keras.strip_pruning(model)
                save_model(model_for_export, f"{path}/pruned_model.h5", include_optimizer=False)
                cp.cprint(f"Saved pruned weights.", "green")
                
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
            else:
                test_CNN(sparsity, fold)


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
    
    length = datacounter(text_path(fold, "test")) // TEST_SKIP
    result_1d = []
    result_2d = []
    
    Utils.makedir(f"{path}/test_noise")
    Utils.makedir(f"{path}/test_pred")
    
    timecounter = TimeCounter(length)
    index = 0
    
    for i, (noised, origin) in enumerate(dataGenerator(text_path(fold, "test"), test=True)):
        if (i >= length): break
        
        interpreter.set_tensor(input_index, noised)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        pred = np.copy(output()[0])
        
        pred_img = np.clip(np.uint8(pred*255), 0, 255)
        pred_img = Image.fromarray(pred_img)
        pred_img.save(f"{path}/test_pred/{index}.png")
        
        nois = noised[0]
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
        
        
def test_CNN(sparsity, fold):
    """
    Test for Compaction Judgment Model
    """
    
    path = save_path(sparsity, fold)
    ## Create changing permission script
    create_permission_sh(path)
    
    ## Load quantied model
    interpreter = tf.lite.Interpreter(model_path=f"{path}/quantized_model.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    p_ae_path = pruned_ae_path(sparsity, fold)

    ## Load quantied model
    ae_interpreter = tf.lite.Interpreter(model_path=f"{p_ae_path}/quantized_model.tflite")
    ae_interpreter.allocate_tensors()
    ae_input_index = ae_interpreter.get_input_details()[0]["index"]
    ae_output_index = ae_interpreter.get_output_details()[0]["index"]
    
    cp.cprint(f"@ load path : {path}", "orange")
    cp.cprint(f"@ load AE path : {p_ae_path}", "orange")
    
    length = datacounter(text_path(fold, "test")) // TEST_SKIP
    
    timecounter = TimeCounter(length)
    print_flag = 0
    pixels = SIZE[0] * SIZE[1]
    correct = 0
    
    true_positive = {"before":0, "just":0}
    true_negative = {"before":0, "just":0}
    false_positive = {"before":0, "just":0}
    false_negative = {"before":0, "just":0}
    iou = {"before":0, "just":0}
    iou_accum = {"before":0, "just":0}
    
    for i, (noised, mask) in enumerate(dataGenerator(text_path(fold, "test"), test=True)):
        if (i >= length): break
        
        # pred ,= model.predict([noised])
        ae_interpreter.set_tensor(ae_input_index, noised)
        ae_interpreter.invoke()
        ae_output = ae_interpreter.tensor(ae_output_index)
        ae_pred = np.copy(ae_output())
        
        interpreter.set_tensor(input_index, ae_pred)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        predict = np.copy(output()[0])

        ## 4 classes to 2 classes
        ans_label = np.argmax(mask[0], axis=2)
        ans_label = ans_label >= 2
        four_pred_buf = np.zeros(SIZE)
        four_pred_buf += (predict[:, :, 0] + predict[:, :, 1]) <= 0.5
        pred_label = np.uint8(four_pred_buf)
        del predict
        
        ans_label = np.uint8(ans_label)
        
        ## 不正解数を集計
        incorrect = np.sum(ans_label ^ pred_label)
        ## 正解数を逆算
        correct += pixels - np.sum(incorrect)
        
        #### 各種の値を行列演算(a : answer, p : prediction)
        ## a & p
        a_and_p = np.sum(ans_label & pred_label)
        ## ~a & ~p
        not_a_and_not_p = np.sum(Calc.inverse_matrix(ans_label) & Calc.inverse_matrix(pred_label))
        ## ~(a & p)
        not_1_a_and_p_1 = np.sum(Calc.inverse_matrix(ans_label & pred_label))
        ## ~p & a
        not_p_or_a = np.sum(Calc.inverse_matrix(pred_label) & ans_label)
        ## p & ~a
        p_or_not_a = np.sum(pred_label & Calc.inverse_matrix(ans_label))
        ## a | p
        a_or_p = np.sum(pred_label | ans_label)
        ## ~a | ~p
        not_a_or_not_p = np.sum(Calc.inverse_matrix(pred_label) | Calc.inverse_matrix(ans_label))
        
        # True Positive
        true_positive["before"] += np.nan_to_num(not_1_a_and_p_1)
        true_positive["just"] += np.nan_to_num(a_and_p)
        ## True Negative
        true_negative["before"] += np.nan_to_num(a_and_p)
        true_negative["just"] += np.nan_to_num(not_1_a_and_p_1)
        ## False Positive
        false_positive["before"] += np.nan_to_num(not_p_or_a)
        false_positive["just"] += np.nan_to_num(p_or_not_a)
        ## False Negative
        false_negative["before"] += np.nan_to_num(p_or_not_a)
        false_negative["just"] += np.nan_to_num(not_p_or_a)
        
        ## IoUの計算
        if a_or_p:
            iou["just"] += np.nan_to_num(a_and_p / a_or_p)
            iou_accum["just"] += 1
        if not_a_or_not_p:
            iou["before"] += np.nan_to_num(not_a_and_not_p / not_a_or_not_p)
            iou_accum["before"] += 1
        
        if not print_flag:
            print_flag = 1
            print("\n\n")
        
        remining_time = timecounter.predictTime(i+1)
        cp.cprint(f"\033[1A{i+1} / {length} \t{remining_time}\t\tAcc : {round(correct/(pixels*(i+1))*100, 3)} [%]{' '*30}", "green")
        
    print()
        
    denominator = pixels*length
    
    ## Accuracyの算出
    accuracy = correct/denominator
    
    ## 最終的なIoUの算出
    for key in iou.keys():
        iou[key] /= iou_accum[key]
    
    ## F1値の算出
    f1 = []
    for key in iou.keys():
        f1.append(Calc.f1_score(
            Calc.precision(
                tp=true_positive[key],
                fp=false_positive[key]
            ),
            Calc.recall(
                tp=true_positive[key],
                fn=false_negative[key]
            )
        ))
    
    ## テスト結果を辞書に格納
    survey_keys = ["accuracy", "F1(Before)", "F1(Just)", "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)", "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)", "IoU(Before)", "IoU(Just)"]
    survey_values = [accuracy, *f1,
                    true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
                    true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
                    *iou.values()]
    
    with open(f"{path}/testResult.txt", mode="w") as f:
        f.write(" ".join(survey_keys) + "\n")
        f.write(" ".join(map(str, survey_values)))


def create_permission_sh(sh_dir):

    text = """\    
dir=`pwd`
echo 'kit67304' | sudo chmod 777 $dir/*
echo 'kit67304' | sudo chmod 777 $dir/*/*
echo 'kit67304' | sudo chmod 777 $dir/*/*/*
"""

    with open(f"{sh_dir}/permission.sh", mode="w") as f:
        f.write(text)
        
        
def main():
    
    prune()


main()
