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
EPOCHS = 10
SIZE = [576, 576]
LIMIT = None
SKIP = 1
RANDOM_SEED = 1
TEST_SKIP = 5

# AE_TRAIN = True
# LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_TRAIN = False
LOAD_ID = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"
AE_LOAD_ID = "AE_e-unet_20221014_ssim_mse"
AE_LOSS = MyLosses.ssim_mse_loss


# save_path = lambda *fold: f"{RESULT_DIR}/filter-reduce-trail/{fold[0]}_fold{fold[1]}"
save_path = lambda *fold: f"{RESULT_DIR}/filter-reduce/{fold[0]}_fold{fold[1]}_zero"
pruned_ae_path = lambda *fold: f"{RESULT_DIR}/{AE_LOAD_ID}_45_fold{fold[1]}"

ae_path = lambda fold: f"{AE_DIR}/{AE_LOAD_ID}_fold{fold}"
cnn_path = lambda fold: f"{CNN_DIR}/{LOAD_ID}_fold{fold}_576x576"

text_path = lambda *target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"


def create_model(fold:int=1, reduce_const:float=1., load:bool=False, rectified:bool=True):
    
    if AE_TRAIN:
        model = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=AE_LOSS, autoencoder=True, reduce_const=reduce_const)
        if load:
            if (reduce_const == 1.):
                model.load_weights(ae_path(fold))
            else:
                model.load_weights(f"{save_path(f'AE_{int(100-reduce_const*100):02d}', fold)}/{['', 'rectified_', 'rectified_1_'][rectified]}pruned_model.h5")
                
    else:
        model = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False, reduce_const=reduce_const)
        if load:
            if (reduce_const == 1.):
                model.load_weights(cnn_path(fold))
            else:
                model.load_weights(f"{save_path(f'{int(100-reduce_const*100):02d}', fold)}/{['', 'rectified_', 'rectified_1_'][rectified]}pruned_model.h5")
    
    return model


def datacounter(datapath):

    with open(datapath) as f:
        readlines = (f.readlines())[::SKIP]
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines) // TEST_SKIP


def dataGenerator(resourcepath:str, noise:bool=True, is_flip:bool=False, is_rotate:bool=False,):

    image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        readlines = readlines[::TEST_SKIP]
        
        if AE_TRAIN:
            random.seed(RANDOM_SEED)
            random.shuffle(readlines)
        
        # if LIMIT: readlines = readlines[:LIMIT]
        # data_length = len(readlines)        
        # readlines = readlines[:data_length]
        
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
                                                        batch_size=1,
                                                        shuffle=False)
    mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
                                                      x_col="mask",
                                                      target_size=SIZE.copy(),
                                                      color_mode="rgb",
                                                      class_mode=None,
                                                      batch_size=1,
                                                      shuffle=False)
    
    for i, (image, mask) in enumerate(zip(image_generator, mask_generator)):
        noised, origin = im.adjust_data(
            image, mask, is_fullframe=True, size=SIZE.copy(), classification="fourclasses", num_classes=4,
            autoencoder=AE_TRAIN, noise=noise, noise_type="mix" if AE_TRAIN else "includeAE-noise",
            is_flip=is_flip, flip_list=[0, 0, 1, 1, 2, 3], is_rotate=is_rotate, rotate_degrees=[[0, 359],],
            use_AE_input=not AE_TRAIN
            )
        
        yield noised, origin


def test_AE(folds, sparsities):
    """
    Test for Compaction Judgment Model
    """
    
    # folds = [3]
    # sparsities = [96, 92]
    
    for fold in folds:
        
        cp.cprint(f"fold : {fold}", "pink")
        
        for sparsity in sparsities:
        
            cp.cprint(f"sparsity : {sparsity}", "pink")
            
            path = save_path(f"AE_{int(100-sparsity):02d}", fold)
            cp.cprint(f"save_path : {path}", "pink")
            
            reduce_const = sparsity / 100
            model = create_model(fold, reduce_const, True, False)
            
            cp.cprint(f"@ load path : {path}", "orange")
            
            # length = datacounter(text_path(fold, "test"))
            length = 200
            
            result_1d = []
            result_2d = []
            
            Utils.makedir(f"{path}/test_noise")
            Utils.makedir(f"{path}/test_pred")
            
            timecounter = TimeCounter(length)
            index = 0
            
            for i, (noised, origin) in enumerate(dataGenerator(text_path(fold, "test"))):
                if (i >= length): break
                
                pred ,= model.predict([noised], verbose=0)
                
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
        

def test_CNN(folds, sparsities, rectified=True):
    """
    Test for Compaction Judgment Model
    """
    
    # folds = [3]
    # sparsities = [96, 92]
    
    for fold in folds:
        
        cp.cprint(f"fold : {fold}", "pink")
        
        for sparsity in sparsities:
        
            cp.cprint(f"sparsity : {sparsity}", "pink")
            
            path = save_path(f"{int(100-sparsity):02d}", fold)
            cp.cprint(f"save_path : {path}", "pink")
            
            cp.cprint(f"rectified : {rectified}", "pink")
            
            reduce_const = sparsity / 100
            model = create_model(fold, reduce_const, True, rectified)
            length = datacounter(text_path(fold, "test"))
            
            timecounter = TimeCounter(length)
            pixels = SIZE[0] * SIZE[1]
            correct = 0
            
            true_positive = {"before":0, "just":0}
            true_negative = {"before":0, "just":0}
            false_positive = {"before":0, "just":0}
            false_negative = {"before":0, "just":0}
            iou = {"before":0, "just":0}
            iou_accum = {"before":0, "just":0}
            
            for i, (noised, mask) in enumerate(dataGenerator(text_path(fold, "test"))):
                if (i >= length): break
                
                predict ,= model.predict([noised], verbose=0)

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
                
                if not i:
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
            
            with open(f"{path}/testResult{['', '_rectified'][rectified]}.txt", mode="w") as f:
                f.write(" ".join(survey_keys) + "\n")
                f.write(" ".join(map(str, survey_values)))
        
            del model


def test_CNN_4c(folds, sparsities, rectified=True):
    """
    Test for Compaction Judgment Model
    """
    
    # folds = [3]
    # sparsities = [96, 92]
    
    for fold in folds:
        
        cp.cprint(f"fold : {fold}", "pink")
        
        for sparsity in sparsities:
        
            cp.cprint(f"sparsity : {sparsity}", "pink")
            
            path = save_path(f"{int(100-sparsity):02d}", fold)
            cp.cprint(f"save_path : {path}", "pink")
            
            cp.cprint(f"rectified : {rectified}", "pink")
            
            reduce_const = sparsity / 100
            model = create_model(fold, reduce_const, True, rectified)
            length = datacounter(text_path(fold, "test")) // TEST_SKIP
            
            timecounter = TimeCounter(length)
            pixels = SIZE[0] * SIZE[1]
            correct = 0
            
            true_positive = {"before":0, "b-before":0, "b-just":0, "just":0}
            true_negative = {"before":0, "b-before":0, "b-just":0, "just":0}
            false_positive = {"before":0, "b-before":0, "b-just":0, "just":0}
            false_negative = {"before":0, "b-before":0, "b-just":0, "just":0}
            iou = {"before" : 0, "b-before" : 0, "b-just" : 0, "just" : 0}
            iou_accum = {"before" : 0, "b-before" : 0, "b-just" : 0, "just" : 0}
            
            for i, (noised, mask) in enumerate(dataGenerator(text_path(fold, "test"))):
                if (i >= length): break
                
                predict ,= model.predict([noised], verbose=0)

                ans_label = np.copy(mask[0])
                ans_label = np.uint8(ans_label)

                pred_label = np.identity(4)[np.argmax(predict, axis=(2))]
                pred_label = np.uint8(pred_label)
                del predict
                
                
                ## 不正解数を集計
                incorrect = np.sum(np.sum(ans_label ^ pred_label, axis=(2)) >= 1)
                ## 正解数を逆算
                correct += pixels - np.sum(incorrect)
                
                #### 各種の値を行列演算(a : answer, p : prediction)
                # for class_number in range(4):
                ## a & p
                a_and_p = np.sum(ans_label & pred_label, axis=(0, 1))
                ## ~a & ~p
                not_a_and_not_p = np.sum(Calc.inverse_matrix(ans_label) & Calc.inverse_matrix(pred_label), axis=(0, 1))
                ## ~(a & p)
                not_1_a_and_p_1 = np.sum(Calc.inverse_matrix(ans_label & pred_label), axis=(0, 1))
                ## ~p & a
                not_p_or_a = np.sum(Calc.inverse_matrix(pred_label) & ans_label, axis=(0, 1))
                ## p & ~a
                p_or_not_a = np.sum(pred_label & Calc.inverse_matrix(ans_label), axis=(0, 1))
                ## a | p
                a_or_p = np.sum(pred_label | ans_label, axis=(0, 1))
                ## ~a | ~p
                not_a_or_not_p = np.sum(Calc.inverse_matrix(pred_label) | Calc.inverse_matrix(ans_label), axis=(0, 1))
                
                for class_number, key in enumerate(true_positive.keys()):
                    # True Positive
                    true_positive[key] += a_and_p[class_number]
                    ## True Negative
                    true_negative[key] += not_1_a_and_p_1[class_number]
                    ## False Positive
                    false_positive[key] += p_or_not_a[class_number]
                    ## False Negative
                    false_negative[key] += not_p_or_a[class_number]
                
                    ## IoUの計算
                    if a_or_p[class_number]:
                        iou[key] += np.nan_to_num(a_and_p[class_number] / a_or_p[class_number])
                        iou_accum[key] += 1

                if not i: print("\n\n")
                
                ## 終了時刻の予測
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
            survey_keys = ["accuracy", "F1(Before)", "F1(B-Before)", "F1(B-Just)", "F1(Just)",
                        "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)",
                        "TP(B-Before)", "TN(B-Before)", "FP(B-Before)", "FN(B-Before)",
                        "TP(B-Just)", "TN(B-Just)", "FP(B-Just)", "FN(B-Just)",
                        "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)",
                        "IoU(Before)", "IoU(B-Before)", "IoU(B-Just)", "IoU(Just)"]
            survey_values = [accuracy, *f1,
                            true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
                            true_positive["b-before"], true_negative["b-before"], false_positive["b-before"], false_negative["b-before"],
                            true_positive["b-just"], true_negative["b-just"], false_positive["b-just"], false_negative["b-just"],
                            true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"],
                            *iou.values()]
            
            with open(f"{path}/testResult_4classes{['', '_rectified', 'rectified_1_'][rectified]}.txt", mode="w") as f:
                f.write(" ".join(survey_keys) + "\n")
                f.write(" ".join(map(str, survey_values)))
            
            del model


def main():
    
    folds = [3]
    # sparsities = [88, 84, 80, 76, 72, 68]
    # sparsities = [90, 80, 70, 60, 40, 30, 20]
    sparsities = [75, 50, 25]
    
    # test_CNN(folds, sparsities, 1)
    
    # test_CNN(folds, sparsities, 0)
    # test_CNN_4c(folds, sparsities, 0)
    
    if AE_TRAIN:
        test_AE(folds, sparsities)
    else:
        test_CNN(folds, sparsities, 0)
    # test_CNN_4c(folds, sparsities, 1)


main()
