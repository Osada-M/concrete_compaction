import tensorflow as tf
from tensorflow.keras.models import Sequential
import os
import numpy as np
from PIL import Image

## my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from mymodel.SemSegLight import E_UNet


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ================


SAVE_DIR = "/workspace/fullframe/result/robust_model"
AE_DIR = "/workspace/fullframe/result/autoencoder"
CNN_DIR = "/workspace/fullframe/result/540x540"

AE_ID = [
    # ["20221014-1", "20x-mse"],
    ["20221014_ssim_mse", "ssim_10x-mse"],
    ]

CNN_ID = [
    # [lambda *ae: f"{CNN_DIR}/e-unet_4class_use-AE-input_{ae[0]}_flip_20221103_AutoLearning_fold{ae[1]}_576x576", "e-unet_4classes_flip"],
    [lambda *ae: f"{CNN_DIR}/e-unet_4class_use-AE-input_{ae[0]}_flip_rotate_20221110_AutoLearning_fold{ae[1]}_576x576", "e-unet_4classes_flip_rotate"],
]


## ========================================


decode_ae_id = lambda *ae: f"{AE_DIR}/AE_e-unet_{ae[0]}_fold{ae[1]}"


def marge_model(ae, cnn, fold, check_only=False):
    """
    照度補正用Autoencoderと締固め判定用CNNの統合
    """
    ae_id, ae_name = ae
    cnn_id, cnn_name = cnn
    model_name = "_".join([cnn_name, ae_name, f"fold{fold}"])
    Utils.makedir(f"{SAVE_DIR}/{model_name}")
    Utils.makedir(f"{SAVE_DIR}/{model_name}/accuracy_check")
    if not check_only:
        with open(f"{SAVE_DIR}/{model_name}/compose.txt", mode="w"): pass
    with open(f"{SAVE_DIR}/{model_name}/accuracy_check/result.txt", mode="w"): pass
    
    cp.cprint(f"fold {fold}, {model_name}", "pink")
    
    ae_model = E_UNet.run((576, 576, 3), num_classes=3, autoencoder=True)
    ae_model.load_weights(decode_ae_id(ae_id, fold))
    cp.cprint(f"> Autoencoder : {ae_name}", "cyan")
    
    cnn_model = E_UNet.run((576, 576, 3), num_classes=4)
    cnn_model.load_weights(cnn_id(ae_id, fold))
    cp.cprint(f"> CNN : {cnn_name}", "cyan")
    
    model = Sequential()
    model.add(ae_model)
    model.add(cnn_model)
    
    with open(f"{SAVE_DIR}/{model_name}/accuracy_check/result.txt", mode="a") as f:
        for itr in range(10):
            Utils.makedir(f"{SAVE_DIR}/{model_name}/accuracy_check/test{itr+1}")
            test_img = np.random.rand(1, 576, 576, 3)
            ae_img = ae_model.predict(test_img)
            cnn_img = cnn_model.predict(ae_img)
            marge_img = model.predict(test_img)
            
            cnn_pred = np.identity(4)[np.argmax(cnn_img, axis=3)]
            marge_pred = np.identity(4)[np.argmax(marge_img, axis=3)]
            diff_img = np.sum(np.maximum(cnn_pred - marge_pred, 0), axis=3)
            diff_accum = np.uint32(np.sum(diff_img))
            print(f"test {itr+1:02d} : [ {['OK', 'NG'][bool(diff_accum)]} ] , erorr : {diff_accum}")
            print(f"test {itr+1:02d} : [ {['OK', 'NG'][bool(diff_accum)]} ] , erorr : {diff_accum}", file=f)
            
            Image.fromarray(np.uint8(test_img[0]*255)).save(f"{SAVE_DIR}/{model_name}/accuracy_check/test{itr+1}/test.png")
            Image.fromarray(np.uint8(diff_img[0]*255)).save(f"{SAVE_DIR}/{model_name}/accuracy_check/test{itr+1}/diff.png")
            for dim in range(4):
                Image.fromarray(np.uint8(cnn_pred[0, :, :, dim]*255)).save(f"{SAVE_DIR}/{model_name}/accuracy_check/test{itr+1}/cnn_{dim}.png")
                Image.fromarray(np.uint8(marge_pred[0, :, :, dim]*255)).save(f"{SAVE_DIR}/{model_name}/accuracy_check/test{itr+1}/marge_{dim}.png")
    
    if check_only: return
    
    model.save(f"{SAVE_DIR}/{model_name}")
    cp.cprint("Saved SavedModel format (.pb)", "green")
    model.save(f"{SAVE_DIR}/{model_name}/{model_name}.h5")
    cp.cprint("Saved HDF5 format (.h5)", "green")
    
    quant_model = quantization(model, "float32")
    with open(f"{SAVE_DIR}/{model_name}/quantized-f32_{model_name}.bin", mode="wb") as quant_model_output:
        quant_model_output.write(quant_model)
    cp.cprint("Saved Tensorflow Lite format (.bin)", "green")

    with open(f"{SAVE_DIR}/{model_name}/compose.txt", mode="a") as f:
        print(f"Autoencoder, {ae_name}, {decode_ae_id(ae_id, fold)}", file=f)
        print(f"CNN, {cnn_name}, {cnn_id(ae_id, fold)}", file=f)
        
    cp.cprint(f"All component has completed!\n", "pink")


def quantization(model, mode:str="float32"):
    """
    TFLiteに変換
    """
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if (mode == "float16"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
    
    elif (mode == "int8"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
    
    elif (mode == "float32"): pass
    else: raise "You must choice on [float32, float16, int8]."
    
    tflite_quant_model = converter.convert()
    
    return tflite_quant_model


def main():
    
    Utils.makedir(SAVE_DIR)
    
    for ae in AE_ID:
        for cnn in CNN_ID:
            for fold in range(1, 6):
                marge_model(ae, cnn, fold, False)
    
    # excepts = [
    #     [AE_ID[0], CNN_ID[0], 1],
    #     [AE_ID[0], CNN_ID[0], 5],
    #     [AE_ID[1], CNN_ID[0], 2],
    #     [AE_ID[1], CNN_ID[0], 5],
    # ]
    
    # for e in excepts:
    #     ae, cnn, fold = e
    #     marge_model(ae, cnn, fold, True)


main()
