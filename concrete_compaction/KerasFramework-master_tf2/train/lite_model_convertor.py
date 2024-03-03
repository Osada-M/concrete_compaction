import tensorflow as tf
from tensorflow.keras.models import load_model

## my modules
from my_loss_function import MyLosses
from colorPrint import Cprint as cp


DIR = "/workspace/fullframe/result/540x540"
# DIR = "/workspace/fullframe/result/autoencoder"
# DIR = "/workspace/fullframe/result/pruning_model/filter-reduce-trail"
# MODEL = lambda fold: f"e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning_fold{fold}_576x576"
MODEL = lambda fold: f"e-unet_4class_use-AE-input_20221014_ssim_mse_rotate_reduce-0.5_20221209_AutoLearning_fold3_576x576"
# MODEL = lambda fold: f"AE_e-unet_20221014_ssim_mse_fold{fold}"
# MODEL = lambda fold: f"{fold:02d}_fold3"

SAVE_DIR = "/workspace/fullframe/result/pruning_model/filter-reduce-trail"
# FILE = lambda fold: f"AE_fold{fold}.tflite"
FILE = "quantized_model.tflite"


def quantization(model, mode:str="float32"):
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
    else: raise "float32, float16, int8のどれかにしてください。"
    
    tflite_quant_model = converter.convert()
    
    return tflite_quant_model


def save(path, model):
    
    with open(path, mode="wb") as model_output:
        model_output.write(model)


def main():
    
    # for fold in range(1, 6):
    for fold in [4, 8, 12, 16, 20, 24, 28, 32]:
        print()
        
        model = load_model(f"{DIR}/{MODEL(fold)}", custom_objects={"ssim_mse_loss" : MyLosses.ssim_mse_loss})
        # model = load_model(f"{DIR}/{MODEL(fold)}/rectified_pruned_model.h5", custom_objects={"ssim_mse_loss" : MyLosses.ssim_mse_loss})
        cp.cprint(f"Loaded \"{DIR}/{MODEL(fold)}/saved_model.pb\"", "orange")
        # cp.cprint(f"Loaded \"{DIR}/{MODEL(fold)}/rectified_pruned_model.h5\"", "orange")
        
        # tflite_quant_model = quantization(model, "int8")
        tflite_quant_model = quantization(model, "float32")
        cp.cprint("Quantized a model.", "yellow")
        
        # save(f"{SAVE_DIR}/{MODEL(fold)}/{FILE(fold)}", tflite_quant_model)
        # save(f"{SAVE_DIR}/{FILE(fold)}", tflite_quant_model)
        # save(f"{SAVE_DIR}/{MODEL(fold)}/{FILE}", tflite_quant_model)
        save(f"{SAVE_DIR}/{FILE}", tflite_quant_model)
        cp.cprint("Saved a quantized model.", "green")
        
        break

main()
