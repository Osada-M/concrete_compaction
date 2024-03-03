import tensorflow as tf
import numpy as np
from PIL import Image
import os

# my module
from MyUtils import Utils, Calc, TimeCounter
from mymodel.SemSegLight import E_UNet
from colorPrint import Cprint as cp
from MyPruning import MyPruning


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ================ config ================


DIR = "/workspace/fullframe/result/540x540"
RESULT_DIR = "/workspace/fullframe/result/pruning_model/pruned_map"
LOAD_ID = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"


## ========================================


MyPruning.debug = True


def take():
    
    model_bf = tf.keras.models.load_model(f"{DIR}/{LOAD_ID}_fold3_576x576")
    model_af = E_UNet.run((576, 576, 3), num_classes=4, reduce_const=0.5)

    # for reduce in ["ssim", "error", "square-error", "zero"]:
    for reduce in ["error", "square-error", "zero"]:
        
        cp.cprint(f"reduce : {reduce}\n", "green")
        
        path = f"{RESULT_DIR}/{reduce}"
        Utils.makedir(path)
        length = len(model_bf.layers)
        
        for i, name, weights, next, matrix, vals in MyPruning.prune_tuning(model_before=model_bf, model_after=model_af, reduce=reduce):
            if (i < 0): break
            
            Utils.makedir(f"{path}/{i}_{name}")
            
            row, col, be, af = weights.shape
            with open(f"{path}/{i}_{name}/config.txt", mode="w") as f:
                print(f"shape : {row} x {col}\nnext : {' '.join(map(str, next))}", file=f)
                for j in range(weights.shape[-1]):
                    print(f"{j}\t{['x', ''][j in next]}", file=f)
            
            weights -= np.min(weights)
            mx = np.max(weights)
            
            mat_img = matrix - np.min(matrix)
            mat_img /= np.max(mat_img)
            mat_img = Image.fromarray(np.uint8(mat_img * 255.))
            mat_img.save(f"{path}/{i}_{name}/matrix.png")
            
            val_img = vals - np.min(vals)
            val_img /= np.max(val_img)
            val_img = Image.fromarray(np.uint8(val_img * 255.))
            val_img.save(f"{path}/{i}_{name}/values.png")
            
            with open(f"{path}/{i}_{name}/matrix.txt", mode="w") as f:
                for mat_row in matrix:
                    print(" ".join(map(lambda x: f"{x:.03f}", mat_row)), file=f)
                    
            with open(f"{path}/{i}_{name}/values.txt", mode="w") as f:
                for val_row in vals:
                    print(" ".join(map(str, val_row)), file=f)
            
            for b in range(be):
                folder = f"{path}/{i}_{name}/{b}"
                Utils.makedir(folder)
                
                for a in range(af):
                    rm_flag = a in next
                    w = weights[:, :, b, a]
                    image = Image.fromarray(np.uint8(w/mx*255.))
                    
                    image.save(f"{folder}/{a}{['_rm', ''][rm_flag]}.png")
            

def main():
    
    Utils.makedir(RESULT_DIR)
    take()


main()
