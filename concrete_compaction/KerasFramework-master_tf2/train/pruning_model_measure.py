import numpy as np
import os
import time
import tensorflow as tf

# my module
from mymodel.SemSegLight import E_UNet
from colorPrint import Cprint as cp

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DATASET_DIR = "/workspace/Dataset/fullframe"
DIR = "/workspace/fullframe/result/pruning_model/filter-reduce-trail"
SIZE = [1, 576, 576, 3]
LIMIT = 200
SKIP = 1
RANDOM_SEED = 1
TEST_SKIP = 10


TARGETS = [
    # [True, "AE_e-unet_20221014_ssim_mse_50", [1]],
    # [False, "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning_20", [3]],
    [False, lambda *x: f"{x[0]:02d}_fold{x[1]}", [3]]
]


def measure(pruned:bool=True):
    
    np.random.seed(seed=RANDOM_SEED)
    img = np.float32(np.random.rand(*SIZE))
    
    if pruned:
        
        for target in TARGETS:
            
            for sparsity in [96, 92, 88, 84, 80]:
            
                is_ae, path_f, folds = target
                
                reduce_const = sparsity / 100
                
                for fold in folds:
                    path = path_f(100 - sparsity, fold)
                    cp.cprint(f"fold : {fold}", "orange")
                    cp.cprint(f"path : {path}", "pink")
                    
                    if (sparsity != 1):
                        model = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False, reduce_const=reduce_const)
                        model.load_weights(f"{DIR}/{path}/pruned_model.h5")
    
                    interpreter = tf.lite.Interpreter(model_path=f"{DIR}/{path}/quantized_model.tflite")
                    interpreter.allocate_tensors()
                    input_index = interpreter.get_input_details()[0]["index"]
                    output_index = interpreter.get_output_details()[0]["index"]
                    
                    result_d_text = f"{DIR}/{path}/measurement_details.txt"
                    result_text = f"{DIR}/{path}/measurement.txt"
                    with open(result_d_text, mode="w"): pass

                    values = [0]*LIMIT
                    
                    print()
                    with open(result_d_text, mode="a") as f:
                        for i in range(LIMIT):
                            start = time.time()
                            # pred = model.predict(np.copy(img), verbose=0)
                            interpreter.set_tensor(input_index, np.copy(img))
                            interpreter.invoke()
                            output = interpreter.tensor(output_index)
                            pred = np.copy(output()[0])
                            end = time.time()
                            diff = end - start
                            
                            values[i] = diff
                            
                            print(diff, file=f)
                            
                            cp.cprint(f"\033[1A{i+1} / {LIMIT}", "green")
                    
                    avg_val = sum(values) / LIMIT
                    max_val = max(values)
                    min_val = min(values)
                    mean_val = (list(sorted(values)))[LIMIT // 2]
                    
                    params = model.count_params() if sparsity != 1 else 0
                    
                    print("\n")
                    
                    with open(result_text, mode="w") as f:
                        f.write(f"""\
[second]
params min max avg mean
{params} {min_val} {max_val} {avg_val} {mean_val}
""")

    else:
        
        target = "/workspace/fullframe/result/540x540/e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning_fold3_576x576"
        model = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False)
        model.load_weights(target)

        path = "00_fold3"
        interpreter = tf.lite.Interpreter(model_path=f"{DIR}/{path}/quantized_model.tflite")
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
                    
        result_d_text = f"{DIR}/00_fold3/measurement_details.txt"
        result_text = f"{DIR}/00_fold3/measurement.txt"
        with open(result_d_text, mode="w"): pass

        values = [0]*LIMIT
        
        print()
        with open(result_d_text, mode="a") as f:
            for i in range(LIMIT):
                start = time.time()
                # pred = model.predict(np.copy(img), verbose=0)
                interpreter.set_tensor(input_index, np.copy(img))
                interpreter.invoke()
                output = interpreter.tensor(output_index)
                pred = np.copy(output()[0])
                end = time.time()
                diff = end - start
                
                values[i] = diff
                
                print(diff, file=f)
                
                cp.cprint(f"\033[1A{i+1} / {LIMIT}", "green")
        
        avg_val = sum(values) / LIMIT
        max_val = max(values)
        min_val = min(values)
        mean_val = (list(sorted(values)))[LIMIT // 2]
        
        params = model.count_params()
        
        print("\n")
        
        with open(result_text, mode="w") as f:
            f.write(f"""\
[second]
params min max avg mean
{params} {min_val} {max_val} {avg_val} {mean_val}
""")
        
        
def main():
    
    measure(1)
    # measure(0)


main()
