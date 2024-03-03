import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time

## my modules
from colorPrint import Cprint as cp


DIR = "/workspace/fullframe/result/540x540"
MODEL = "e-unet_20220629_AutoLearning_fold1_576x576"
# FILE = "quantized_model.bin"
FILE = "quantized_model_f32.bin"
RESULT_TEXT = "speed.txt"
BUF = "-f32"


class Test:

    LOOP = 100
    
    @staticmethod
    def quantized_model(model_path, head):
        """
        @機能：量子化モデル
        @引数：
        @戻値：
        """
        print(f"Test : quantized_model ( {model_path} )\n", "orange")
        
        start = time.time()
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        
        time_load = time.time() - start
        image = take_image(576)
        start = time.time()
        
        for l in range(Test.LOOP):
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            cp.cprint(f"\033[1A{head} | {l+1} / {Test.LOOP} | {round(time.time() - start, 2)}", "cyan")
        
        time_diff = time.time() - start
        
        return head, time_load, time_diff, time_diff/Test.LOOP

    
    @staticmethod
    def normal_model(model_path, head):
        """
        @機能：普通のモデル
        @引数：
        @戻値：
        """
        print(f"Test : quantized_model ( {model_path} )\n", "orange")
        
        start = time.time()
        
        model = load_model(model_path)
        
        time_load = time.time() - start
        image = take_image(576)
        start = time.time()
        
        for l in range(Test.LOOP):
            output_data = model.predict(image)
            cp.cprint(f"\033[1A{head} | {l+1} / {Test.LOOP} | {round(time.time() - start, 2)}", "cyan")
        
        time_diff = time.time() - start
        
        return head, time_load, time_diff, time_diff/Test.LOOP
    
    
def take_image(size:int=576):
    """
    @機能：
    @引数：
    @戻値：
    """    
    
    file = "/workspace/Dataset/fullframe/image/190731/04_1525.png"
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((size, size))
    data = np.asarray(image)
    data = data[np.newaxis,:, :, :]
    input_data = np.array(data, dtype=np.float32)

    return input_data


def write_data(data:list=None, *, initialize:bool=False):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    if initialize:
        with open(f"{DIR}/{MODEL}/{RESULT_TEXT}", mode="w") as f:
            f.write("head load[s] predict[s] predict[s/image]\n")
    
    else:
        with open(f"{DIR}/{MODEL}/{RESULT_TEXT}", mode="a") as f:
            print(" ".join(map(str, data)), file=f)
    
    
def main():
    
    # write_data(initialize=True)
    
    quant_results = [f"quant{BUF}-avg", 0, 0, 0]
    normal_results = ["normal-avg", 0, 0, 0]
    
    _ = Test.quantized_model(f"{DIR}/{MODEL}/{FILE}", f"quant-init")
    for head in range(5):
        quant = Test.quantized_model(f"{DIR}/{MODEL}/{FILE}", f"quant{BUF}-{head}")
        write_data(quant)
        quant_results[1] += quant[1] / 5.
        quant_results[2] += quant[2] / 5.
        quant_results[3] += quant[3] / 5.
    write_data(quant_results)
    
    # _ = Test.normal_model(f"{DIR}/{MODEL}", f"normal-init")
    # for head in range(5):
    #     normal = Test.normal_model(f"{DIR}/{MODEL}", f"normal-{head}")
    #     write_data(normal)
    #     normal_results[1] += normal[1] / 5.
    #     normal_results[2] += normal[2] / 5.
    #     normal_results[3] += normal[3] / 5.
    # write_data(normal_results)

main()
