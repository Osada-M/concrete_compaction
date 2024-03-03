import tensorflow as tf
import os
import numpy as np
import time


class ModelMaster:
    
    def __init__(self):
        
        self.size = [576, 576, 3]
        self.dir = "/workspace/fullframe/result/pruning_model/speed_check"
        self.epochs = 100
    
    
    def make_models(self, ):
        
        if (__name__ == "__main__"):
            from mymodel.SemSegLight import E_UNet
        else: return
        
        for reduce_const in [1, .5, .25, .75]:
            
            path = f"{self.dir}/models/{int(100*reduce_const)}.tflite"
            
            model = tf.keras.models.Sequential()
            
            model_a = E_UNet.run(self.size, num_classes=3, is_compile=True, reduce_const=reduce_const)
            model.add(model_a)
            del model_a
            
            model_b = E_UNet.run(self.size, num_classes=4, is_compile=True, reduce_const=reduce_const)
            model.add(model_b)
            del model_b
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_quant_model = converter.convert()
            with open(path, mode="wb") as model_output:
                model_output.write(tflite_quant_model)
            
            print(f"Completed {reduce_const}")
        
    
    def measurement(self, ):
        
        for reduce_const in [1, .75, .5, .25]:
            
            path = f"{self.dir}/models/{int(100*reduce_const)}.tflite"
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            input_index = interpreter.get_input_details()[0]["index"]
            output_index = interpreter.get_output_details()[0]["index"]
            
            img = np.float32(np.ones((1, *self.size)))
    
            result_d_text = f"{self.dir}/measurement/{int(100*reduce_const)}_details.txt"
            result_text = f"{self.dir}/measurement/{int(100*reduce_const)}.txt"
            with open(result_d_text, mode="w"): pass

            values = [0]*self.epochs
            
            print()
            with open(result_d_text, mode="a") as f:
                for i in range(self.epochs):
                    start = time.time()
                    
                    interpreter.set_tensor(input_index, np.copy(img))
                    interpreter.invoke()
                    output = interpreter.tensor(output_index)
                    pred = np.copy(output()[0])
                    
                    end = time.time()
                    diff = end - start
                    values[i] = diff
                    
                    print(diff, file=f)
                    print(f"\033[1A{i+1} / {self.epochs}")
            
            avg_val = sum(values) / self.epochs
            max_val = max(values)
            min_val = min(values)
            mean_val = (list(sorted(values)))[self.epochs // 2]
            
            print("\n")
            
            with open(result_text, mode="w") as f:
                f.write(f"""\
[second]
min max avg mean
{min_val} {max_val} {avg_val} {mean_val}
""")
                        

def main():
    
    mm = ModelMaster()
    mm.measurement()


main()
