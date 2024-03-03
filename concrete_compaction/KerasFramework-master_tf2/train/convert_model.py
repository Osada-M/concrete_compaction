from typing import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D


from mymodel.SemSegLight import E_UNet
from colorPrint import Cprint as cp
from Optimizer import Optimizer


class After:
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def convert(model, loss="categorical_crossentropy", dropout_const:float=0.01, learning_rate:float=1e-3, optimizer:str="adam"):
        """
        Bottleneckまるごと
        """
        
        model_c3 = E_UNet.run((576, 576, 3), num_classes=3, dropout_const=dropout_const, is_compile=False)
        available_c3 = False
        
        after_model = Sequential()
        
        for i, (layer, c3_layer) in enumerate(zip(model.layers, model_c3.layers)):
            if After.is_bottlenack(layer.name):
                available_c3 = True
                cp.cprint(f"@ Recgnized No.{i} conv layer is classifier de-bottleneck input.", "orange")
                after_model.add(Model(inputs=model.layers[1].input, outputs=model.layers[i-1].output))
                # break
            
            print(layer.name)
            if available_c3:
                after_model.add(c3_layer)
            # else:
            #     after_model.add(layer)
                
            # if ("classifier" in layer.name): break
        
        
        # after_model.add(Model(inputs=model_c3.layers[i].input, outputs=model.outputs))
        # after_model.add(Conv2D(filters=3, 
        #                         kernel_size=1,
        #                         activation="softmax",
        #                         name="classifier"))
        
        my_optimizer_func = Optimizer.decide_optimizer(optimizer)
        after_model.compile(optimizer=my_optimizer_func(learning_rate=learning_rate), 
                            loss=loss, 
                            metrics=["acc"],
                            run_eagerly=True)
        
        return after_model
    
    
    @staticmethod
    def convert_v0(model, loss="categorical_crossentropy", learning_rate:float=1e-3, optimizer:str="adam"):
        """
        Classifierのみ
        """
        
        for i, layer in enumerate(model.layers):
            if ("classifier" in layer.name): break
            
        after_model = Sequential()
        after_model.add(Model(inputs=model.inputs, outputs=model.layers[i-1].output))
        after_model.add(Conv2D(filters=2,
                                kernel_size=1,
                                activation="softmax",
                                name="classifier"))
        
        my_optimizer_func = Optimizer.decide_optimizer(optimizer)
        after_model.compile(optimizer=my_optimizer_func(learning_rate=learning_rate), 
                            loss=loss, 
                            metrics=["acc"],
                            run_eagerly=True)
        
        return after_model


    @staticmethod
    def is_bottlenack(name):
        """
        Bottleneckの入力層の発見
        """
        
        name = name.replace("conv2d", "").replace("_", "")
        
        try:
            name = (int(name) - 151) % 178
        except:
            return False
        
        return not bool(name)


if(__name__ == "__main__"):
    
    model_3 = E_UNet.run((576, 576, 3), 3)
    model_4 = E_UNet.run((576, 576, 3), 4)
    model_5 = E_UNet.run((576, 576, 3), 5)
    
    for l3, l4, l5 in zip(model_3.layers, model_4.layers, model_5.layers):
        l3_shape = l3.get_output_at(0).get_shape().as_list()[1:]
        l4_shape = l4.get_output_at(0).get_shape().as_list()[1:]
        l5_shape = l5.get_output_at(0).get_shape().as_list()[1:]
        
        if After.is_bottlenack(l3.name) and After.is_bottlenack(l4.name) and After.is_bottlenack(l5.name):
            print(l3_shape, l4_shape, l5_shape)
        