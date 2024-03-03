from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, concatenate, Activation
from tensorflow.keras.layers import Conv2D, Input, Dense, Conv2DTranspose, Conv1D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Reshape, LayerNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU, Add, InputSpec, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda, MaxPooling2D, Reshape, ZeroPadding2D

# my modelu
import MetricLearning_for_semseg as metric_semseg
from colorPrint import Cprint as cp
from Normalization import Normalization as norms


class ClassDivider:
    """
    @機能：
    @引数：
    @戻値：
    """
    
    @staticmethod
    def three_classes_image(input_shape=(256,256,3), fresh_shape=(5), loss:str="categorical_crossentropy", is_compile:bool=True, is_classifier:bool=True, norm:str="batch_norm"):
        """
        @機能：
        @引数：input_shape=(256, 256, *)を想定
        @戻値：
        """
        
        l2_reg = 0.0001
        
        input = Input(input_shape, name="image_input")
        fresh = Input(fresh_shape, name="fresh_input")
        
        image_model = ClassDivider.expert_CNN(input, mode="256", l2_reg=l2_reg)
        fresh_model = ClassDivider.expert_Dense(fresh, mode="256")
        
        # model = Add(image_model, fresh_model)
        x = concatenate([image_model, fresh_model])
        
        x = Dense(256)(x)
        x = norms.decide_norm(norm)(x)
        x = Activation("relu", name="before_classifier")(x)
        
        if is_classifier:
            output = Dense(3, name="classifier")(x)
            output = Activation("softmax")(x)
            model = Model([input, fresh], output)
        else:
            model = Model([input, fresh], x)
        
        if is_compile:
            model.compile(optimizer=Adam(lr=0.001),
                        loss=loss,
                        metrics=["acc"])
        
        return model
    
    
    @staticmethod
    def divide_classes_difftime(loss:str="categorical_crossentropy", classes:int=4, is_compile:bool=True, is_classifier:bool=True, norm:str="batch_norm", dropout_const:float=0.1):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        # l2_reg = 0.0001
        
        nodes = 1<<10
        
        input = Input((6))
        
        x_0_0 = ClassDivider.dense_block(input, nodes, norm)
        x_0_1 = ClassDivider.dense_block(x_0_0, nodes, norm)
        x_0_2 = ClassDivider.dense_block(x_0_1, nodes, norm)
        x_0_3 = ClassDivider.dense_block(x_0_2, nodes, norm)
        
        x_0_c = concatenate([x_0_0, x_0_3])
        # x = Dropout(dropout_const)(x)
        
        x_1_0 = ClassDivider.dense_block(x_0_c, nodes>>1, norm)
        x_1_1 = ClassDivider.dense_block(x_1_0, nodes>>1, norm)
        x_1_2 = ClassDivider.dense_block(x_1_1, nodes>>1, norm)
        x_1_3 = ClassDivider.dense_block(x_1_2, nodes>>1, norm)
        
        x_1_c = concatenate([x_1_0, x_1_3])
        # x = Dropout(dropout_const)(x)
        
        x_2_0 = ClassDivider.dense_block(x_1_c, nodes>>2, norm)
        x_2_1 = ClassDivider.dense_block(x_2_0, nodes>>2, norm)
        x_2_2 = ClassDivider.dense_block(x_2_1, nodes>>2, norm)
        x_2_3 = ClassDivider.dense_block(x_2_2, nodes>>2, norm)
        
        x_2_c = concatenate([x_2_0, x_2_3])
        # x = Dropout(dropout_const)(x)
        
        x_3_0 = ClassDivider.dense_block(x_2_c, nodes>>3, norm)
        x_3_1 = ClassDivider.dense_block(x_3_0, nodes>>3, norm)
        x_3_2 = ClassDivider.dense_block(x_3_1, nodes>>3, norm)
        x_3_3 = ClassDivider.dense_block(x_3_2, nodes>>3, norm)
        
        x_3_c = concatenate([x_3_0, x_3_3])
        
        x_3_4 = ClassDivider.dense_block(x_3_c, nodes>>3, norm)
        x_3_5 = ClassDivider.dense_block(x_3_4, nodes>>3, norm)
        x_3_6 = ClassDivider.dense_block(x_3_5, nodes>>3, norm)
        x_3_7 = ClassDivider.dense_block(x_3_6, nodes>>3, norm)
        
        x_drop = Dropout(dropout_const)(x_3_7)
        
        x_final = Dense(classes)(x_drop)
        output = Activation("softmax")(x_final)
        
        model = Model(input, output)
        
        if is_compile:
            model.compile(optimizer=Adam(lr=0.001),
                        loss=loss,
                        metrics=["acc"])
        
        return model
        
        
    @staticmethod
    def conv_block(input, kernel_size, filters, l2_reg, norm:str="batch_norm"):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg))(input)
        x = norms.decide_norm(norm)(x)
        
        return x
    
    
    @staticmethod
    def dense_block(input, nodes, norm:str="batch_norm"):
        """
        @機能：
        @引数：
        @戻値：
        """
        x = Dense(nodes)(input)
        x = norms.decide_norm(norm)(x)
        x = Activation("relu")(x)
        
        return x
    
    
    @staticmethod
    def expert_CNN(input, mode:str="256", l2_reg:float=0.0001):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        x = None
        
        if (mode == "256"):
            x = ClassDivider.conv_block(input, kernel_size=3, filters=32, l2_reg=l2_reg)
            x = ClassDivider.conv_block(x, kernel_size=3, filters=32, l2_reg=l2_reg)
            x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
            
            x = ClassDivider.conv_block(x, kernel_size=3, filters=64, l2_reg=l2_reg)
            x = ClassDivider.conv_block(x, kernel_size=3, filters=64, l2_reg=l2_reg)
            x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
            
            x = ClassDivider.conv_block(x, kernel_size=3, filters=128, l2_reg=l2_reg)
            x = ClassDivider.conv_block(x, kernel_size=3, filters=128, l2_reg=l2_reg)
            x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
            
            x = ClassDivider.conv_block(x, kernel_size=3, filters=256, l2_reg=l2_reg)
            x = ClassDivider.conv_block(x, kernel_size=3, filters=256, l2_reg=l2_reg)
            x = GlobalAveragePooling2D()(x)
        
        return x
    
    
    @staticmethod
    def expert_Dense(input, mode="256", norm="batch_norm"):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        x = None
        if (mode == "256"):
            x = ClassDivider.dense_block(input, 256)
            x = ClassDivider.dense_block(x, 256)
        
        return x
    
