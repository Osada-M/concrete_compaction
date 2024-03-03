# from email.mime import base
# from os import stat

# from yaml import load
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, concatenate, Activation
from tensorflow.keras.layers import Conv2D, Input, Dense, Conv2DTranspose, Conv1D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Reshape, LayerNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU, Add, InputSpec, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.backend import int_shape, resize_images, image_data_format
from tensorflow.python.keras.layers import Lambda
# from tensorflow.keras.engine import InputSpec
# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Lambda, MaxPooling2D, Permute, Reshape, ZeroPadding2D, add, multiply
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.utils import conv_utils

# my modelu
import MetricLearning_for_semseg as metric_semseg
from colorPrint import Cprint as cp
from Normalization import Normalization as norms# BatchInstanceNormalization
from my_loss_function import MyLosses


class CroppingLike2D(Layer):
    def __init__(self, target_shape, offset=None, data_format=None, **kwargs):
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or (offset == "centered"):
            self.offset = "centered"
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, "__len__"):
            if (len(offset) != 2):
                raise ValueError(
                    "`offset` should have two elements. " "Found: " +
                        str(offset)
                )
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)


    def compute_output_shape(self, input_shape):
        if (self.data_format == "channels_first"):
            return (
                input_shape[0],
                input_shape[1],
                self.target_shape[2],
                self.target_shape[3],
            )
        else:
            return (
                input_shape[0],
                self.target_shape[1],
                self.target_shape[2],
                input_shape[3],
            )


    def call(self, inputs):
        input_shape = int_shape(inputs)
        if self.data_format == "channels_first":
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError(
                    "The Tensor to be cropped need to be smaller"
                    "or equal to the target Tensor."
                )

            if self.offset == "centered":
                self.offset = [
                    int((input_height - target_height) / 2),
                    int((input_width - target_width) / 2),
                ]

            if self.offset[0] + target_height > input_height:
                raise ValueError(
                    "Height index out of range: " + str(self.offset[0] + target_height)
                )
            if self.offset[1] + target_width > input_width:
                raise ValueError(
                    "Width index out of range:" + str(self.offset[1] + target_width)
                )

            return inputs[
                :,
                :,
                self.offset[0] : self.offset[0] + target_height,
                self.offset[1] : self.offset[1] + target_width,
            ]
        elif self.data_format == "channels_last":
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError(
                    "The Tensor to be cropped need to be smaller"
                    "or equal to the target Tensor."
                )

            if self.offset == "centered":
                self.offset = [
                    int((input_height - target_height) / 2),
                    int((input_width - target_width) / 2),
                ]

            if self.offset[0] + target_height > input_height:
                raise ValueError(
                    "Height index out of range: " + str(self.offset[0] + target_height)
                )
            if self.offset[1] + target_width > input_width:
                raise ValueError(
                    "Width index out of range:" + str(self.offset[1] + target_width)
                )
            output = inputs[
                :,
                self.offset[0] : self.offset[0] + target_height,
                self.offset[1] : self.offset[1] + target_width,
                :,
            ]
            return output


class BilinearUpSampling2D(Layer):
    def __init__(self, target_shape=None, factor=None, data_format=None, **kwargs):
        # conmpute dataformat
        if data_format is None:
            data_format = image_data_format()
        assert data_format in {"channels_last", "channels_first"}

        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == "channels_first":
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == "channels_last":
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            return (
                input_shape[0],
                self.target_size[0],
                self.target_size[1],
                input_shape[3],
            )
        else:
            return (
                input_shape[0],
                input_shape[1],
                self.target_size[0],
                self.target_size[1],
            )

    def call(self, inputs):
        return resize_images(inputs, self.factor, self.factor, self.data_format)

    def get_config(self):
        config = {"target_shape": self.target_shape, "data_format": self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SemanticSegmentation:
    
    @staticmethod
    def adj_concat(base, target):
        base_h = base.shape[1]
        base_w = base.shape[2]
        target_h = target.shape[1]
        target_w = target.shape[2]        
        diff_h = base_h - target_h
        diff_w = base_w - target_w
        
        if diff_h:
            pad_h = abs(diff_h)
            if not (diff_h % 2):
                if(diff_h > 0):
                    target = ZeroPadding2D(padding=(int(pad_h/2), 0))(target)
                else:
                    base = ZeroPadding2D(padding=(int(pad_h/2), 0))(base)
            else:
                if(diff_h > 0):
                    for i in range(pad_h):
                        if(i % 2):
                            target = ZeroPadding2D(padding=((1, 0), (0, 0)))(target)
                        else:
                            target = ZeroPadding2D(padding=((0, 1), (0, 0)))(target)
                else:
                    for i in range(pad_h):
                        if(i % 2):
                            base = ZeroPadding2D(padding=((1, 0), (0, 0)))(base)
                        else:
                            base = ZeroPadding2D(padding=((0, 1), (0, 0)))(base)
        if diff_w:
            pad_w = abs(diff_w)
            if not (diff_w % 2):
                if(diff_w > 0):
                    target = ZeroPadding2D(padding=(0, int(pad_w/2)))(target)
                else:
                    base = ZeroPadding2D(padding=(0, int(pad_w/2)))(base)
            else:
                if(diff_w > 0):
                    for i in range(pad_w):
                        if(i % 2):
                            target = ZeroPadding2D(padding=((0, 0), (1, 0)))(target)
                        else:
                            target = ZeroPadding2D(padding=((0, 0), (0, 1)))(target)
                else:
                    for i in range(pad_w):
                        if(i % 2):
                            base = ZeroPadding2D(padding=((0, 0), (1, 0)))(base)
                        else:
                            base = ZeroPadding2D(padding=((0, 0), (0, 1)))(base)
                    
        return base, target


    @staticmethod
    def create_conv(input, filters, l2_reg, name, norm:str="batch_norm"):
        x = Conv2D(filters=filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
                name=name)(input)
        x = norms.decide_norm(norm)(x)
        return x


    @staticmethod
    def create_trans(input, filters, l2_reg, name, norm:str="batch_norm"):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=2,
                            strides=2,
                            activation="relu",
                            padding="same",
                            kernel_regularizer=regularizers.l2(l2_reg),
                            name=name)(input)
        x = norms.decide_norm(norm)(x)
        return x


    @staticmethod
    def tuning_model(model=None, params=None):
        if model is None: model = Sequential()
        model.add(Conv2D(filters=3,
                         kernel_size=params["kernel_size_0"],
                         strides=params["strides_0"],
                         activation=params["activation_0"],
                         padding="valid",
                         kernel_regularizer=regularizers.l2(params["l2_reg"]),
                         name="first_convolution_0"))
        model.add(BatchNormalization(name="batch_norm_0"))
        model.add(Conv2D(filters=3,
                         kernel_size=params["kernel_size_1"],
                         strides=params["strides_1"],
                         activation=params["activation_1"],
                         padding="valid",
                         kernel_regularizer=regularizers.l2(params["l2_reg"]),
                         name="first_convolution_1"))
        model.add(BatchNormalization(name="batch_norm_1"))
        
        return model
    
    
    @staticmethod
    def model_compiler(model):
        model.compile(optimizer=Adam(lr=0.001),
                     loss="categorical_crossentropy",
                     metrics=["acc"])
        return model
    
    
    @staticmethod
    def pointwise_conv(x, filters, name, weight_decay=1e-4):
        """
        PW畳み込み(距離学習用)
        """
        x = Conv2D(filters=filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=regularizers.l2(weight_decay),
                    name=name)(x)
        return x
    
    
    @staticmethod
    def naive_conv_block(x, filters, kernel_size, name, weight_decay=1e-4, is_bottle=False):
        """
        畳み込み(距離学習用)
        """
        if is_bottle:
            x = SemanticSegmentation.pointwise_conv(x, filters, f"{name}_before", weight_decay)
        x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=regularizers.l2(weight_decay),
                    name=name)(x)
        if is_bottle:
            x = SemanticSegmentation.pointwise_conv(x, filters, f"{name}_after", weight_decay)
            
        return x
    
    
    @staticmethod
    def naive_deconv_block(x, filters, kernel_size, name, weight_decay=1e-4, is_bottle=False):
        """
        逆畳み込み(距離学習用)
        """
        if is_bottle:
            x = SemanticSegmentation.pointwise_conv(x, filters, f"{name}_before", weight_decay)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=kernel_size,
                            padding="same",
                            activation="relu",
                            kernel_regularizer=regularizers.l2(weight_decay),
                            name=name)(x)
        if is_bottle:
            x = SemanticSegmentation.pointwise_conv(x, filters, f"{name}_after", weight_decay)
            
        return x
    
    
    @staticmethod
    def unet(input_shape=None, loss:str="categorical_crossentropy", is_compile:bool=True, is_classifier:bool=True, norm:str="batch_norm", num_classes:int=2):
        """
        @概要：U-Net
        @引数：input_shape = コンクリ画像の形状, loss:str = 損失関数
        @戻値：コンパイル済のモデル
        """
        l2_reg = 0.0001

        input = Input(input_shape)

        conv1_1 = SemanticSegmentation.create_conv(input, filters=64, l2_reg=l2_reg, name="conv1c_1", norm=norm)
        conv1_2 = SemanticSegmentation.create_conv(conv1_1, filters=64, l2_reg=l2_reg, name="conv1c_2", norm=norm)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1")(conv1_2)

        conv2_1 = SemanticSegmentation.create_conv(pool1, filters=128, l2_reg=l2_reg, name="conv2c_1", norm=norm)
        conv2_2 = SemanticSegmentation.create_conv(conv2_1, filters=128, l2_reg=l2_reg, name="conv2c_2", norm=norm)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")(conv2_2)

        conv3_1 = SemanticSegmentation.create_conv(pool2, filters=256, l2_reg=l2_reg, name="conv3c_1", norm=norm)
        conv3_2 = SemanticSegmentation.create_conv(conv3_1, filters=256, l2_reg=l2_reg, name="conv3c_2", norm=norm)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool3")(conv3_2)
        
        conv4_1 = SemanticSegmentation.create_conv(pool3, filters=512, l2_reg=l2_reg, name="conv4c_1", norm=norm)
        conv4_2 = SemanticSegmentation.create_conv(conv4_1, filters=512, l2_reg=l2_reg, name="conv4c_2", norm=norm)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool4")(conv4_2)
        
        conv5_1 = SemanticSegmentation.create_conv(pool4, filters=1024, l2_reg=l2_reg, name="conv5m_1", norm=norm)
        conv5_2 = SemanticSegmentation.create_conv(conv5_1, filters=1024, l2_reg=l2_reg, name="conv5m_2", norm=norm)
        trans1 = SemanticSegmentation.create_trans(conv5_2, filters=512, l2_reg=l2_reg, name="trans1", norm=norm)
        concat1 = concatenate(SemanticSegmentation.adj_concat(trans1, conv4_2))#, name="concat1")

        conv6_1 = SemanticSegmentation.create_conv(concat1, filters=512, l2_reg=l2_reg, name="conv6e_1", norm=norm)
        conv6_2 = SemanticSegmentation.create_conv(conv6_1, filters=512, l2_reg=l2_reg, name="conv6e_2", norm=norm)
        trans2 = SemanticSegmentation.create_trans(conv6_2, filters=256, l2_reg=l2_reg, name="trans2", norm=norm)
        concat2 = concatenate(SemanticSegmentation.adj_concat(trans2, conv3_2))#, name="concat2")

        conv7_1 = SemanticSegmentation.create_conv(concat2, filters=256, l2_reg=l2_reg, name="conv7e_1", norm=norm)
        conv7_2 = SemanticSegmentation.create_conv(conv7_1, filters=256, l2_reg=l2_reg, name="conv7e_2", norm=norm)
        trans3 = SemanticSegmentation.create_trans(conv7_2, filters=128, l2_reg=l2_reg, name="trans3", norm=norm)
        concat3 = concatenate(SemanticSegmentation.adj_concat(trans3, conv2_2))#, name="concat3")

        conv8_1 = SemanticSegmentation.create_conv(concat3, filters=128, l2_reg=l2_reg, name="conv8e_1", norm=norm)
        conv8_2 = SemanticSegmentation.create_conv(conv8_1, filters=128, l2_reg=l2_reg, name="conv8e_2", norm=norm)
        trans4 = SemanticSegmentation.create_trans(conv8_2, filters=64, l2_reg=l2_reg, name="trans4", norm=norm)
        concat4 = concatenate(SemanticSegmentation.adj_concat(trans4, conv1_2))#, name="concat4")

        conv9_1 = SemanticSegmentation.create_conv(concat4, filters=64, l2_reg=l2_reg, name="conv9e_1", norm=norm)
        conv9_2 = SemanticSegmentation.create_conv(conv9_1, filters=64, l2_reg=l2_reg, name="conv9e_2", norm=norm)

        if is_classifier:
            output = Conv2D(filters=num_classes,
                            kernel_size=1,
                            activation="softmax",
                            name="output")(conv9_2)
            model = Model(input, output)
        else:
            model = Model(input, conv9_2)
        
        if is_compile:
            model.compile(optimizer=Adam(lr=0.001),
                        loss=loss,
                        metrics=["acc"])
        
        return model


    ### ================================================================
    
    
    @staticmethod
    def unet_include_fresh(input_shape, fresh_shape, loss:str="categorical_crossentropy"):
        """
        @概要：フレッシュ性状データの入力を含んだU-Net
        @引数：input_shape = コンクリ画像の形状, fresh_shape = フレッシュ性状データの形状, loss:str = 損失関数
        @戻値：コンパイル済のモデル
        """
        l2_reg = 0.0001

        input = Input(input_shape, name="image_input")
        fresh = Input(fresh_shape, name="fresh_input")
        

        conv1_1 = SemanticSegmentation.create_conv(input, filters=64, l2_reg=l2_reg, name="conv1c_1")
        conv1_2 = SemanticSegmentation.create_conv(conv1_1, filters=64, l2_reg=l2_reg, name="conv1c_2")
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1")(conv1_2)

        conv2_1 = SemanticSegmentation.create_conv(pool1, filters=128, l2_reg=l2_reg, name="conv2c_1")
        conv2_2 = SemanticSegmentation.create_conv(conv2_1, filters=128, l2_reg=l2_reg, name="conv2c_2")
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")(conv2_2)

        conv3_1 = SemanticSegmentation.create_conv(pool2, filters=256, l2_reg=l2_reg, name="conv3c_1")
        conv3_2 = SemanticSegmentation.create_conv(conv3_1, filters=256, l2_reg=l2_reg, name="conv3c_2")
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool3")(conv3_2)

        conv4_1 = SemanticSegmentation.create_conv(pool3, filters=512, l2_reg=l2_reg, name="conv4c_1")
        conv4_2 = SemanticSegmentation.create_conv(conv4_1, filters=512, l2_reg=l2_reg, name="conv4c_2")
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, name="pool4")(conv4_2)
        
        conv5_1 = SemanticSegmentation.create_conv(pool4, filters=1024, l2_reg=l2_reg, name="conv5m_1")
        conv5_2 = SemanticSegmentation.create_conv(conv5_1, filters=1024, l2_reg=l2_reg, name="conv5m_2")
        ## pool ?
        
        #### fresh layer
        
        # fresh_conv1 = Conv2D(filters=256, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv1")(fresh)
        # fresh_batch1 = BatchNormalization()(fresh_conv1)
        # fresh_conv2 = Conv2D(filters=512, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv2")(fresh_batch1)
        # fresh_batch2 = BatchNormalization()(fresh_conv2)
        # fresh_conv3 = Conv2D(filters=1024, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv3")(fresh_batch2)
        # fresh_batch3 = BatchNormalization()(fresh_conv3)
        
        # concat_fresh = concatenate(SemanticSegmentation.adj_concat(conv5_2, fresh_batch3))
        
        # fresh_conv4 = Conv2D(filters=1024, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv4")(concat_fresh)
        
        fresh_conv1 = Conv2D(filters=256, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv1")(fresh)
        fresh_batch1 = BatchNormalization()(fresh_conv1)
        fresh_conv2 = Conv2D(filters=512, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv2")(fresh_batch1)
        fresh_batch2 = BatchNormalization()(fresh_conv2)
        fresh_conv3 = Conv2D(filters=1024, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv3")(fresh_batch2)
        fresh_batch3 = BatchNormalization()(fresh_conv3)
        
        concat_fresh = concatenate(SemanticSegmentation.adj_concat(conv5_2, fresh_batch3))
        
        fresh_conv4 = Conv2D(filters=1024, kernel_size=1, activation="relu", padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="fresh_conv4")(concat_fresh)
        # fresh_batch4 = BatchNormalization()(fresh_conv4)
        
        #### 
        
        # trans1 = SemanticSegmentation.create_trans(concat_fresh, filters=512, l2_reg=l2_reg, name="trans1")
        trans1 = SemanticSegmentation.create_trans(fresh_conv4, filters=512, l2_reg=l2_reg, name="trans1")
        concat1 = concatenate(SemanticSegmentation.adj_concat(trans1, conv4_2))

        conv6_1 = SemanticSegmentation.create_conv(concat1, filters=512, l2_reg=l2_reg, name="conv6e_1")
        conv6_2 = SemanticSegmentation.create_conv(conv6_1, filters=512, l2_reg=l2_reg, name="conv6e_2")
        trans2 = SemanticSegmentation.create_trans(conv6_2, filters=256, l2_reg=l2_reg, name="trans2")
        concat2 = concatenate(SemanticSegmentation.adj_concat(trans2, conv3_2))

        conv7_1 = SemanticSegmentation.create_conv(concat2, filters=256, l2_reg=l2_reg, name="conv7e_1")
        conv7_2 = SemanticSegmentation.create_conv(conv7_1, filters=256, l2_reg=l2_reg, name="conv7e_2")
        trans3 = SemanticSegmentation.create_trans(conv7_2, filters=128, l2_reg=l2_reg, name="trans3")
        concat3 = concatenate(SemanticSegmentation.adj_concat(trans3, conv2_2))

        conv8_1 = SemanticSegmentation.create_conv(concat3, filters=128, l2_reg=l2_reg, name="conv8e_1")
        conv8_2 = SemanticSegmentation.create_conv(conv8_1, filters=128, l2_reg=l2_reg, name="conv8e_2")
        trans4 = SemanticSegmentation.create_trans(conv8_2, filters=64, l2_reg=l2_reg, name="trans4")
        concat4 = concatenate(SemanticSegmentation.adj_concat(trans4, conv1_2))

        conv9_1 = SemanticSegmentation.create_conv(concat4, filters=64, l2_reg=l2_reg, name="conv9e_1")
        conv9_2 = SemanticSegmentation.create_conv(conv9_1, filters=64, l2_reg=l2_reg, name="conv9e_2")

        output = Conv2D(filters=2,
                        kernel_size=1,
                        activation="softmax",
                        name="classifier")(conv9_2)
        
        model = Model([input, fresh], output)
        model.compile(optimizer=Adam(lr=0.001),
                    loss=loss,
                    metrics=["acc"])
        
        return model
    
    
    ### ================================================================
    
    
    @staticmethod
    def unet_available_metric(base_model, face_name:str, input_shape, num_classes:int=2, loss:str="categorical_crossentropy",
                              weight_decay=1e-4, is_fusion_face:bool=False, is_test:bool=False, freeze_classifier:bool=False,
                              nullfication_metric:bool=False, dropout_const:float=0.25, label_smoothing:float=0., is_eunet:bool=False,
                              norm="batch_norm", eunet_metric_mode="conv1333", eunet_metric_subcontext="default"):
        """
        @機能：距離学習用のモデル構築
        @引数：
        @戻値：モデル
        """
        
        label_input = Input((input_shape[0], input_shape[0], num_classes), name="label_input")
        
        trimed_base_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-2].output)
        
        base_layer = Sequential()
        base_layer.add(trimed_base_model)
        
        ## Dropout
        base_layer.add(Dropout(dropout_const))
        cp.cprint(f"@ Set dropout( {dropout_const} )", "orange")
        
        ## 基本の1x1Conv(Denseの代わり)
        base_layer.add(Conv2D(filters=64,
                                kernel_size=1,
                                activation="relu",
                                kernel_regularizer=regularizers.l2(weight_decay),
                                name="conv_before_face"))
        base_layer.add(norms.decide_norm(norm))
        
        
        ## E-Unet用 ==============================================================
        
        if is_eunet:
            cp.cprint(f"@ attached E-Unet Conv layers", "orange")
            
            
            ## 畳み込み１層のみ
            if (eunet_metric_mode == "conv1"):
                pass
            
            
            ## 単純に層を増やしたもの
            elif (eunet_metric_mode == "conv1333"):
                for i in range(3):
                    base_layer.add(SemanticSegmentation.naive_conv_block(64, 3, f"conv_before_face_{i+1}"))
                    base_layer.add(norms.decide_norm(norm))
                    
            
            ## 多層Denseを畳み込みで代用
            elif (eunet_metric_mode == "conv_dense"):
                for i in range(9):
                    base_layer.add(SemanticSegmentation.naive_conv_block(64, 1, f"conv_before_face_{i+1}"))
                    base_layer.add(Dropout(dropout_const))
                    base_layer.add(norms.decide_norm(norm))
            
            
            ## U-Net Like
            elif (eunet_metric_mode == "u-conv"):
                
                filters = [4, 5, 6]
                layer_mode = "default"
                                
                if (eunet_metric_subcontext != "default"):
                    filters, layer_mode = eunet_metric_subcontext.split("_")
                    filters = list(map(int, filters.split(",")))
                
                conv_filters = [1 << i for i in filters]
                is_bottle = layer_mode == "bottle"
                is_second_conv = layer_mode in ["2", "bottle"]
                uconv_input = Input((input_shape[0], input_shape[0], 64), name="uconv_input")
                xs = []
                
                ## Encoder
                for i, fil in enumerate(conv_filters):
                    x = SemanticSegmentation.naive_conv_block(x if i else uconv_input, fil, 3, f"u-conv_encoder_{i+1}_1")
                    x = norms.decide_norm(norm)(x)
                    if is_second_conv:
                        x = SemanticSegmentation.naive_conv_block(x, fil, 3, f"u-conv_encoder_{i+1}_2")
                        x = norms.decide_norm(norm)(x)
                    x = MaxPooling2D(pool_size=(2, 2), strides=2, name=f"u-conv_encoder_pool_{i+1}")(x)
                    xs = [x] + xs
                
                ## Decoder
                for i, fil in enumerate(conv_filters[::-1]):
                    x = SemanticSegmentation.naive_conv_block(x, fil, 3, f"u-conv_decoder_{i+1}_1")
                    x = norms.decide_norm(norm)(x)
                    if is_second_conv:
                        x = SemanticSegmentation.naive_conv_block(x, fil, 3, f"u-conv_decoder_{i+1}_2")
                        x = norms.decide_norm(norm)(x)
                    x = SemanticSegmentation.naive_deconv_block(x, fil//2, 2, f"u-conv_decoder_trans_{i+1}")
                    x = norms.decide_norm(norm)(x)
                    x = concatenate(SemanticSegmentation.adj_concat(xs[i], x), axis=-1)

                ## Ex-Decoder
                fil = conv_filters[0]//2
                for i in range(2):
                    x = SemanticSegmentation.naive_conv_block(x, fil, 3, f"u-conv_decoder_ex{i+1}", is_bottle=is_bottle)
                    x = norms.decide_norm(norm)(x)

                uconv_model = Model(uconv_input, x)
                uconv_model.summary()
                
                base_layer.add(uconv_model)
                
                base_layer.add(Dropout(dropout_const))


            ## Classifier
            base_layer.add(Conv2D(filters=num_classes,
                                  kernel_size=1,
                                  activation="relu",
                                  kernel_regularizer=regularizers.l2(weight_decay),
                                  name="conv_before_face_final"))
            
        ## =======================================================================

        
        face = metric_semseg.SphereFace
        if (face_name == "arcface"):
            face = metric_semseg.ArcFace
        elif (face_name == "cosface"):
            face = metric_semseg.CosFace
        face_layer = face(num_classes, regularizer=regularizers.l2(weight_decay), nullfication=nullfication_metric, label_smoothing=label_smoothing, name="face_layer")([base_layer.output, label_input])
        
        if nullfication_metric:cp.cprint("@ Metric learning is invalid.", "red")
        
        if is_fusion_face:
            model = Model(inputs=[base_layer.input, label_input], outputs=[face_layer])
            del base_layer
            base_layer = Sequential()
            base_layer.add(model)
        
        # base_model.summary()
        # base_layer.summary()
        
        base_layer.add(base_model.layers[-1])
        # base_layer.layers[0].trainable = False
        base_layer.layers[-1].trainable = not freeze_classifier
        
        if is_fusion_face:
            model = Model(inputs=[base_layer.input, label_input], outputs=[base_layer.output])
        else:
            model = Model(inputs=[base_layer.input, label_input], outputs=[face_layer, base_layer.output])
        
        model.compile(optimizer=Adam(lr=0.001),
                    loss=loss,
                    metrics=["acc", MyLosses.iou_loss, MyLosses.ce_loss_debug],
                    run_eagerly=True)
        
        ## U-Netの畳み込み層を凍結
        for layer in model.layers:
            if (layer.name == "model_1"):
                layer.trainable = is_test
        
        model.summary()
        for layer in model.layers:
            print(layer.name, f": [ {cp.colored('freezed', 'blue') if not layer.trainable else cp.colored('trainable', 'green')} ]")
        print()
        
        return model
    
    
    @staticmethod
    def unet_learning_classifier(base_model, loss:str="categorical_crossentropy", weight_decay=1e-4, num_classes:int=4):
        """
        @機能：距離学習後の、Classifierの学習
        @引数：
        @戻値：モデル
        """
        ## 距離学習の入力層・出力層を切り取る
        model_foot = 0
        for i, layer in enumerate(base_model.layers):
            if (layer.name == "label_input"):
                model_foot = i-1
                print(f"Layer No.{i} :", layer.name, cp.colored("<- Recognised this as label input layer.", "pink"))
            elif ("output" in layer.name) and not model_foot:
                model_foot = i-1
                print(f"Layer No.{i} :", layer.name, cp.colored("<- Recognised this as classifier layer.", "pink"))
            else:
                print(f"Layer No.{i} :", layer.name)
        
        ## U-Netと、距離学習されたConv層のみを取り出す
        trimed_base_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[model_foot].output)
        
        ## 上で取り出した層を凍結する(入力層はそのままにしておく)
        print()
        for i, layer in enumerate(trimed_base_model.layers):
            layer.trainable = bool(not i)
            print(f"base_model layer No.{i} {layer.name} : [ {cp.colored('freezed', 'blue') if i else cp.colored('trainable', 'green')} ]")
        
        ## 出力層の定義
        base_layer = Sequential()
        base_layer.add(trimed_base_model)        
        base_layer.add(base_model.layers[-1])
        
        ## 入力層と出力層を学習可能にする
        base_layer.layers[0].trainable = True
        base_layer.layers[-1].trainable = True
        
        ## 学習するモデルを定義
        model = Model(inputs=[base_layer.input], outputs=[base_layer.output])
        
        model.compile(optimizer=Adam(lr=0.0001),
                    loss=loss,
                    metrics=["acc", MyLosses.iou_loss, MyLosses.ce_loss_debug],
                    run_eagerly=True)
        
        ## 入力層と出力層以外を(念の為)凍結
        for layer in model.layers[1:-1]:
            layer.trainable = False
        
        model.summary()
        for layer in model.layers:
            print(layer.name, f": [ {cp.colored('freezed', 'blue') if not layer.trainable else cp.colored('trainable', 'green')} ]")
        print()
        
        return model
    
    