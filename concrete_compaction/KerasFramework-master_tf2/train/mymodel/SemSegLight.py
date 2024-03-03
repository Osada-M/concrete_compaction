import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, PReLU, Multiply
from tensorflow.keras.layers import MaxPooling2D, Permute, ZeroPadding2D, add
# from tensorflow.keras.optimizers import Adam
import numpy as np
import os

## my modelu
from Optimizer import Optimizer
from colorPrint import Cprint as cp
import pruning_layer as PL


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


class E_UNet:
    
    is_PL = False
    
    @staticmethod
    def initial_block(inp, number_filter=13, filter_size=(3, 3), stride=(2, 2), reduce_const:float=1):
        
        number_filter = int(np.ceil((number_filter + 3) * reduce_const)) - 3
        
        conv = Conv2D(number_filter, filter_size, padding="same", strides=stride)(inp) 
        max_pool = MaxPooling2D()(inp)
        merged = concatenate([conv, max_pool], axis=3)
        batch = BatchNormalization(momentum=0.1)(merged)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
        if E_UNet.is_PL:
            output = PL.PL_PReLU(shared_axes=[1, 2])(batch)
        else:
            output = PReLU(shared_axes=[1, 2])(batch)
            
        return output


    @staticmethod
    def bottleneck(inp, number_filter=32, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.01):
        
        # main branch
        internal = number_filter // internal_scale
        encoder = inp

        # 1x1
        input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
        encoder = Conv2D(internal, (input_stride, input_stride), (input_stride, input_stride) , use_bias=False) (encoder)

        # Batch normalization + PReLU
        encoder = BatchNormalization(momentum=0.1) (encoder)
        if E_UNet.is_PL:
            encoder = PL.PL_PReLU(shared_axes=[1, 2])(encoder)
        else:
            encoder = PReLU(shared_axes=[1, 2])(encoder)
        
        #con
        if not asymmetric and not dilated:
            encoder =  Conv2D(internal, (3, 3), padding="same" ) (encoder)
        elif asymmetric:
            encoder = Conv2D(internal, (1, asymmetric), use_bias=False, padding="same") (encoder)
            encoder = Conv2D(internal, (asymmetric, 1), padding="same") (encoder)
        elif dilated:
            encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding="same") (encoder)
        
        # Batch normalization + PReLU
        encoder = BatchNormalization(momentum=0.1)(encoder)
        if E_UNet.is_PL:
            encoder = PL.PL_PReLU(shared_axes=[1, 2])(encoder)
        else:
            encoder = PReLU(shared_axes=[1, 2])(encoder)
    
        # 1x1
        encoder = Conv2D(number_filter, (1, 1), use_bias=False)(encoder)
        encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
        encoder = SpatialDropout2D(dropout_rate)(encoder)
        
        other = inp
        if downsample:
            other = MaxPooling2D()(other)

            other = Permute((1, 3, 2))(other)

            pad_feature_maps = int(number_filter - inp.get_shape().as_list()[3])
            tb_pad = (0, 0)
            lr_pad = (0, pad_feature_maps)
            other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
            other = Permute((1, 3, 2))(other)
        
        encoder = add([encoder, other])
        if E_UNet.is_PL:
            encoder = PL.PL_PReLU(shared_axes=[1, 2])(encoder)
        else:
            encoder = PReLU(shared_axes=[1, 2])(encoder)
        
        return encoder


    @staticmethod
    def en_build(inp, dropout_rate=0.01, reduce_const:float=1):
        
        cp.cprint(f"@ Set reduce_const is {reduce_const}.", "orange")
        enc_layer1 = E_UNet.initial_block(inp, reduce_const=reduce_const)
        
        enc_layer2 = E_UNet.bottleneck(enc_layer1, int(np.ceil(32 * reduce_const)), downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
        for _ in range(3):
            enc_layer2 = E_UNet.bottleneck(enc_layer2, int(np.ceil(32 * reduce_const)), dropout_rate=dropout_rate)  # bottleneck 1.i
        
        enc_layer3 = E_UNet.bottleneck(enc_layer2, int(np.ceil(64 * reduce_const)), downsample=True)  # bottleneck 2.0
        for _ in range(2):
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)))  # bottleneck 2.1
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), dilated=2)  # bottleneck 2.2
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), asymmetric=5)  # bottleneck 2.3
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), dilated=4)  # bottleneck 2.4
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)))  # bottleneck 2.5
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), dilated=8)  # bottleneck 2.6
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), asymmetric=5)  # bottleneck 2.7
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)), dilated=16)  # bottleneck 2.8
            enc_layer3 = E_UNet.bottleneck(enc_layer3, int(np.ceil(64 * reduce_const)))  # bottleneck 2.9

        enc_layer4 = E_UNet.bottleneck(enc_layer3, int(np.ceil(128 * reduce_const)), downsample=True)  # bottleneck 2.0
        for _ in range(1):
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)))  # bottleneck 3.1
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), dilated=2)  # bottleneck 3.2
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), asymmetric=5)  # bottleneck 3.3
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), dilated=4)  # bottleneck 3.4
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)))  # bottleneck 3.5
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), dilated=8)  # bottleneck 3.6
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), asymmetric=5)  # bottleneck 3.7
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)), dilated=16)  # bottleneck 3.8
            enc_layer4 = E_UNet.bottleneck(enc_layer4, int(np.ceil(128 * reduce_const)))  # bottleneck 3.5

        return enc_layer1, enc_layer2, enc_layer3, enc_layer4#, enc_layer4
        

    @staticmethod
    def de_bottleneck(encoder, decoder, number_filter, upsample=False, reverse_module=False, last=False, dropout_const=0.01, num_classes=2, use_attention=False, attention_filters=1, multi_losses=False, autoencoder=False):
        
        if decoder is not None:
            if use_attention:
                ## Self-Attention(模索中)
                att_encoder = Conv2D(filters=64*attention_filters, kernel_size=(1, 1), strides=(1, 1), padding="valid")(encoder)
                att_encoder = BatchNormalization()(att_encoder)
                
                att_decoder = Conv2D(filters=64*attention_filters, kernel_size=(1, 1), strides=(1, 1), padding="valid")(decoder)
                att_decoder = BatchNormalization()(att_decoder)
                
                att_layer = add([att_encoder, att_decoder])
                att_layer = Activation("relu")(att_layer)
                att_layer = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(att_layer)
                att_layer = BatchNormalization()(att_layer)
                att_layer = Activation("sigmoid")(att_layer)
                encoder = Multiply()([decoder, att_layer])
                
            else:
                encoder = add([encoder, decoder])
            
        if  (number_filter // 4  == 0):
            internal = number_filter
        else:
            internal = number_filter // 4
        
        x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
        x = BatchNormalization(momentum=0.1)(x)
        x = Activation('relu')(x)
        if not upsample:
            x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
        else:
            x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization(momentum=0.1)(x)
        x = Activation('relu')(x)

        x = Conv2D(number_filter, (1, 1), padding='same', use_bias=False)(x)

        other = encoder
        if encoder.get_shape()[-1] != number_filter or upsample:
            other = Conv2D(number_filter, (1, 1), padding='same', use_bias=False)(other)
            other = BatchNormalization(momentum=0.1)(other)
            if upsample and reverse_module is not False:
                other = UpSampling2D(size=(2, 2))(other)

        if upsample and reverse_module is False:
            decoder = x
        elif last:
            x = BatchNormalization(momentum=0.1)(x)
            decoder = add([x, other])
            # decoder = Activation('sigmoid')(decoder)
            decoder = Dropout(dropout_const)(decoder)
            
            activation = "sigmoid" if autoencoder else "softmax"
            
            ## 複数の損失を出すパターン
            if multi_losses:
                decoder_4cls = Conv2D(filters=4, 
                                    kernel_size=1, 
                                    activation=activation, 
                                    name="classifier_4cls")(decoder)
                decoder_2cls = Conv2D(filters=2, 
                                    kernel_size=1, 
                                    activation=activation, 
                                    name="classifier")(decoder)
                return decoder_2cls, decoder_4cls
            else:
                decoder = Conv2D(filters=num_classes, 
                                kernel_size=1, 
                                activation=activation, 
                                name="classifier")(decoder)
        else:
            x = BatchNormalization(momentum=0.1)(x)
            decoder = add([x, other])
            decoder = Activation('relu')(decoder)
        
        return decoder


    @staticmethod
    def de_build(enc_layer4, enc_layer3, enc_layer2, enc_layer1, num_classes=3, dropout_const=0.01, use_attention=False, multi_losses=False, autoencoder=False, skip_connection:int=3, reduce_const:int=1):
        
        dec_1 = E_UNet.de_bottleneck(enc_layer4, None, int(np.ceil(64 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=8)
        dec_1_1 = E_UNet.de_bottleneck(dec_1, None, int(np.ceil(64 * reduce_const)))  # bottleneck 1.1
        dec_1_2 = E_UNet.de_bottleneck(dec_1_1, None, int(np.ceil(64 * reduce_const)))  # bottleneck 1.2
        
        dec_2 = E_UNet.de_bottleneck(dec_1_2, enc_layer3, int(np.ceil(32 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=4)

        # if (skip_connection > 0):
        #     dec_2 = E_UNet.de_bottleneck(dec_1_2, enc_layer3, int(np.ceil(32 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=4)
        # else:
        #     dec_2 = E_UNet.de_bottleneck(dec_1_2, None, int(np.ceil(32 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=4)
        #     cp.cprint(f"@ Skip Connection No.1 : Disenable", "red")
        dec_2_1 = E_UNet.de_bottleneck(dec_2, None, int(np.ceil(32 * reduce_const)))  # bottleneck 2.1
        dec_2_2 = E_UNet.de_bottleneck(dec_2_1, None, int(np.ceil(32 * reduce_const)))  # bottleneck 2.2
        dec_2_3 = E_UNet.de_bottleneck(dec_2_2, None, int(np.ceil(32 * reduce_const)))  # bottleneck 23
        dec_2_4 = E_UNet.de_bottleneck(dec_2_3, None, int(np.ceil(32 * reduce_const)))  # bottleneck 2.4

        # if (skip_connection > 1):
        dec_3 = E_UNet.de_bottleneck(dec_2_4, enc_layer2, int(np.ceil(16 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=2)
        # else:
            # dec_3 = E_UNet.de_bottleneck(dec_2_4, None, int(np.ceil(16 * reduce_const)), upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=2)
            # cp.cprint(f"@ Skip Connection No.2 : Disenable", "red")
        dec_3_1 = E_UNet.de_bottleneck(dec_3, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.1.
        dec_3_2 = E_UNet.de_bottleneck(dec_3_1, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.2
        dec_3_3 = E_UNet.de_bottleneck(dec_3_2, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.3
        dec_3_4 = E_UNet.de_bottleneck(dec_3_3, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.4
        dec_3_5 = E_UNet.de_bottleneck(dec_3_4, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.5
        dec_3_6 = E_UNet.de_bottleneck(dec_3_5, None, int(np.ceil(16 * reduce_const)))  # bottleneck 3.6

        # if (skip_connection > 2):
        dec_4 = E_UNet.de_bottleneck(dec_3_6, enc_layer1, num_classes, upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=1)
        # else:
            # dec_4 = E_UNet.de_bottleneck(dec_3_6, None, num_classes, upsample=True, reverse_module=True, use_attention=use_attention, attention_filters=1)
            # cp.cprint(f"@ Skip Connection No.3 : Disenable", "red")
        dec_4_1 = E_UNet.de_bottleneck(dec_4, None, num_classes)# bottleneck 5.1 
        dec_4_2 = E_UNet.de_bottleneck(dec_4_1, None, num_classes)# bottleneck 5.2 
        dec_4_3 = E_UNet.de_bottleneck(dec_4_2, None, num_classes)# bottleneck 5.3     
        dec_4_4 = E_UNet.de_bottleneck(dec_4_3, None, num_classes)# bottleneck 5.4 
        dec_4_5 = E_UNet.de_bottleneck(dec_4_4, None, num_classes)# bottleneck 5.5 
        dec_4_6 = E_UNet.de_bottleneck(dec_4_5, None, num_classes)# bottleneck 5.6 
        dec_4_7 = E_UNet.de_bottleneck(dec_4_6, None, num_classes)# bottleneck 5.7 
        dec_4_8 = E_UNet.de_bottleneck(dec_4_7, None, num_classes, last=True, dropout_const=dropout_const, num_classes=num_classes, multi_losses=multi_losses, autoencoder=autoencoder)# bottleneck 5.8 
    
        return dec_4_8

    @staticmethod
    def run(input_shape=(256, 256, 3), num_classes=2, dropout_const:float=0.1, is_compile:bool=True, loss="categorical_crossentropy", use_attention:bool=False, optimizer:str="adam", multi_losses:bool=False, autoencoder:bool=False, skip_connection:int=3, reduce_const:int=1, learning_rate:float=1e-3):
        input_height, input_width, dim = input_shape
        assert input_height % 32 == 0
        assert input_width % 32 == 0
        
        img_input = Input(shape=(input_height, input_width, dim))
        enc_layer1, enc_layer2, enc_layer3, enc_layer4 = E_UNet.en_build(img_input, dropout_rate=dropout_const, reduce_const=reduce_const)
        output = E_UNet.de_build(enc_layer4, enc_layer3, enc_layer2, enc_layer1, num_classes=num_classes, dropout_const=dropout_const, use_attention=use_attention, multi_losses=multi_losses, autoencoder=autoencoder, skip_connection=skip_connection, reduce_const=reduce_const)
        
        # output = Dropout(dropout_const)(output)
        # output = Conv2D(filters=num_classes, 
        #                 kernel_size=1, 
        #                 activation="softmax", 
        #                 name="classifier")(output)
        
        model = Model(img_input, output)
        
        ## 最適化関数の振り分け
        my_optimizer_func = Optimizer.decide_optimizer(optimizer)
        cp.cprint(f"@ set loss function : {loss}", "orange")
        cp.cprint(f"@ set optimizer : {optimizer}", "orange")
        
        if is_compile:
            model.compile(optimizer=my_optimizer_func(learning_rate=learning_rate), 
                         loss=loss, 
                        #  metrics=["acc", MyLosses.iou_loss, MyLosses.ce_loss_debug],
                         metrics=["acc"],
                         run_eagerly=True)

        return model


class ESPNet:
    
    @staticmethod
    def conv_layer(ip, number_of_filters, kernel, stride, layer_name="conv"):
        """
        Convolution layer
        Args:
            ip: Input
            number_of_filters: Number of filters
            kernel: Kernel Size
            stride: Stride for down-sampling
            layer_name:optional parameter to specify layer name
            """
        # with tf.name_scope(layer_name):
        network = Conv2D(use_bias=False, filters=number_of_filters, kernel_size=kernel,
                                strides=stride, padding="SAME",
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))(ip)
        return network


    @staticmethod
    def dilation_conv_layer(ip, number_of_filters, kernel, stride, dilation_rate, layer_name="conv"):
        """
        Dilation Convolution Layer
        Args:
            ip: Input
            number_of_filters: Number of filters
            kernel: Kernel Size
            stride: Stride for down-sampling
            dilation_rate: dilation rate
            layer_name:optional parameter to specify layer name
        """
        # with tf.name_scope(layer_name):
        network = Conv2D(use_bias=False, filters=number_of_filters, kernel_size=kernel,
                                strides=stride, padding="SAME", dilation_rate=dilation_rate,
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))(ip)
        return network


    # @staticmethod
    # def prelu(x, scope=None):
    #     with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
    #         alpha = tf.get_variable("prelu", shape=x.get_shape()[-1],
    #                                 dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    #     return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


    @staticmethod
    def BN_PRelu(out):
        """
        Does Batch Normalization followed by PReLU.
        """
        batch_conv = BatchNormalization()(out)
        # prelu_batch_norm = ESPNet.prelu(batch_conv)
        prelu_batch_norm = PReLU(shared_axes=[1, 2])(batch_conv)
        return prelu_batch_norm


    @staticmethod
    def conv_one_cross_one(ip, number_of_classes, layer_name="FINAL"):
        """
        Performs 1X1 concolution to project high-dimensional feature maps onto a low-dimensional space.
        """
        # with tf.name_scope(layer_name):
        network = Conv2D(use_bias=False, filters=number_of_classes, kernel_size=[1, 1],
                                strides=1, padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))(ip)
        return network


    @staticmethod
    def esp(ip, n_out):
        """
        ESP module based  on  principle of:-
        Reduce -> Split -> Transform -> Merge
        Args:
            ip: Input
            n_out: number of output channels
        """
        number_of_branches = 5
        n = int(n_out / number_of_branches)
        n1 = n_out - (number_of_branches - 1) * n

        # Reduce
        output1 = ESPNet.conv_layer(ip, number_of_filters=n, kernel=[3, 3], stride=1)

        # Split and Transform
        dilated_conv1 = ESPNet.dilation_conv_layer(output1, number_of_filters=n1, kernel=[3, 3], stride=1, dilation_rate=(1, 1))
        dilated_conv2 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(2, 2))
        dilated_conv4 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(4, 4))
        dilated_conv8 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(8, 8))
        d16 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(16, 16))
        add1 = dilated_conv2
        add2 = add([add1, dilated_conv4])
        add3 = add([add2, dilated_conv8])
        add4 = add([add3, d16])

        # Merge
        concat = tf.concat(
            (dilated_conv1, add1, add2, add3, add4),
            axis=-1)
        concat = ESPNet.BN_PRelu(concat)
        return concat


    @staticmethod
    def esp_alpha(ip, n_out):
        """
        ESP-alpha module where alpha controls depth of network.
        Args:
            ip: Input
            n_out: number of output channels
        """
        number_of_branches = 5
        n = int(n_out / number_of_branches)
        n1 = n_out - (number_of_branches - 1) * n
        
        output1 = ESPNet.conv_layer(ip, number_of_filters=n, kernel=[1, 1], stride=1)
        dilated_conv1 = ESPNet.dilation_conv_layer(output1, number_of_filters=n1, kernel=[3, 3], stride=1, dilation_rate=(1, 1))
        dilated_conv2 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(2, 2))
        dilated_conv4 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(4, 4))
        dilated_conv8 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(8, 8))
        dilated_conv16 = ESPNet.dilation_conv_layer(output1, number_of_filters=n, kernel=[3, 3], stride=1, dilation_rate=(16, 16))
        add1 = dilated_conv2
        add2 = add([add1, dilated_conv4])
        add3 = add([add2, dilated_conv8])
        add4 = add([add3, dilated_conv16])
        concat = tf.concat(
            (dilated_conv1, add1, add2, add3, add4),
            axis=-1)
        concat = ESPNet.BN_PRelu(concat)
        return concat


    @staticmethod
    def espnet(x, alpha1=2, alpha2=8, num_classes=4):
        """
        ESPNet model architecture
        Args:
            x: Input
            model: Name of the model architecture which either "espnet_c" or "espnet"
        """
        conv_output = ESPNet.conv_layer(x, number_of_filters=16, kernel=[3, 3], stride=1, layer_name="first_layer_conv1")
        prelu_ = ESPNet.BN_PRelu(conv_output)
        avg_pooling = AveragePooling2D(3, 1, padding='same', data_format='channels_last', name=None)(x)
        concat1 = tf.concat((avg_pooling, prelu_), axis=-1, name='concat_avg_prerelu')
        concat1 = ESPNet.BN_PRelu(concat1)
        esp_1 = ESPNet.esp(concat1, 64)
        esp_1 = ESPNet.BN_PRelu(esp_1)
        esp_alpha_1 = esp_1
        # alpha1 = 2
        # alpha2 = 8
        
        for i in range(alpha1):
            esp_alpha_1 = ESPNet.esp_alpha(esp_alpha_1, 64)
        concat2 = tf.concat((esp_alpha_1, esp_1, avg_pooling), axis=-1)
        
        esp_2 = ESPNet.esp(concat2, 128)
        esp_alpha_2 = esp_2
        
        for i in range(alpha2):
            esp_alpha_2 = ESPNet.esp_alpha(esp_alpha_2, 128)
        concat3 = tf.concat((esp_alpha_2, esp_2), axis=-1)
        
        pred = ESPNet.conv_one_cross_one(concat3, num_classes)

        deconv1 = Conv2DTranspose(num_classes, [2, 2], strides=(1, 1), padding='same')(pred)
        conv_1 = ESPNet.conv_one_cross_one(concat2, num_classes)
        concat4 = tf.concat((deconv1, conv_1), axis=-1)
        # esp_3 = ESPNet.esp(concat4, num_classes)
        esp_3 = ESPNet.esp_alpha(concat4, num_classes*4)
        deconv2 = Conv2DTranspose(num_classes, [2, 2], strides=(1, 1), padding='same')(esp_3)
        conv_2 = ESPNet.conv_one_cross_one(concat1, num_classes)
        concat5 = tf.concat((deconv2, conv_2), axis=-1)
        conv_3 = ESPNet.conv_one_cross_one(concat5, num_classes)
        deconv3 = Conv2DTranspose(num_classes, [2, 2], strides=(1, 1), padding='same')(conv_3)
        return deconv3
    
    
    @staticmethod
    def run(input_shape=(576, 576, 3), num_classes=4, alphas=[2, 8], dropout_const:float=0.01, is_compile:bool=True, loss="categorical_crossentropy", optimizer:str="adam", autoencoder:bool=False, learning_rate:float=1e-3):
        
        inp = Input(input_shape)
        esp = ESPNet.espnet(inp, *alphas, num_classes)

        # x = BatchNormalization(momentum=0.1)(esp)

        # x = Dropout(dropout_const)(x)
        
        activation = "sigmoid" if autoencoder else "softmax"
        
        x = Conv2D(filters=num_classes, 
                        kernel_size=1, 
                        activation=activation, 
                        name="classifier")(esp)
        
        model = Model(inp, x)
        
        ## 最適化関数の振り分け
        my_optimizer_func = Optimizer.decide_optimizer(optimizer)
        cp.cprint(f"@ set loss function : {loss}", "orange")
        cp.cprint(f"@ set optimizer : {optimizer}", "orange")
        
        if is_compile:
            model.compile(optimizer=my_optimizer_func(learning_rate=learning_rate), 
                         loss=loss, 
                        #  metrics=["acc", MyLosses.iou_loss, MyLosses.ce_loss_debug],
                         metrics=["acc"],
                         run_eagerly=True)
        
        return model


class CFPNetM:
        
    @staticmethod
    def conv2d_bn(x, filters, ksize, d_rate, strides,padding='same', activation='relu', groups=1, name=None):
        '''
        2D Convolutional layers
        
        Arguments:
            x {keras layer} -- input layer 
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters
        
        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})
        
        Returns:
            [keras layer] -- [output layer]
        '''

        x = Conv2D(filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate = d_rate, use_bias=False)(x)
        x = BatchNormalization(axis=3, scale=False)(x)

        if activation is None:
            return x

        x = Activation(activation, name=name)(x)

        return x


    @staticmethod
    def CFPModule(inp, filters, d_size):
        '''
        CFP module for medicine
        
        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer 
        
        Returns:
            [keras layer] -- [output layer]
        '''
        x_inp = CFPNetM.conv2d_bn(inp, filters//4, ksize=1, d_rate=1, strides=1)
        
        x_1_1 = CFPNetM.conv2d_bn(x_inp, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
        x_1_2 = CFPNetM.conv2d_bn(x_1_1, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
        x_1_3 = CFPNetM.conv2d_bn(x_1_2, filters//8, ksize=3, d_rate=1, strides=1,groups=filters//8)
        
        x_2_1 = CFPNetM.conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
        x_2_2 = CFPNetM.conv2d_bn(x_2_1, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
        x_2_3 = CFPNetM.conv2d_bn(x_2_2, filters//8, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//8)

        x_3_1 = CFPNetM.conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
        x_3_2 = CFPNetM.conv2d_bn(x_3_1, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
        x_3_3 = CFPNetM.conv2d_bn(x_3_2, filters//8, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//8)
        
        x_4_1 = CFPNetM.conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
        x_4_2 = CFPNetM.conv2d_bn(x_4_1, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
        x_4_3 = CFPNetM.conv2d_bn(x_4_2, filters//8, ksize=3, d_rate=d_size+1, strides=1, groups=filters//8)
        
        o_1 = concatenate([x_1_1,x_1_2,x_1_3], axis=3)
        o_2 = concatenate([x_2_1,x_2_2,x_2_3], axis=3)
        o_3 = concatenate([x_1_1,x_3_2,x_3_3], axis=3)
        o_4 = concatenate([x_1_1,x_4_2,x_4_3], axis=3)
        
        o_1 = BatchNormalization(axis=3)(o_1)
        o_2 = BatchNormalization(axis=3)(o_2)
        o_3 = BatchNormalization(axis=3)(o_3)
        o_4 = BatchNormalization(axis=3)(o_4)
        
        ad1 = o_1
        ad2 = add([ad1,o_2])
        ad3 = add([ad2,o_3])
        ad4 = add([ad3,o_4])
        output = concatenate([ad1,ad2,ad3,ad4],axis=3)
        output = BatchNormalization(axis=3)(output)
        output = CFPNetM.conv2d_bn(output, filters, ksize=1, d_rate=1, strides=1,padding='valid')
        output = add([output, inp])

        return output

    
    @staticmethod
    def CFPNetM(height, width, channels, num_classes=2, dropout_const=0.1, autoencoder=False):

        inputs = Input(shape=(height, width, channels))
        
        conv1=CFPNetM.conv2d_bn(inputs, 32, 3, 1, 2)
        conv2 = CFPNetM.conv2d_bn(conv1, 32, 3, 1, 1)
        conv3 = CFPNetM.conv2d_bn(conv2, 32, 3, 1, 1)
        
        injection_1 = AveragePooling2D()(inputs)
        injection_1 = BatchNormalization(axis=3)(injection_1)
        injection_1 = Activation('relu')(injection_1)
        opt_cat_1 = concatenate([conv3,injection_1], axis=3)
        
        #CFP block 1
        opt_cat_1_0 = CFPNetM.conv2d_bn(opt_cat_1, 64, 3, 1, 2)
        cfp_1 = CFPNetM.CFPModule(opt_cat_1_0, 64, 2)
        cfp_2 = CFPNetM.CFPModule(cfp_1, 64, 2)
        
        injection_2 = AveragePooling2D()(injection_1)
        injection_2 = BatchNormalization(axis=3)(injection_2)
        injection_2 = Activation('relu')(injection_2)
        opt_cat_2 = concatenate([cfp_2,opt_cat_1_0,injection_2], axis=3)
        
        #CFP block 2
        opt_cat_2_0 = CFPNetM.conv2d_bn(opt_cat_2, 128, 3, 1, 2)
        cfp_3 = CFPNetM.CFPModule(opt_cat_2_0, 128, 4)
        cfp_4 = CFPNetM.CFPModule(cfp_3, 128, 4)
        cfp_5 = CFPNetM.CFPModule(cfp_4, 128, 8)
        cfp_6 = CFPNetM.CFPModule(cfp_5, 128, 8)
        cfp_7 = CFPNetM.CFPModule(cfp_6, 128, 16)
        cfp_8 = CFPNetM.CFPModule(cfp_7, 128, 16)
        
        injection_3 = AveragePooling2D()(injection_2)
        injection_3 = BatchNormalization(axis=3)(injection_3)
        injection_3 = Activation('relu')(injection_3)
        opt_cat_3 = concatenate([cfp_8,opt_cat_2_0,injection_3], axis=3)
        
        
        conv4 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same',activation='relu')(opt_cat_3)
        up_1 = concatenate([conv4,opt_cat_2])    
        conv5 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same',activation='relu')(up_1)
        up_2 = concatenate([conv5, opt_cat_1],axis=3)        
        conv6 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same',activation='relu')(up_2)   
        
        if autoencoder: 
            conv7 = CFPNetM.conv2d_bn(conv6, 3, 1, 1, 1, activation='sigmoid', padding='valid')
        else:
            conv7 = CFPNetM.conv2d_bn(conv6, num_classes, 1, 1, 1, activation='softmax', padding='valid')
        
        model = Model(inputs=inputs, outputs=conv7)
        
        return model
    
    
    @staticmethod
    def run(input_shape=(256, 256, 3), num_classes=2, dropout_const:float=0.1, is_compile:bool=True, loss="categorical_crossentropy", optimizer:str="adam", autoencoder:bool=False):
        
        height, width, channels = input_shape
        model = CFPNetM.CFPNetM(height, width, channels, num_classes)
        
        my_optimizer_func = Optimizer.decide_optimizer(optimizer)
        cp.cprint(f"@ set loss function : {loss}", "orange")
        cp.cprint(f"@ set optimizer : {optimizer}", "orange")
        
        if is_compile:
            model.compile(optimizer=my_optimizer_func(learning_rate=0.001), 
                         loss=loss, 
                         metrics=["acc"],
                         run_eagerly=True)

        return model
        
