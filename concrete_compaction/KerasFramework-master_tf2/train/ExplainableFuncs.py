from code import interact
import cv2
import numpy as np
from tensorflow.keras.backend import gradients, function
from tensorflow.keras.models import Model
import gc
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
tfv1.disable_eager_execution()
# tfv1.enable_eager_execution()

# my module
from colorPrint import Cprint as cp
from MyUtils import Calc, TimeCounter


## ================ config ================


CLASSES = ["before", "just"]
NUM_CLASSES = 2


## ========================================


class ExplainableFuncs:
    
    def __init__(self):
        pass


    @staticmethod
    def Seg_Grad_CAM(model, image, output_layer:str, hidden_layers:list, decode_size=(540, 540), is_realtime_preview=False, preview_size=(270*6, 270*4)):
        """
        @機能：Grad_CAMをセグメンテーションに適用させた版
        @引数：model = モデル, image = 画像データ(numpy配列), output_layer = 対象とする層の定義名, hidden_layers:list = Grad-CAMの計算の対象とする層の名前, decode_size = 出力する画像サイズ, 空要素, 空要素 
        @戻値：
        """

        ## 前処理
        img = np.expand_dims(image, axis=0)
        img = img.astype('float32')

        ## 推論
        prediction ,= model.predict(img)
        ## 予測クラスの算出
        pred_label = np.argmax(prediction, axis=2)
        
        ## 対象とする出力層の出力を抽出
        integraled_map = model.get_layer(output_layer).output
        
        grad_cam_images = [None]*len(hidden_layers)
        grad_cam_maps = [[None]*2 for _ in range(len(hidden_layers))]
        mixed_images = [None]*len(hidden_layers)
        print("\n")
        
        ## 開始時刻の保存
        timecounter = TimeCounter(len(hidden_layers))
        
        ## 指定された層の数だけ実行
        for i, layer in enumerate(hidden_layers):
            
            ## 元画像の読み込み
            # grad_cam_images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/2.
            # grad_cam_images[i] = np.copy(image)*255./2.
            grad_cam_images[i] = np.zeros((*decode_size, 3))
            
            ## 対象とする層の出力を抽出
            conv_map = model.get_layer(layer).output
            
            ## クラスの数だけ実行
            for class_index in range(NUM_CLASSES):
                
                ## class_index番目のクラスに属する領域のマッピング
                mask_map = np.round(prediction[:, :, class_index] * (pred_label == class_index)).reshape(1, prediction.shape[0], prediction.shape[1], 1)
                mask_map = tf.constant(mask_map)
                ## 出力層における、class_index番目のクラスに属している部分を抽出
                masked_map = tf.multiply(integraled_map, mask_map)
                
                ## 勾配を取得
                grads ,= gradients(masked_map, conv_map)
                
                ## ここまでのテンソル演算を関数化
                gradient_function = function([model.input], [conv_map, grads])
                
                ## 画像の入力
                output, grads_val = gradient_function([img])
                
                #### Grad-CAMを計算
                ## 特徴マップと勾配の取り出し
                output, grads_val = output[0], grads_val[0]
                ## 平均を計算
                grads_val = np.mean(grads_val, axis=(0, 1))
                ## 乗算
                grad_cam = np.dot(output, grads_val)
                ## ReLUの適用
                grad_cam = np.maximum(grad_cam, 0)
                ## (0, 1)に正規化
                if (np.max(grad_cam) <= 0):
                    grad_cam *= 0
                else:
                    grad_cam /= grad_cam.max()
                ####
                
                grad_cam_maps[i][class_index] = np.copy(grad_cam)
                
                grad_cam = cv2.resize(grad_cam*255, decode_size)
                grad_cam_images[i][:, :, 2*(not class_index%2)] += grad_cam
                
                del grads, grad_cam, gradient_function, output, grads_val, mask_map
            
            grad_cam_images[i] = np.clip(grad_cam_images[i], 0, 255)
            
            mixed_images[i] = np.copy(grad_cam_images[i])
            mixed_images[i][:, :, 1] += pred_label*255./8.
            mixed_images[i] = np.clip(mixed_images[i], 0, 255)
            
            ## リアルタイムで可視化
            if is_realtime_preview:
                image_buf = cv2.resize(mixed_images[i]/255., preview_size)
                image_buf += cv2.resize(image/2., preview_size)
                cv2.imshow("[q] : Quit", image_buf)
                key = cv2.waitKey(1) & 0xff
                ## "q" : プログラム終了
                if (key == ord("q")):
                    cv2.destroyAllWindows()
                    return
            
            ## 終了時刻の予測
            remining_time = timecounter.predictTime(i+1)
            
            cp.cprint(f"\033[1Acompleted : {i+1} / {len(hidden_layers)} - {remining_time}   ", "cyan")
                        
        gc.collect()
        
        return grad_cam_maps, grad_cam_images, mixed_images, prediction
                
                
    
    @staticmethod
    def Grad_CAM(model, image, output_layer, buf, decode_size=(540,540), is_binary=True, plot_multi:float=2.):
        """
        @機能：Grad-CAM
        @引数：model = モデル, image = 画像データ(numpy配列), output_layer = 対象とする層の定義名, 空要素, decode_size = 出力する画像サイズ, is_binary = ２クラス分類の出力部分を対象とするか否か, plot_multi = 色の強調表現の指数
        @戻値：!誤り(GradCAMの値, RGBに変換した値, 入力画像に加算した値)
        """

        return
        
        ## 前処理
        img = np.expand_dims(image, axis=0)
        img = img.astype('float32')

        ## 予測クラスの算出
        prediction ,= model.predict(img)
        if is_binary:
            ## Before, Justをそれぞれ別々に取り出す
            class_indexes = [0, 1]
        else:
            ## 特徴マップより、チャネルごとに最大の値を取り出す
            class_indexes = np.array(np.unique(np.argmax(prediction, axis=2)))
        classes = [len(class_indexes)]
        layer_shape = [model.get_layer(output_layer).output.shape[1], model.get_layer(output_layer).output.shape[2]]
        feature_map = np.array([[[0]*layer_shape[1] for _ in range(layer_shape[0])] for __ in range(classes[0])], dtype=np.float64)
                
        for i, index in enumerate(class_indexes):
            class_output = model.output[:, :, :, index]
        
            ## 勾配を取得
            conv_output = model.get_layer(output_layer).output   # layer_nameのレイヤーのアウトプット
            grads = gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
            gradient_function = function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
            
            output, grads_val = gradient_function([img])
            output, grads_val = output[0], grads_val[0]
            
            ## 重みを平均化して、レイヤーのアウトプットに乗じる
            weights = np.mean(grads_val, axis=(0, 1))
            feature_map[i] = np.dot(output, weights)
            
            del conv_output, grads, gradient_function, output, grads_val
                            
        ## 平均を求める
        # feature_map /= pixels_accum
        
        gradcam_image = [None]*classes[0]
        added_image = [None]*classes[0]
        feature_map_buf = np.copy(feature_map)
        
        ## ReLU
        for i in range(classes[0]):
            ## 全て負ならば全て0
            if (np.max(feature_map[i]) < 0):
                feature_map[i] *= 0
                
            ## それ以外なら普通にReLU
            else:
                ## ReLU
                feature_map = np.maximum(feature_map, 0)
                ## (0, 1)に正規化
                feature_map /= np.max(feature_map)
        
        if is_binary:
            ## Before
            gradcam_buf = (np.maximum(feature_map[0], 0.5)-0.5)*2.
            ## Just
            in_gradcam_buf = (np.maximum(feature_map[1], 0.5)-0.5)*2.
        
        else:
            gradcam_buf = np.mean(feature_map, axis=0)
            in_gradcam_buf = 0
        
        ## グレースケールかつ指定サイズの画像に変換
        gradcam = cv2.resize(gradcam_buf**plot_multi, decode_size, cv2.INTER_LINEAR)
        in_gradcam = cv2.resize(in_gradcam_buf**plot_multi, decode_size, cv2.INTER_LINEAR)
        
        ## (0, 1)に正規化
        if (gradcam.max() > 1):
            gradcam /= 255.
            in_gradcam /= 255.
        
        added = np.copy(image)/2.
        ## 尤度により色分け
        ## 青色
        added[:, :, 0] += np.float32(in_gradcam)*2
        ## 赤色
        added[:, :, 2] += np.float32(gradcam)*2
        added = np.clip(added, 0, 1)
        
        ## 出力する配列
        gradcam_image = np.copy(gradcam)
        added_image = np.copy(added)
        
        gradcam_image *= 255.
        added_image *= 255.

        gc.collect()
        
        return classes, feature_map, gradcam_image, added_image, prediction, feature_map_buf
