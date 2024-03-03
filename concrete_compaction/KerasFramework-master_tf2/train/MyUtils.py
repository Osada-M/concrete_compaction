import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import random
import time
import os
from PIL import Image
import functools
import pickle
from skimage import feature

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from mymodel.SemSegLight import E_UNet, ESPNet
from colorPrint import Cprint as cp
from luminance_extender import LuminanceExtender
from my_loss_function import MyLosses
from affine_transform import AffineTransform as AT
from convert_model import After
from MyPruning import MyPruning


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
random.seed(time.time())


class Utils:
    """
    @機能：便利な機能たち
    @関数：makedir(), datacounter(), log(), ignore_unhashable(), print_dict()
    """
    @staticmethod
    def makedir(path):
        """
        @機能：ディレクトリの新規作成（既にあれば作らない、無ければ作る）
        @引数：作りたいパス
        @戻値：None
        """
        if not os.path.isdir(path):
            os.mkdir(path)
    
    
    @staticmethod
    def datacounter(datapath, limit:int=None):
        """
        @機能：テキストファイルの行数を数えるだけ
        @引数：datapath = 数えたいテキストファイルのパス, limit = データ数の上限
        @戻値：行数
        """
        with open(datapath) as f:
            readlines = f.readlines()
            if limit : readlines = readlines[:limit]
            
        return len(readlines)
    
    
    @staticmethod
    def log(dir, strings:list):
        """
        @機能：ログファイルにログを出力
        @引数：dir = ログファイルのディレクトリ, strings;list = 出力したい数値などのリスト
        @戻値：None
        """
        with open(f"{dir}/log.txt", mode="a") as logoutput:
            print(strings, file=logoutput)
    
    
    @staticmethod
    def ignore_unhashable(func):
        """
        @機能：functoolsのunhashableのエラーを回避
        @引数：引数は関数だけど、デコレータで使って
        @戻値：与えた関数の戻り値
        """
        uncached = func.__wrapped__
        attributes = functools.WRAPPER_ASSIGNMENTS + ("cache_info", "cache_clear")
        @functools.wraps(func, assigned=attributes) 
        def wrapper(*args, **kwargs): 
            try: 
                return func(*args, **kwargs) 
            except TypeError as error: 
                if ("unhashable type" in str(error)):
                    return uncached(*args, **kwargs) 
                raise 
        wrapper.__uncached__ = uncached
        return wrapper
    
    
    @staticmethod
    def print_dict(dictionary:dict, head:int=None, tail:int=None):
        """
        @機能：辞書のきれいなprint
        @引数：辞書, headまたはtail=表示行数の設定
        @戻値：None
        """
        
        keys = sorted(dictionary.keys())
        vals = sorted(dictionary.values())
        if head is not None:
            keys = keys[:head+1]
            vals = keys[:head+1]
        elif tail is not None:
            keys = keys[-1*tail:]
            vals = keys[-1*tail:]
        for i, (key, val) in enumerate(zip(keys, vals)):
            print(f"{i}\t\t{key}\t: {val}")

        
class Calc:
    """
    @機能：諸々の計算を含んだクラス
    @関数：precosion(), recall(), f1_score(), inverse_matrix(), cos_sim(), make_fresh_tensor()
    """
    @staticmethod
    def precision(tp, fp):
        """
        @機能：precisionの計算
        @引数：tp = TruePositive, fp = FalsePositive
        @戻値：precision
        """
        return tp/(tp+fp)


    @staticmethod
    def recall(tp, fn):
        """
        @機能：recallの計算
        @引数：tp = TruePositive, fn = FalseNegative
        @戻値：recall
        """
        return tp/(tp+fn)
    
    
    @staticmethod
    def f1_score(precision, recall):
        """
        @機能：F1値の計算
        @引数：precision, recall
        @戻値：F1値
        """
        return 2*(precision*recall) / (precision+recall)
    
    
    @staticmethod
    def inverse_matrix(mat):
        """
        @機能：行列を反転
        @引数：配列(numpy配列 or list)
        @戻値：配列(numpy配列 or list)
        """
        if isinstance(mat, list):
            return list(np.array(mat) ^ 1)
        return mat ^ 1
    
    
    @staticmethod
    def cos_sim(v1, v2):
        """
        @機能：cosine distance
        @引数：numpy配列２つ
        @戻値：cos類似度
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    
    @staticmethod
    def make_fresh_tensor(fresh:list, kernel_size:int, batch_size:int=None, isStr:bool=False):
        """
        @機能：フレッシュ性状データを学習用に整形
        @引数：fresh = フレッシュ性状データ, kernel_size = 変換後のカーネルサイズ, batch_size = バッチサイズ, isStr = freshがstring型であるか否か(FalseでOK)
        @戻値：shape:(kernel_size, kernel_size, len(fresh))に整形されたnumpy配列
        """
        if isStr: fresh = list(map(lambda x : list(map(float, x.split(" "))), fresh))
        if batch_size is None:
            output = [[fresh for col in range(kernel_size[1])] for row in range(kernel_size[0])]
        else:
            output = [[[fresh[batch] for col in range(kernel_size[1])] for row in range(kernel_size[0])] for batch in range(batch_size)]
        
        return np.asarray(output, dtype=np.float32)

    
    @staticmethod
    def make_fresh_4x4_tensor(fresh:list):
        """
        @機能：畳み込み層への入力を考慮したフレッシュ性状データの整形
        @引数：fresh = フレッシュ性状データ
        @戻値：Shape:(4,4,1)のnumpy配列
        """
        mask_0 = [[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]]
        mask_1 = [[0, 0, 1, 1],
                  [0, 4, 4, 1],
                  [2, 4, 4, 2],
                  [2, 2, 3, 3]]
        
        tensor = [[0]*4 for _ in range(4)]
        
        for row, (m_0, m_1) in enumerate(zip(mask_0, mask_1)):
            for col, (val_0, val_1) in enumerate(zip(m_0, m_1)):
                tensor[row][col] = (val_0 + val_1)/2.
        
        return np.asarray(tensor, dtype=np.float32)


class TimeCounter:
    """
    @機能：終了時刻の予測
    @引数：datacount = 総データ数
    @関数：predictTime()
    """
    def __init__(self, datacount):
        self.datacount = datacount
        self.first = time.time()
    
    
    def calcSecond(self, index:int):
        """
        @機能：入力を元に残り時間を計算
        @引数：index:int = 現在までに処理し終わったデータ数
        @戻値：残り秒数
        """
        
        try:
            return True, (self.datacount-index)/index * (time.time() - self.first)
        
        except:
            return False, 0

    
    def predictTime(self, index:int):
        """
        @機能：終了時刻の予測
        @引数：index:int = 現在までに処理し終わったデータ数
        @戻値：予測終了時刻(形式はhh:mm:ss)
        """
        frag, second = self.calcSecond(index)
        if not frag: return f"( Time input error. The argment is {index}.)"
        hour = second//(60*60)
        minite = (second//60) - (hour*60)
        second %= 60
        
        return f"{int(hour):02d}:{int(minite):02d}:{int(second):02d}"


class ImageManager:
    """
    @機能：画像の処理に関わるクラス
    @関数：get_palette(), get_average_image(), adjustData(), resize(), get_image_key(), get_fresh(), get_id()
    """
    
    random.seed(1)
    with open("/workspace/mesh_dataset/fresh.pickle", mode="rb") as f:
        fresh_data = pickle.load(f)
    with open("/workspace/mesh_dataset/fold.pickle", mode="rb") as f:
        fold_data = pickle.load(f)
    with open("/workspace/mesh_dataset/answer.pickle", mode="rb") as f:
        answer_data = pickle.load(f)
    
    fold = 3
    LE = None
    AE_model = None
    AE_model_id = "e-unet_20221014_ssim_mse"
    # AE_model_weights = lambda *x: f"/workspace/fullframe/result/autoencoder/AE_e-unet_{x[0]}_fold{x[1]}"
    AE_model_weights = lambda *x: f"/workspace/fullframe/result/autoencoder/AE_{x[0]}_fold{x[1]}"
    
    @staticmethod
    def get_palette(is_fourclasses:bool=False, is_afterclass:bool=False):
        """
        @機能：マスク画像から読み取る色の設定
        @引数：void
        @戻値：対象のRGBの値が入った配列
        """
        
        if is_fourclasses:
            palette = [[0]*3,   # Before
                       [64]*3,  # b-Before
                       [128]*3, # b-Just
                       [255]*3] # Just
        elif is_afterclass:
            palette = [[0]*3,   # Before
                       [128]*3, # Just
                       [255]*3] # After
        else:
            palette = [[0]*3,       # Before
                       [255]*3] # Just
            
        return np.asarray(palette)
    
    
    @staticmethod
    def get_average_image(size, is_grayscale:bool=False, average_image_path:str="/workspace/osada_ws/average_image_0516.png", is_hsv:bool=False):
        """
        @機能：平均画像の読み込み
        @引数：size = 入力画像サイズ, isGrayScale:bool = グレースケールで読み込むか否か
        @戻値：平均画像のデータが格納された配列
        """
        avg_img = img_to_array(load_img(average_image_path, color_mode='rgb'))
        avg_img = Image.fromarray(np.uint8(avg_img))
        avg_img = avg_img.resize(size)
        
        ## RGB -> HSV
        if is_hsv:
            avg_img = avg_img.convert("HSV")
            
        avg_img = np.array(avg_img, dtype=np.float32)
        
        if is_grayscale:
            avg_img = np.reshape(np.sum(avg_img, axis=2)/3., (1, *size, 1))
            
        return avg_img
    
    
    @staticmethod
    def adjust_data(img, mask, is_fullframe=True, is_absolute_resize=False,
                   is_use_average_image=True, size=(540, 540), is_grayscale=False, num_classes=2,
                   average_image_path="/workspace/osada_ws/average_image_0516.png",
                   is_use_bcl=False, mix=None, classification:str="before-just",
                   to_two_classes:bool=False, is_use_LE:bool=False,
                   LE_mode:str="circle", LE_const:int=50, color_type:str="rgb",
                   normalization:str="default", minmax_area:list=(32, 32),
                   autoencoder:bool=False, noise:bool=True, noise_type:str="linear", use_AE_input:bool=False,
                   autoencoder_loss:str="ssim_mse", is_flip:bool=False, flip_list:list=[0, 0, 1, 1, 2, 3],
                   is_rotate:bool=False, rotate_rate:float=0.75, rotate_degrees:list=[[0, 359]], is_enlarge:bool=False,
                   all_in_one:bool=False,):
        """
        @機能：コンクリ画像を0~1に正規化、マスク画像をone-hot表現に変換
        @引数：img = コンクリ画像一枚, mask = マスク画像一枚, その他定数が沢山
        @戻値：変換し終わったコンクリ画像, one-hot表現にしたマスク画像
        """
        
        ## 分類クラス数の振り分け
        is_afterclass = classification == "before-just-after"
        is_fourclasses = classification == "fourclasses" and not is_afterclass
        if to_two_classes:
            is_fourclasses = False
            num_classes = 2
        
        ## 輝度拡張クラスのインスタンス生成
        if ImageManager.LE is None:
            ImageManager.LE = LuminanceExtender(size)
        
        ## 正規化の大いなる力
        master_normalization = None
        
        ## AE無しの照度変更
        if all_in_one:
            use_AE_input = False
        
        ## 画像サイズをconfigで設定した値に調節
        if is_fullframe or is_absolute_resize:
            img_buf = [None]*(img.shape[0])
            mask_buf = [None]*(mask.shape[0])
            for index, (i, m) in enumerate(zip(img, mask)):
                ## Transform illuminance
                if is_use_LE:
                    i = ImageManager.LE.extend(i, LE_mode, LE_const, noise_type)
                
                ## Flip upside to bottom side
                if is_flip:
                    """
                    - : up-down | left-right
                    ------------------------
                    0 : False   | False
                    1 : True    | False
                    2 : True    | True
                    3 : False   | True
                    """
                    flip = random.choice(flip_list)
                    ## up-down
                    if (flip in [1, 2]):
                        i = np.flipud(i)
                        m = np.flipud(m)
                    ## left-right
                    if (flip in [2, 3]):
                        i = np.fliplr(i)
                        m = np.fliplr(m)
                
                ## Rotation from affine transformation
                if is_rotate:
                    if (random.random() < rotate_rate):
                        rotate = random.randint(*random.choice(rotate_degrees))
                        i = AT.rotate_enlargement(i, rotate)
                        m = AT.rotate_enlargement(m, rotate)
                
                ## Enlargement
                if is_enlarge:
                    large = random.random() + 1
                    i = AT.enlargement(i, large)
                    m = AT.enlargement(m, large)
                
                try:
                    img_buf[index] = Image.fromarray(np.uint8(i))
                    mask_buf[index] = Image.fromarray(np.uint8(m))
                except:
                    img_buf[index] = Image.fromarray(np.uint8(np.reshape(i, i.shape[:2])))
                    mask_buf[index] = Image.fromarray(np.uint8(np.reshape(m, m.shape[:2])))

                img_buf[index] = img_buf[index].resize(size)
                mask_buf[index] = mask_buf[index].resize(size)
                
                ## RGB -> HSV
                if (color_type == "hsv"):
                    img_buf[index] = img_buf[index].convert("HSV")
                
                ## RGB -> LBP
                if (color_type == "lbp"):
                    gray_img = img_buf[index].convert("L")
                    radius = 1
                    lbp = feature.local_binary_pattern(gray_img, radius*8, radius, method="uniform")
                    new_img = np.zeros((*size, 3))
                    for dim in range(3):
                        new_img[:, :, dim] += lbp
                    img_buf[index] = np.copy(new_img)
                    del lbp, new_img, gray_img
                    
                ## RGB -> LBP-3c
                ## LBPの計算半径を変えて３チャネルにしたもの(RGBの重みをそのまま使える)
                if (color_type == "lbp-3c"):
                    gray_img = img_buf[index].convert("L")
                    new_img = np.zeros((*size, 3))
                    for dim in range(3):
                        radius = dim+1
                        new_img[:, :, dim] += feature.local_binary_pattern(gray_img, radius*8, radius, method="uniform")
                    img_buf[index] = np.copy(new_img)
                    del new_img, gray_img
                
                ## RGB -> Gray(1c) + LBP(2c)
                if (color_type == "gray-lbp"):
                    gray_img = img_buf[index].convert("L")
                    new_img = np.zeros((*size, 3))
                    for dim in range(3):
                        if dim:
                            radius = dim
                            new_img[:, :, dim] += feature.local_binary_pattern(gray_img, radius*8, radius, method="uniform")
                        else:
                            new_img[:, :, dim] += gray_img
                        new_img[:, :, dim] = new_img[:, :, dim] / np.max(new_img[:, :, dim])
                    img_buf[index] = np.copy(new_img)
                    del new_img, gray_img
                    
                    ## 既に正規化されたことをあらわす
                    master_normalization = "pre"
                
                ## RGB -> HoG
                if (color_type == "hog"):
                    # gray_img = np.array(img_buf[index].convert("L"), dtype=np.float32)
                    fd, hog = feature.hog(img_buf[index], orientations=9, pixels_per_cell=(4, 4),
                                          cells_per_block=(2, 2), visualize=True, multichannel=True)
                    ## 正規化後に0~1になるように逆に0~255に正規化する(効率最悪だけど一旦)
                    hog_coef = 255./np.max(hog)
                    hog *= hog_coef
                    
                    new_img = np.zeros((*size, 3))
                    for dim in range(3):
                        new_img[:, :, dim] += hog
                    img_buf[index] = np.copy(new_img)
                    # img_buf[index] = np.reshape(hog * hog_coef, (*size, 1))
                    del hog, fd, new_img
                    
                ## RGB -> HoG-3c
                if (color_type == "hog-3c"):
                    gray_img = np.array(img_buf[index].convert("L"), dtype=np.float32)
                    new_img = np.zeros((*size, 3))
                    for dim in range(3):
                        fd, hog = feature.hog(gray_img, orientations=9, pixels_per_cell=(1<<(dim+2), 1<<(dim+2)),
                                              cells_per_block=(2, 2), visualize=True, multichannel=False)
                        hog_coef = 255./np.max(hog)
                        hog *= hog_coef
                        new_img[:, :, dim] += hog
                    img_buf[index] = np.copy(new_img)
                    del hog, fd, new_img

                try:
                    img[index] = np.array(img_buf[index], dtype=np.float32)
                    mask[index] = np.array(mask_buf[index], dtype=np.float32)
                except:
                    img[index] = np.array(np.reshape(img_buf[index], (*size, 1)), dtype=np.float32)
                    mask[index] = np.array(np.reshape(mask_buf[index], (*size, 1)), dtype=np.float32)
        
        ## Mixup用
        if is_use_bcl:
            mix_buf = [None]*(img.shape[0])
            mix_coefes = [0]*(img.shape[0])
            for index, (i, m) in enumerate(zip(img, mix)):
                try:
                    mix_buf[index] = Image.fromarray(np.uint8(m))
                except:
                    mix_buf[index] = Image.fromarray(np.uint8(np.reshape(m, m.shape[:2])))
                mix_buf[index].resize(size)
                try:
                    mix[index] = np.asarray(mix_buf[index])
                except:
                    mix[index] = np.asarray(np.reshape(mix_buf[index], (*size, 1)))
                
                mix_coef = random.random()
                if (mix_coef < 0.5):
                    mix_coef = 1. - mix_coef
                mix_coefes[index] = mix_coef
                img[index] *= mix_coef
                img[index] += (1.-mix_coef)*mix[index]

        img = np.float32(img)
        
        ## for AutoEncoder
        if autoencoder and not use_AE_input:
            
            if noise:
                origin = np.copy(img)
                random_bound_area = [1<<i for i in range(5)] + [3*i for i in range(1, 4)]
                add_const = 100
                
                div_row, div_col = random.choice(random_bound_area), random.choice(random_bound_area)
                row, col = img.shape[1] // div_row, img.shape[2] // div_col
                for index, _ in enumerate(img):
                    for r in range(div_row):
                        for c in range(div_col):
                            ## 線形／非線形をランダムに
                            if (noise_type == "mix"):
                                co_noise_type = random.randint(0, 1)
                            else:
                                co_noise_type = -1
                                
                            ## 線形
                            if (noise_type == "linear") or (co_noise_type == 0):
                                add_value = random.randint(-add_const, add_const)
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] += add_value
                            
                            ## 非線形(tanh)
                            elif (noise_type == "tanh") or (co_noise_type == 1):
                                origin_value = random.random()*1.5
                                buf = np.copy(img[index, row*r:row*(r+1), col*c:col*(c+1), :])
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] *= 0
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] += 255. * (np.tanh(5 * (buf/255 - 0.5) + origin_value) + 1.) / 2.
                                del buf
                
                ## 0 ~ 1
                origin /= 255.
                img = np.clip(img, 0, 255)
                img /= 255.
                
                return img, origin
            
            img /= 255.
            
            return img, img
        
        
        ## AutoEncoderによる照度補正
        elif use_AE_input:
            
            if ImageManager.AE_model is None:
                if (autoencoder_loss == "rec_mse"):
                    autoencoder_loss = MyLosses.rectified_mse_loss
                elif (autoencoder_loss == "ssim"):
                    autoencoder_loss = MyLosses.ssim_loss
                elif (autoencoder_loss == "ssim_mse"):
                    autoencoder_loss = MyLosses.ssim_mse_loss
                
                skip_buf = ImageManager.AE_model_id.split("skip-connection-")
                if (len(skip_buf) > 1):
                    skip_connection = int(skip_buf[-1])
                else:    
                    skip_connection = 3
                
                if ("esp" in ImageManager.AE_model_id):
                    ImageManager.AE_model = ESPNet.run((128, 128, 3), num_classes=3, dropout_const=0.01, optimizer="adam", loss=autoencoder_loss, autoencoder=True)
                else:
                    ImageManager.AE_model = E_UNet.run((576, 576, 3), num_classes=3, loss=autoencoder_loss, autoencoder=True, skip_connection=skip_connection)
                w = ImageManager.AE_model_weights(ImageManager.AE_model_id, ImageManager.fold)
                ImageManager.AE_model.load_weights(w)
                cp.cprint(f"@ load AE_model : ( {ImageManager.AE_model_id}, fold:{ImageManager.fold} ) > {w}", "orange")
                if (noise_type == "includeAE-noise"):
                    cp.cprint(f"@ available AE-input and luminance noise", "orange")
            
            ## 締固め判定用E-UNetの学習時にノイズを入れるパターン
            if (noise_type == "includeAE-noise"):
                
                ## ノイズの有無をランダムに決定
                noise_rate = 0.25
                is_noise = [random.random() <= noise_rate for _ in img]
                random_bound_area = [1<<i for i in range(4)] + [3*i for i in range(1, 3)]
                add_const = 100
                
                for index, val in enumerate(img):
                    if not is_noise[index]: continue
                    
                    div_row, div_col = random.choice(random_bound_area), random.choice(random_bound_area)
                    row, col = val.shape[0] // div_row, val.shape[1] // div_col
                    
                    for r in range(div_row):
                        for c in range(div_col):
                            ## 線形
                            if random.randint(0, 1):
                                add_value = random.randint(-add_const, add_const)
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] += add_value
                            
                            ## 非線形(tanh)
                            else:
                                origin_value = random.random()*1.5
                                buf = np.copy(img[index, row*r:row*(r+1), col*c:col*(c+1), :])
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] *= 0
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] += 255. * (np.tanh(5 * (buf/255 - 0.5) + origin_value) + 1.) / 2.
                                del buf
                
            img /= 255.
            img = ImageManager.AE_model.predict(img, verbose=0)
            
            ## もう一度
            if (use_AE_input == 2):
                img = ImageManager.AE_model.predict(img, verbose=0)
            
            ## ラベルの設定色を取得
            palette = ImageManager.get_palette(is_fourclasses, is_afterclass)
            
            ## マスク画像を元に、正解ラベル(ont-hot表現)を作成する
            onehot = np.zeros((mask.shape[0], *size, num_classes-int(is_afterclass)), dtype=np.float32)#, dtype=np.uint8)
            for i in range(2 + 2*is_fourclasses + is_afterclass):
                cat_color = palette[i]
                if is_grayscale:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]), 1, 0)
                else:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                                    (mask[:, :, :, 1] == cat_color[1]) &
                                    (mask[:, :, :, 2] == cat_color[2]), 1, 0)
                if is_afterclass:
                    index = [0, 0, 1][i]
                else:
                    index = i
                onehot[:, :, :, index] += temp
            
            return img, onehot
        
        
        ## 照度補正、締固め判定など全部の学習を一挙にやる
        elif all_in_one:
            
            ## ノイズの有無をランダムに決定
            # noise_rate = 0.25
            # is_noise = [random.random() <= noise_rate for _ in img]
            random_bound_area = [1<<i for i in range(4)] + [3*i for i in range(1, 3)]
            add_const = 100
            
            for index, val in enumerate(img):
                # if not is_noise[index]: continue
                
                div_row, div_col = random.choice(random_bound_area), random.choice(random_bound_area)
                row, col = val.shape[0] // div_row, val.shape[1] // div_col
                
                for r in range(div_row):
                    for c in range(div_col):
                        ## 線形
                        if random.randint(0, 1):
                            add_value = random.randint(-add_const, add_const)
                            img[index, row*r:row*(r+1), col*c:col*(c+1), :] += add_value
                        
                        ## 非線形(tanh)
                        else:
                            origin_value = random.random()*1.5
                            buf = np.copy(img[index, row*r:row*(r+1), col*c:col*(c+1), :])
                            img[index, row*r:row*(r+1), col*c:col*(c+1), :] *= 0
                            img[index, row*r:row*(r+1), col*c:col*(c+1), :] += 255. * (np.tanh(5 * (buf/255 - 0.5) + origin_value) + 1.) / 2.
                            del buf
            
            ## 平均画像を減算
            # if is_use_average_image:
            #     avg_img = ImageManager.get_average_image(size, is_grayscale, average_image_path, color_type=="hsv")
            #     img -= avg_img
                
            img /= 255.

            ## ラベルの設定色を取得
            palette = ImageManager.get_palette(is_fourclasses, is_afterclass)
            
            ## マスク画像を元に、正解ラベル(ont-hot表現)を作成する
            onehot = np.zeros((mask.shape[0], *size, num_classes-int(is_afterclass)), dtype=np.float32)#, dtype=np.uint8)
            for i in range(2 + 2*is_fourclasses + is_afterclass):
                cat_color = palette[i]
                if is_grayscale:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]), 1, 0)
                else:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                                    (mask[:, :, :, 1] == cat_color[1]) &
                                    (mask[:, :, :, 2] == cat_color[2]), 1, 0)
                if is_afterclass:
                    index = [0, 0, 1][i]
                else:
                    index = i
                onehot[:, :, :, index] += temp
            
            return img, onehot
        
        
        ## AutoEncoder以外(通常)
        else:
            
            ## 特に正規化に制約が無い場合
            if master_normalization is None:
                ## 平均画像を定義
                avg_img = ImageManager.get_average_image(size, is_grayscale, average_image_path, color_type=="hsv")
                
                ## z-index norm
                if (normalization == "minmax"):
                    # img -= avg_img
                    max_val = np.max(img, axis=3)
                    min_val = np.min(img, axis=3)
                    diff = max_val - min_val
                    ## 0除算を回避
                    without_zero = (diff == 0) * min_val
                    without_zero += without_zero == 0
                    ## -1 ~ 1
                    img = ((img - min_val) / (diff + without_zero) * 2.) - 1.
                
                ## mesh-area z-index norm
                elif (normalization == "area_minmax"):
                    # img -= avg_img
                    div_row, div_col = minmax_area
                    row, col = img.shape[1] // div_row, img.shape[2] // div_col
                    for index, _ in enumerate(img):
                        for r in range(div_row):
                            for c in range(div_col):
                                max_val = np.max(img[index, row*r:row*(r+1), col*c:col*(c+1), :], axis=(0, 1))
                                min_val = np.min(img[index, row*r:row*(r+1), col*c:col*(c+1), :], axis=(0, 1))
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] -= min_val
                                diff = max_val - min_val
                                ## 0除算を回避
                                without_zero = (diff == 0) * min_val
                                without_zero += without_zero == 0
                                img[index, row*r:row*(r+1), col*c:col*(c+1), :] = img[index, row*r:row*(r+1), col*c:col*(c+1), :] / (diff + without_zero)
                    ## -1 ~ 1
                    img = (img*2.) - 1.
                
                ## 画像データを0~1に正規化
                else:
                    if(np.max(img) > 1):
                        if is_use_average_image:
                            img -= avg_img
                        img /= 255.
                    ## 平均画像を引いた後、画像データを0~1に正規化
                    elif is_use_average_image:
                        avg_img /= 255.
                        img -= avg_img
            
            ## 既に正規化が完了していれば無視
            elif (master_normalization == "pre"):
                pass
            
            else:
                pass
            
            ## ラベルの設定色を取得
            palette = ImageManager.get_palette(is_fourclasses, is_afterclass)
            
            ## マスク画像を元に、正解ラベル(ont-hot表現)を作成する
            onehot = np.zeros((mask.shape[0], *size, num_classes), dtype=np.float32)#, dtype=np.uint8)
            for i in range(2 + 2*is_fourclasses + is_afterclass):
                cat_color = palette[i]
                if is_grayscale:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]), 1, 0)#1*(1-label_smoothing), 0*label_smoothing)
                else:
                    temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                                    (mask[:, :, :, 1] == cat_color[1]) &
                                    (mask[:, :, :, 2] == cat_color[2]), 1, 0)#, 1*(1-label_smoothing), 0*label_smoothing)
                onehot[:, :, :, i] = temp
            
            ## Mixup用の正解ラベルに整形
            if is_use_bcl:
                for i, batch in enumerate(onehot):
                    ## Before
                    before = np.copy(onehot[i, :, :, 0])
                    just = np.copy(onehot[i, :, :, 1])
                    if (np.sum(batch[:, :, 0]) > np.sum(batch[:, :, 0])):
                        onehot[i, :, :, 0] *= 0
                        onehot[i, :, :, 1] *= 0
                        onehot = np.float64(onehot)
                        onehot[i, :, :, 0] += mix_coefes[i]*before
                        onehot[i, :, :, 0] += (1.-mix_coefes[i])*just
                        onehot[i, :, :, 1] += (1.-mix_coefes[i])*before
                        onehot[i, :, :, 1] += mix_coefes[i]*just
                    ## Just
                    else:
                        onehot[i, :, :, 0] *= 0
                        onehot[i, :, :, 1] *= 0
                        onehot = np.float64(onehot)
                        onehot[i, :, :, 0] += (1.-mix_coefes[i])*before
                        onehot[i, :, :, 0] += mix_coefes[i]*just
                        onehot[i, :, :, 1] += mix_coefes[i]*before
                        onehot[i, :, :, 1] += (1.-mix_coefes[i])*just
            
            return img, onehot
    
    
    @staticmethod
    def adjust_data_multilosses(img, mask, is_fullframe=True,
                   is_use_average_image=True, size=(570, 570), is_grayscale=False,
                   average_image_path="/workspace/osada_ws/average_image_0516.png"):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        ## 画像サイズをconfigで設定した値に調節
        if is_fullframe:
            img_buf = [None]*(img.shape[0])
            mask_buf = [None]*(mask.shape[0])
            for index, (i, m) in enumerate(zip(img, mask)):
                try:
                    img_buf[index] = Image.fromarray(np.uint8(i))
                    mask_buf[index] = Image.fromarray(np.uint8(m))
                except:
                    img_buf[index] = Image.fromarray(np.uint8(np.reshape(i, i.shape[:2])))
                    mask_buf[index] = Image.fromarray(np.uint8(np.reshape(m, m.shape[:2])))
                
                img_buf[index] = img_buf[index].resize(size)
                mask_buf[index] = mask_buf[index].resize(size)
                
                ## RGB -> HSV
                # if (color_type == "hsv"):
                    # img_buf[index] = img_buf[index].convert("HSV")
                
                try:
                    img[index] = np.asarray(img_buf[index])
                    mask[index] = np.asarray(mask_buf[index])
                except:
                    img[index] = np.asarray(np.reshape(img_buf[index], (*size, 1)))
                    mask[index] = np.asarray(np.reshape(mask_buf[index], (*size, 1)))
        
        ## 画像データを0~1に正規化
        if(np.max(img) > 1):
            if is_use_average_image:
                img -= ImageManager.get_average_image(size, is_grayscale, average_image_path)
            img /= 255.
        ## 平均画像を引いた後、画像データを0~1に正規化
        elif is_use_average_image:
            avg_img = ImageManager.get_average_image(size, is_grayscale, average_image_path)
            avg_img /= 255.    
            img -= avg_img
 
        ## ラベルの設定色を取得
        palette = ImageManager.get_palette(1)

        ## マスク画像を元に、正解ラベル(ont-hot表現)を作成する
        onehot = np.zeros((mask.shape[0], *size, 2), dtype=np.float32)#, dtype=np.uint8)
        onehot_ml = np.zeros((mask.shape[0], *size, 4), dtype=np.float32)#, dtype=np.uint8)
        
        for i in range(4):
            cat_color = palette[i]
            if is_grayscale:
                temp = np.where((mask[:, :, :, 0] == cat_color[0]), 1, 0)#1*(1-label_smoothing), 0*label_smoothing)
            else:
                temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                                (mask[:, :, :, 1] == cat_color[1]) &
                                (mask[:, :, :, 2] == cat_color[2]), 1, 0)#, 1*(1-label_smoothing), 0*label_smoothing)
            onehot_ml[:, :, :, i] = temp
        
        onehot[:, :, :, 0] = onehot_ml[:, :, :, 0] + onehot_ml[:, :, :, 1]
        onehot[:, :, :, 1] = onehot_ml[:, :, :, 2] + onehot_ml[:, :, :, 3]
        
        return img, onehot, onehot_ml


    @staticmethod
    def resize(img, row, col, is_numpy:bool=True, dtype=np.uint8):
        """
        @機能：配列(画像)のリサイズ
        @引数：img = 配列, row = 変換後の行数, col = 変換後の列数, is_numpy = Numpy配列で出力するか否か, dtype = 出力の型
        @戻値：変換後の配列
        """
        number_rows = len(img)
        number_columns = len(img[0])
        output = [[img[int(number_rows*r/row)][int(number_columns*c/col)] for c in range(col)] for r in range(row)]
        if is_numpy: output = np.array(output, dtype=dtype)
        return output
    
    
    @staticmethod
    def get_image_key(image_path, without_mesh_id:bool=False, is_fullframe:bool=False):
        """
        @機能：画像の主キーの取得
        @引数：画像パス
        @戻値：主キー
        """
        ## /workspace/Dataset/fullframe/image/190802/06_0150.png /workspace/Dataset/fullframe/masked_4class/190802/masked_06_0150.png
        
        if not len(image_path): return "missing"
        
        if is_fullframe:
            path_elements = image_path.split("/")
            _0, _1, _2, _3, _4, date, element, *_end = path_elements
            element = element.replace(".png", "")
            place, time_id = element.split("_")
            if without_mesh_id:
                key = f"{date}_{place}"
            else:
                key = f"{date}_{place}_{time_id}"
                
        else:
            path_elements = image_path.split("/")
            _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
            element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
            date, place, time_id, mesh_id = element.split("_")
            if without_mesh_id:
                key = f"{date}_{place}"
            else:
                key = f"{date}_{place}_{mesh_id}"

        return key
    
    
    @staticmethod
    def get_fresh(image_path):
        """
        @機能：画像パス名からフレッシュ性状データの取得
        @引数：画像パス
        @戻値：フレッシュ性状データ
        """
        
        path_elements = image_path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")
        key = f"{date}_{place}"

        return ImageManager.fresh_data[key]
    
    
    @staticmethod
    def get_fold(image_path):
        """
        @機能：画像パス名からFOLDの取得
        @引数：画像パス
        @戻値：FOLD
        """
        
        path_elements = image_path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")
        key = f"{date}_{place}"

        return ImageManager.fold_data[key]

    
    @staticmethod
    def get_answer(image_path):
        """
        @機能：画像パス名から正解ラベルの取得
        @引数：画像パス
        @戻値：正解ラベル
        """
        
        path_elements = image_path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")
        key = f"{date}_{place}_{time_id}_{mesh_id}"

        return ImageManager.answer_data[key]
    
    
    @staticmethod
    def get_id(image_path):
        """
        @機能：画像パス名からメッシュ番号と動画内時間の取得
        @引数：画像パス
        @戻値：メッシュ番号, 動画内時間
        """
        
        path_elements = image_path.split("/")
        _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
        element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
        date, place, time_id, mesh_id = element.split("_")

        return int(mesh_id)-1, int(time_id)
    
    
    @staticmethod
    def encode_bin_img(bin_img):
        """
        @機能：２値画像を圧縮
        @引数：
        @戻値：
        """
        
        encoded = f"{bin_img[0]}e"
        accum = [0, 0]
        for val in bin_img:
            in_val = (val+1)%2
            if accum[in_val]:
                encoded += f"{accum[in_val]},"
                accum[in_val] = 0; accum[val] = 1
            else:
                accum[val] += 1
        encoded += f"{accum[val]}"
        
        return encoded
        
        
    @staticmethod
    def decode_bin_img(bin_txt):
        """
        @機能：圧縮された２値画像を復号
        @引数：
        @戻値：
        """
        
        decoded = []
        now = int(bin_txt[0])
        bin_txt = bin_txt[2:]

        for val in bin_txt.split(","):
            if len(val):
                decoded += list(map(int, list(int(val)*str(now))))
                now = (now+1)%2
        
        return decoded
    
    
    @staticmethod
    def set_AE_id(AE_model_id:str):
        """
        AEのモデル変更
        """
        
        ImageManager.AE_model_id = AE_model_id
        ImageManager.AE_model = None
    

class ModelManager:
    """
    @機能：モデルの生成・処理に関するクラス
    @関数：create_model
    """
    @staticmethod
    def create_model(model_name:str=None, weights:str=None, save_id="", size=(540, 540),
                     is_grayscale=False, fresh_kernel_size=(16, 16), is_load_weight=False,
                     is_use_bcl=False, loss:str=None, is_fusion_face:bool=False, metric_func:str="sphereface",
                     nullfication_metric:bool=False, is_h5:bool=True, dropout_const:float=0.25, label_smoothing:float=0.,
                     norm:bool="batch_norm", use_attention:bool=False, classification:str="before-just",
                     optimizer:str="adam", multi_losses:bool=False, eunet_metric_mode:str="conv1333", eunet_metric_subcontext:str="default",
                     is_HoG:bool=False, reduce_const:float=1, learning_rate:float=1e-3, fold:int=0, reduce:str="ssim"):
        """
        @機能：モデルの定義
        @引数：model_name = モデルの定義名, weights = 読み込む重みファイルのパス
        @戻値：モデル
        """
        model = None
        if model_name is None:
            cp.cprint(f"[!] \"model_name\" is blank.", "red")
                
        else:
            if loss is None:
                if is_use_bcl:
                    loss = "kullback_leibler_divergence"
                else:
                    loss = "categorical_crossentropy"
            
            num_classes = {"before-just" : 2, "fourclasses" : 4, "before-just-after" : 2}[classification]
            
            cp.cprint(f"- model : {model_name} -", "cyan")
            cp.cprint(f"- loss  : {loss} -", "cyan")
            cp.cprint(f"- SADE_ID : {save_id} -", "cyan")
            
            ## モデルの振り分け
            if (model_name in ["unet", "unet_arcface", "unet_cosface", "unet_sphereface"]):
                is_pure_unet = model_name == "unet"
                model = SemanticSegmentation.unet([*size, 3**(not is_grayscale)], loss, is_pure_unet, norm=norm, num_classes=num_classes)
            elif (model_name == "unet_fresh"):
                model = SemanticSegmentation.unet_include_fresh([*size, 3**(not is_grayscale)], [*fresh_kernel_size, 5], loss)
            
            elif (model_name == "pspnet"):
                model = SemanticSegmentation.pspnet([*size, 3**(not is_grayscale)], loss)
            
            elif (model_name == "unet_metric_classifier"):
                model = SemanticSegmentation.unet([*size, 3**(not is_grayscale)], loss, False, norm=norm)
                model = SemanticSegmentation.unet_available_metric(model, "sphereface",
                                                                   [*size, 3**(not is_grayscale)], num_classes, loss, is_fusion_face=is_fusion_face,
                                                                   freeze_classifier=False, nullfication_metric=nullfication_metric, dropout_const=0.,
                                                                   label_smoothing=label_smoothing, is_eunet=("e-unet" in model_name),
                                                                   eunet_metric_mode=eunet_metric_mode, eunet_metric_subcontext=eunet_metric_subcontext)
                if not is_h5:
                    weights = "/".join(weights.split("/")[:-1])
                    weights = weights.replace("metric_classifier", metric_func)
                    # cp.cprint("origin model", "red")
                    # model.summary()
                    # model = load_model(weights)
                    # cp.cprint(weights, "pink")
                    # model.summary();return None
                model.load_weights(weights)
                model = SemanticSegmentation.unet_learning_classifier(model)
                is_load_weight = False
            
            elif (model_name == "unet_only_classifier"):
                model = SemanticSegmentation.unet([*size, 3**(not is_grayscale)], loss, False, norm=norm)
                if not is_h5:
                    weights = "/".join(weights.split("/")[:-1])
                    weights = weights.replace("only_classifier", "")
                model.load_weights(weights)
                model = SemanticSegmentation.unet_learning_classifier(model)
                is_load_weight = False
            
            elif (model_name in ["e-unet", "e-unet_arcface", "e-unet_cosface", "e-unet_sphereface"]):
                is_pure_eunet = model_name == "e-unet"
                model = E_UNet.run([*size, 3**(not is_grayscale)], num_classes=num_classes, dropout_const=dropout_const, is_compile=is_pure_eunet,
                                   loss=loss, use_attention=use_attention, optimizer=optimizer, multi_losses=multi_losses, reduce_const=reduce_const,
                                   learning_rate=learning_rate)
            
            elif (model_name in ["espnet"]):
                is_pure_espnet = model_name == "espnet"
                model = ESPNet.run([*size, 3**(not is_grayscale)], num_classes=num_classes, dropout_const=dropout_const, is_compile=is_pure_espnet,
                                   loss=loss, optimizer=optimizer, autoencoder=False, learning_rate=learning_rate)
            
            elif (model_name == "e-unet_metric_classifier"):
                model = E_UNet.run([*size, 3**(not is_grayscale)], num_classes=num_classes, dropout_const=0., is_compile=False, loss=loss,
                                   use_attention=use_attention, reduce_const=reduce_const, learning_rate=learning_rate)
                model = SemanticSegmentation.unet_available_metric(model, "sphereface",
                                                                   [*size, 3**(not is_grayscale)], num_classes, loss, is_fusion_face=is_fusion_face,
                                                                   freeze_classifier=False, nullfication_metric=nullfication_metric, dropout_const=0.,
                                                                   label_smoothing=label_smoothing, is_eunet=("e-unet" in model_name),
                                                                   eunet_metric_mode=eunet_metric_mode, eunet_metric_subcontext=eunet_metric_subcontext)
                if not is_h5:
                    weights = "/".join(weights.split("/")[:-1])
                    weights = weights.replace("metric_classifier", metric_func)
                    # cp.cprint("origin model", "red")
                    # model.summary()
                    # model = load_model(weights)
                    # cp.cprint(weights, "pink")
                    # model.summary();return None
                model.load_weights(weights)
                model = SemanticSegmentation.unet_learning_classifier(model, num_classes=num_classes)
                is_load_weight = False
                
            else: cp.cprint(f"[!] {model_name} is not defined.", "red")
            
            ## 重みの読み込み
            if is_load_weight:
                if not is_h5:
                    weights = "/".join(weights.split("/")[:-1])
                model.load_weights(weights)
                cp.cprint(f"@ Loaded weights : {weights}", "orange")
            
            ## Metric Learning
            if (model_name in ["unet_arcface", "unet_cosface", "unet_sphereface", "e-unet_arcface", "e-unet_cosface", "e-unet_sphereface"]):
                model = SemanticSegmentation.unet_available_metric(model, model_name.replace("unet_", "").replace("e-unet_", ""),
                                                                   [*size, 3**(not is_grayscale)], num_classes, loss, is_fusion_face=is_fusion_face,
                                                                   freeze_classifier=True, nullfication_metric=nullfication_metric,
                                                                   dropout_const=dropout_const, label_smoothing=label_smoothing,is_eunet=("e-unet" in model_name),
                                                                   eunet_metric_mode=eunet_metric_mode, eunet_metric_subcontext=eunet_metric_subcontext)
            
            ## HoG特徴量を使用する際に入力層を入れ替える
            # if is_HoG:
                # model = HoG_Model(model)
                
            ## Afterの判定を含めたモデルに変換
            if (classification == "before-just-after"):
                model = After.convert_v0(model, loss=loss, learning_rate=learning_rate, optimizer=optimizer)
        

            if (reduce_const < 1):
                
                cp.cprint(f"@ Pruning with filters ( reduce_const : {reduce_const} )", "pink")
                if fold:
                    path = f"e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning_fold{fold}_576x576"
                    cp.cprint(f"@ origin model path : {path}", "pink")
                    
                    full_path = f"/workspace/fullframe/result/540x540/{path}"
                    
                    model_origin = E_UNet.run((576, 576, 3), num_classes=4, is_compile=False, reduce_const=1)
                    model_origin.load_weights(full_path)
                    
                    model = MyPruning.prune_tuning(model_before=model_origin, model_after=model, reduce=reduce)
                    
            
        return model
    