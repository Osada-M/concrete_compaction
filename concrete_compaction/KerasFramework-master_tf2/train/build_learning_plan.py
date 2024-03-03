import datetime
from enum import auto


## ================ config ================


_TRAIN = "/workspace/train/SemanticSegmentationTrain.py"
_TEST = "/workspace/train/SemanticSegmentationTest.py"
_MESH = "/workspace/semanticSegmentation/result"
_FULLFRAME = "/workspace/fullframe/result/540x540"
_KL = "kullback_leibler_divergence"
_CROSS = "categorical_crossentropy"

## ========================================


class Shell:
    """
    @機能：
    @引数：
    @戻値：void
    """
    
    TEST = _TEST
    TRAIN = _TRAIN
    MESH = _MESH
    FULLFRAME = _FULLFRAME
    KL = _KL
    CROSS = _CROSS
    
    
    def __init__(self):
        # self.textpath = textpath
        # with open(textpath, "w") as f: pass
        pass


    def encode(self, params:dict):
        """
        @機能：
        @引数：
        @戻値：
        """
        if (list(params.keys())[0] == "indention"):
            return params["indention"]
        else:
            string = " ".join(map(str, params.values()))
            string = string.replace("$test", self.TEST)
            string = string.replace("$train", self.TRAIN)
            string = string.replace("$mesh", self.MESH)
            string = string.replace("$fullframe", self.FULLFRAME)
            string = string.replace("$kl", self.KL)
            string = string.replace("$cross", self.CROSS)

            return f"python {string}"
    
    
    def output_shellscript(self, path:str, params:dict):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        with open(path, "a") as sh:
            print(self.encode(params), file=sh)
            
    
    def output_params(self):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        return self.params
    
    
    @staticmethod
    def new_file(path:str):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        with open(path, "w") as f:
            now = datetime.datetime.now()
            f.write(f"# Created at {now.strftime('%Y/%m/%d %H:%M:%S')}.\n")
    
    
    def add_indention(self, path:str, string:str=""):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        with open(path, "a") as sh:
            print(string, file=sh)


class Train(Shell):
    """
    @機能：
    @引数：
    @戻値：
    """    
    
    def __init__(self, **params):
        self.params = dict()
        self.set_params(**params)
    
    
    def set_params(self,
                   is_autotrain:bool=True,
                   auto_train_acc_limit:int=93,
                   batch_size:int=2,
                   epochs:int=20,
                   fold:int=1,
                   network_name:str="unet",
                   save_path:str="$fullframe",
                   save_id:str="none",
                   is_use_fullframe:bool=True,
                   is_use_fresh:bool=False,
                   is_load_weight:bool=True,
                   load_weight_path:str="none",
                   is_extend_luminance:bool=False,
                   is_grayscale:bool=False,
                   is_use_BCL:bool=False,
                   loss:str="$cross",
                   optimizer:str="adam",
                   is_h5:bool=True,
                   is_use_metric:bool=False,
                   metric_func:str="sphereface",
                   is_load_fullframe_weight:bool=False,
                   output_layer_name:str="classifier",
                   is_fusion_face:bool=False,
                   nullfication_metric:bool=False,
                   dropout_const:float=0.,
                   label_smoothing:float=0.,
                   norm:str="batch_norm",
                   use_attention:bool=False,
                   classification:str="before-just",
                   multi_losses:bool=False,
                   fourclasses_type:str="default",
                   eunet_metric_mode:str="conv1333",
                   eunet_metric_subcontext:str="default",
                   color_type:str="rgb",
                   normalization:str="default",
                   use_AE_input:bool=False,
                   noise_type:bool="linear",
                   autoencoder_loss:str="ssim",
                   AE_model_id:str="20221014_ssim",
                   is_flip:bool=False,
                   flip_list:list=[0, 0, 1, 1, 2, 3],
                   is_rotate:bool=False,
                   rotate_degrees:list=[[0, 359]],
                   is_enlarge:bool=False,
                   reduce_const:float=1,
                   learning_rate:float=1e-3,
                   rotate_rate:float=0.2,
                   all_in_one:bool=False,
                   reduce:str="ssim",
                   ):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        self.params["mode"] = "$train"
        self.params["epochs"] = epochs
        self.params["is_autotrain"] = int(is_autotrain)
        self.params["auto_train_acc_limit"] = auto_train_acc_limit
        self.params["batch_size"] = batch_size
        self.params["fold"] = fold
        self.params["network_name"] = network_name
        self.params["save_path"] = save_path
        self.params["save_id"] = save_id
        self.params["is_use_fullframe"] = int(is_use_fullframe)
        self.params["is_use_fresh"] = int(is_use_fresh)
        self.params["is_load_weight"] = int(is_load_weight)
        self.params["load_weight_path"] = load_weight_path
        self.params["is_extend_luminance"] = int(is_extend_luminance)
        self.params["is_grayscale"] = int(is_grayscale)
        self.params["is_use_BCL"] = int(is_use_BCL)
        self.params["loss"] = loss
        self.params["optimizer"] = optimizer
        self.params["is_h5"] = int(is_h5)
        self.params["is_use_metric"] = int(is_use_metric)
        self.params["metric_funcs"] = metric_func
        self.params["is_load_fullframe_weight"] = int(is_load_fullframe_weight)
        self.params["output_layer_name"] = output_layer_name
        self.params["is_fusion_face"] = int(is_fusion_face)
        self.params["nullfication_metric"] = int(nullfication_metric)
        self.params["dropout_const"] = float(dropout_const)
        self.params["label_smoothing"] = float(label_smoothing)
        self.params["norm"] = norm
        self.params["use_attention"] = int(use_attention)
        self.params["classification"] = classification
        self.params["multi_losses"] = int(multi_losses)
        self.params["fourclasses_type"] = fourclasses_type
        self.params["eunet_metric_mode"] = eunet_metric_mode
        self.params["eunet_metric_subcontext"] = eunet_metric_subcontext
        self.params["color_type"] = color_type
        self.params["normalization"] = normalization
        self.params["use_AE_input"] = int(use_AE_input)
        self.params["noise_type"] = noise_type
        self.params["autoencoder_loss"] = autoencoder_loss
        self.params["AE_model_id"] = AE_model_id
        self.params["is_flip"] = int(is_flip)
        self.params["flip_list"] = ",".join(map(str, flip_list))
        self.params["is_rotate"] = int(is_rotate)
        self.params["is_enlarge"] = int(is_enlarge)
        self.params["reduce_const"] = reduce_const
        self.params["learning_rate"] = learning_rate
        self.params["rotate_rate"] = rotate_rate
        self.params["rotate_degrees"] = "-".join(map(lambda x: ','.join(map(str, x)), rotate_degrees))
        self.params["all_in_one"] = int(all_in_one)
        self.params["reduce"] = reduce

        self._train_encode()
        
    
    def _train_encode(self):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        if self.params["is_autotrain"]:
            self.params["epochs"] = "auto"
            del self.params["batch_size"]
        else:
            del self.params["auto_train_acc_limit"]
        del self.params["is_autotrain"]
        
        # if self.params["is_use_fullframe"]:
            # self.params["load_weight_path"] += "_540x540"


class Test(Shell):
    """
    @機能：
    @引数：
    @戻値：
    """    
    
    def __init__(self, **params):
        self.params = dict()
        self.set_params(**params)
    
    
    def set_params(self,
                   fold:int=1,
                   network_name:str="unet",
                   load_path:str="$fullframe",
                   load_id:str="none",
                   size:list=(540, 540),
                   is_use_fullframe:bool=True,
                   is_use_fresh:bool=False,
                   is_judgement_by_mesh:bool=True,
                   is_grayscale:bool=False,
                   is_fusion_face:bool=False,
                   norm:str="batch_norm",
                   is_h5:bool=False,
                   is_use_averageimage:bool=True,
                   use_attention:bool=False,
                   classification:str="before-just",
                   is_quantized:bool=False,
                   do_fourclasses_test:bool=False,
                   multi_losses:bool=False,
                   fourclasses_type:str="default",
                   is_use_LE:bool=False,
                   LE_mode:str="none",
                   LE_const:int=0,
                   use_custom_loss:bool=False,
                   loss:str="$cross",
                   color_type:str="rgb",
                   normalization:str="default",
                   use_AE_input:bool=False,
                   noise_type:bool="linear",
                   autoencoder_loss:str="ssim",
                   AE_model_id:str="20221014_ssim",
                   do_threeclasses_test:bool=False,
                   ):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        self.params["mode"] = "$test"
        self.params["fold"] = fold
        self.params["network_name"] = network_name
        self.params["load_path"] = load_path
        self.params["load_id"] = load_id
        self.params["row"] = size[0]
        self.params["col"] = size[1]
        self.params["is_use_fullframe"] = int(is_use_fullframe)
        self.params["is_use_fresh"] = int(is_use_fresh)
        self.params["is_judgement_by_mesh"] = int(is_judgement_by_mesh)
        self.params["is_grayscale"] = int(is_grayscale)
        self.params["is_fusion_face"] = int(is_fusion_face)
        self.params["norm"] = norm
        self.params["is_h5"] = int(is_h5)
        self.params["is_use_averageimage"] = int(is_use_averageimage)
        self.params["use_attention"] = int(use_attention)
        self.params["classification"] = classification
        self.params["is_quantized"] = int(is_quantized)
        self.params["do_fourclasses_test"] = int(do_fourclasses_test)
        self.params["multi_losses"] = int(multi_losses)
        self.params["fourclasses_type"] = fourclasses_type
        self.params["is_use_LE"] = int(is_use_LE)
        self.params["LE_mode"] = LE_mode
        self.params["LE_const"] = int(LE_const)
        self.params["use_custom_loss"] = int(use_custom_loss)
        self.params["loss"] = loss
        self.params["color_type"] = color_type
        self.params["normalization"] = normalization
        self.params["use_AE_input"] = int(use_AE_input)
        self.params["noise_type"] = noise_type
        self.params["autoencoder_loss"] = autoencoder_loss
        self.params["AE_model_id"] = AE_model_id
        self.params["do_threeclasses_test"] = int(do_threeclasses_test)

        self._test_encode()
        
    
    def _test_encode(self):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        self.params["load_id"] += f"_{self.params['row']}x{self.params['col']}" if self.params["is_use_fullframe"] else ""


class Indention(Shell):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    def __init__(self, string:str=""):
        self.params = {"indention": string}


class RemoveCaches(Shell):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    def __init__(self):
        self.params = {"indention": "sh ~/rm_caches.sh"}


class CatInBox(Shell):
    """
    @機能：
    @引数：
    @戻値：
    """
    
    def __init__(self, string:str="Meow!"):
        cat = """\
# %s
#           /＼＿＿／ヽ
#         ／_ノ||||ヽ､_＼
#       ／ oﾟ⌒    ⌒ﾟo    ＼
#      |　三（__人__）三   |＿＿
#    ／ ＼    ``ー'´     ／   ／|
#  ／    ／⌒))   ((⌒ヽ      ／
# |￣￣￣￣￣￣￣￣￣￣￣￣|
"""%(string)

        self.params = {"indention": cat}


class WithoutReplace:
    """
    @機能：
    @引数：
    @戻値：
    """
    
    def __init__(self,
                 arcface:bool=True,
                 cosface:bool=True,
                 sphereface:bool=True,
                 only_classifier:bool=True,
                 fourclass:bool=True,
                 after:bool=True,
                 ):
        self.without = {
            "arcface" : arcface,
            "cosface" : cosface,
            "sphereface" : sphereface,
            "only_classifier" : only_classifier,
            "4class" : fourclass,
            "after" : after,
        }
    
    
    def network_name(self, name:str):
        """
        @機能：
        @引数：
        @戻値：
        """
        
        for key, value in self.without.items():
            if value:
                name = name.replace(f"_{key}", "")
                name = name.replace(key, "")
        
        return name