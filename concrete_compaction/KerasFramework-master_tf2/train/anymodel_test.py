from pickletools import optimize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


from mymodel.MyModel import MyModel

# my module
from mymodel.SemanticSegmentation import SemanticSegmentation
from colorPrint import Cprint as cp


# tf.debugging.set_log_device_placement(True)
  

## ================ config ===================


# WORKSPACE_DIR = "/workspace/semanticSegmentation"
# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
# WORKSPACE_DIR = "/workspace/osada_ws/work_220210_nin"
WORKSPACE_DIR = "/workspace/fullframe"
# DATASET_PATH = "/workspace/Dataset/image_dataset220210/all.txt"
DATASET_PATH = "/workspace/Dataset/fullframe/text_dataset/all.txt"

# BATCH_SIZE : 1 only
BATCH_SIZE = 1
# EPOCHS = 10

# FOLD = 5
MODEL = SemanticSegmentation.unet
SIZE = [270*2, 270*2]
# SIZE = [270, 270]
NUM_CLASSES = 2
CLASSES = ["before", "just"]
IS_MLT = False

RANDOM_SEED = 1
LIMIT = None
USE_MULTI_GPU = False
IS_RESIZE = False
BUF = ""

IS_OUTPUT_RESULT = False
IS_OUTPUT_DETAIL_RESULT = False

# LOAD_PATH = "/workspace/semanticSegmentation/result"
LOAD_PATH = "/workspace/fullframe/result/540x540/unet_20220319_AutoLearning_fold1_540x540"
# LOAD_PATH = "/workspace/osada_ws/work_220210_nin"
LOAD_ID = "unet_20220319_AutoLearning_fold1_540x540"
RESULT_ID = f"{LOAD_ID}_trail"

IS_FULLFRAME = True

IS_VISUALIZE = False


## ===========================================



blank = lambda variable : cp.cprint(f"[!] \"{variable}\" is blank.", "red")

if IS_OUTPUT_RESULT:
    with open(f"{LOAD_PATH}/{RESULT_ID}.txt", mode="w") as logoutput: pass
if IS_OUTPUT_DETAIL_RESULT:
    with open(f"{LOAD_PATH}/{RESULT_ID}_correct.txt", mode="w") as logoutput: pass
    with open(f"{LOAD_PATH}/{RESULT_ID}_uncorrect.txt", mode="w") as logoutput: pass
    with open(f"{LOAD_PATH}/{RESULT_ID}_combined.txt", mode="w") as logoutput: pass
def output(strings:list):
    with open(f"{LOAD_PATH}/{RESULT_ID}.txt", mode="a") as logoutput:
        print(strings, file=logoutput)
def output_anyfile(strings:list, filename:str):
    if IS_OUTPUT_DETAIL_RESULT:
        with open(f"{LOAD_PATH}/{RESULT_ID}_{filename}.txt", mode="a") as logoutput:
            print(strings, file=logoutput)

# def get_palette():
#     palette = [[0, 0, 0],
#                [255, 255, 255]]
#     return np.asarray(palette)


# def adjustData(img, mask):
#     if IS_FULLFRAME:
#         img_buf = [None]*(img.shape[0])
#         mask_buf = [None]*(mask.shape[0])
#         for index, (i, m) in enumerate(zip(img, mask)):
#             img_buf[index] = Image.fromarray(np.uint8(i))
#             mask_buf[index] = Image.fromarray(np.uint8(m))
#             img_buf[index].resize(SIZE)
#             mask_buf[index].resize(SIZE)
#             img[index] = np.asarray(img_buf[index])
#             mask[index] = np.asarray(mask_buf[index])
    
#     if(np.max(img) > 1):
#         img = img / 255.

#     palette = get_palette()

#     onehot = np.zeros((mask.shape[0], *SIZE, NUM_CLASSES), dtype=np.uint8)
#     for i in range(2):
#         cat_color = palette[i]
#         temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
#                         (mask[:, :, :, 1] == cat_color[1]) &
#                         (mask[:, :, :, 2] == cat_color[2]), 1, 0)
#         onehot[:, :, :, i] = temp

#     return img, onehot


def fetch_average_image(isGrayScale=False):
    """
    @fn     fetch_average_image()
    @brief  平均画像の読み込み.
    @return pil形式の画像を255で割ったもの
    """

    # 差分画像(とその前処理)
    # aveimg_path = '/media/nagalab/Volume01/kojima_ws/concrete_compaction/average_image_0516.png'
    aveimg_path = '/workspace/osada_ws/average_image_0516.png'

    if isGrayScale:
        ave_img = img_to_array(load_img(aveimg_path, color_mode='grayscale', grayscale=True))
    else:
        ave_img = img_to_array(load_img(aveimg_path, color_mode='rgb'))

    return ave_img


def dataGenerator(resourcepath):
    
    # image_path, mask_path = [], []
    with open(resourcepath) as f:
        readlines = f.readlines()
        random.seed(RANDOM_SEED)
        random.shuffle(readlines)
        if LIMIT: readlines = readlines[:LIMIT]
    image_data = []
    answer_data = []
    fresh_data = []
    for line in readlines:
        linebuffer = line.split(" ")
        length = len(linebuffer)
        if not length: continue
        linebuffer[length-1] = linebuffer[length-1].rstrip("\n")
        image_data.append(linebuffer[0].replace("/media/nagalab/SSD1.7TB/nagalab", "/workspace"))
        answer_data.append(int(linebuffer[1]))
        fresh_data.append([float(i) for i in linebuffer[2:]])
        # train_data.append([input, str(answer), fresh])
        # train_data.append([input, fresh])
    
    data_gen_args = dict(
        rescale=None
    )

    # image_datagen = ImageDataGenerator(**data_gen_args)
    # mask_datagen = ImageDataGenerator(**data_gen_args)

    # image_dataframe = pd.DataFrame(image_path, index=None, columns=["image"])
    # mask_dataframe = pd.DataFrame(mask_path, index=None, columns=["mask"])
    
    datagen = ImageDataGenerator(**data_gen_args)
    dataframe = pd.DataFrame(image_data, index=None, columns=["filename"])

    generator = datagen.flow_from_dataframe(dataframe,
                                            x_col="filename",
                                            target_size=SIZE.copy(),
                                            color_mode="rgb",
                                            classes=CLASSES,
                                            class_mode=None,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

    # image_generator = image_datagen.flow_from_dataframe(image_dataframe,
    #                                                     x_col="image",
    #                                                     target_size=SIZE.copy(),
    #                                                     color_mode="rgb",
    #                                                     classes=["before", "just"],
    #                                                     class_mode=None,
    #                                                     batch_size=BATCH_SIZE,
    #                                                     shuffle=False)
    # mask_generator = mask_datagen.flow_from_dataframe(mask_dataframe,
    #                                                   x_col="mask",
    #                                                   target_size=SIZE.copy(),
    #                                                   color_mode="rgb",
    #                                                   classes=["before", "just"],
    #                                                   class_mode=None,
    #                                                   batch_size=BATCH_SIZE,
    #                                                   shuffle=False)
    
    ave_image = fetch_average_image()
    
    for i, data in enumerate(generator):
        # print(i, data)
        # image, mask = adjustData(image, mask)
        # if(np.max(data) > 1): 
        data -= ave_image
        data /= 255.
        try:
            yield data, fresh_data[i], answer_data[i], image_data[i].replace("/workspace", "/media/nagalab/SSD1.7TB/nagalab")
        except:
            yield data, None, None, None


def datacounter(datapath):
    with open(datapath) as f:
        readlines = f.readlines()
        if LIMIT : readlines = readlines[:LIMIT]
    return len(readlines)


# def createModel(model_name:str=None):
    
#     modell = None
    
#     if model_name is None:
#         blank("model_name")
            
#     else:
#         cp.cprint(f"- model : {model_name} -", "cyan")
#         cp.cprint(f"- LOAD_ID : {LOAD_ID} -", "cyan")  
#         if (model_name == "unet"):
#             model = SemanticSegmentation.unet([*SIZE, 3])
#         else:
#             cp.cprint(f"[!] {model_name} is not defined.", "red")
        
#     return model


class calc:
    @staticmethod
    def precision(tp, fp):
        return tp/(tp+fp)

    @staticmethod
    def recall(tp, fn):
        return tp/(tp+fn)
    
    @staticmethod
    def f1_score(precision, recall):
        return 2*(precision*recall) / (precision+recall)


def test(model_name:str=None):
    # if model_name is None:
    #     blank("model_name")
    #     return None
    
    # model = createModel(model_name)
    
    output_anyfile("inputdata, answer, predict", "correct")
    output_anyfile("inputdata, answer, predict", "uncorrect")
    output_anyfile("inputdata, answer, predict", "combined")
    
    model = MODEL([*SIZE, 3], "categorical_crossentropy", True)
    # model2 = MODEL([*SIZE, 3], "categorical_crossentropy", True)
    # model3 = MODEL([*SIZE, 3], "categorical_crossentropy", True)
    model.load_weights(f"{LOAD_PATH}/{LOAD_ID}.h5")
    # model2.load_weights(f"{LOAD_PATH}/{LOAD_ID}.h5")
    # model3.load_weights(f"{LOAD_PATH}/{LOAD_ID}.h5")
    model.summary()
    # model2.summary()
    # model3.summary()
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    correct, image_accum = 0, 0    
    datacount = datacounter(DATASET_PATH)
    
    true_positive = {"before":0, "just":0}
    true_negative = {"before":0, "just":0}
    false_positive = {"before":0, "just":0}
    false_negative = {"before":0, "just":0}
    
    cp.cprint(f"\n\ncompleted : 0 / {datacount} ( 0.000 [%] )", "green")
    cp.cprint(f"accuracy  : --.--- [%]", "cyan")
    cp.cprint("\n")
    
    # bar = tqdm(total=datacount)
    # bar.set_description('Progression of predictions ')
    
    # predict_map = [[0]*SIZE[1] for _ in range(SIZE[0])]
        
    for i, (data, fresh, answer, path) in enumerate(dataGenerator(DATASET_PATH)):
        if fresh is None:
            cp.cprint(f"[!] \"fresh\" is None. ( index : {i} )\nFinish the process...", "orange")
            break
        if (i >= datacount): break
        
        # input = np.asarray([np.array(data, dtype=np.float32), np.array(fresh, dtype=np.float32)], dtype=np.float32)
        # input = pd.DataFrame(data + fresh)
        # input = dict(zip(["input_1", "input_2"],
                        #  [np.array(data, dtype=np.float32), np.array([fresh], dtype=np.float32)]))
        # input = {"input_1" : np.array(data, dtype=np.float32),
        #          "input_2" : np.array([fresh], dtype=np.float32)}
        
        # predict = model.predict(input, batch_size=BATCH_SIZE)
        
        image = np.asarray(data, np.float32)
        fresh = np.asarray(np.reshape(fresh, (1, 5)), np.float32)
        predict = model.predict_on_batch([image, fresh])
        
        if (BATCH_SIZE == 1): answer = [answer]
        for probability, ans in zip(predict, answer):
            # for j, row in enumerate(batch):
                # for k, pixel in enumerate(row):
            pred_label = int(probability[0] <= probability[1])
            # pred_label = np.argmax(probability)
            # print(pred_label, ans, pred_label==ans)
            # print(probability, ans, pred_label)
                    # if (pixel[1] > 0.90):
                    #     predict_map[j][k] = 1
                    # elif (pixel[0] > 0.90):
                        # predict_map[j][k] = 0
            if ans:
                if pred_label:
                    correct += 1
                    output_anyfile(f"{path}, {ans}, {pred_label}", "correct")
                    true_negative["before"] += 1
                    true_positive["just"] += 1
                else:
                    output_anyfile(f"{path}, {ans}, {pred_label}", "uncorrect")
                    false_positive["before"] += 1
                    false_negative["just"] += 1
            else:
                if pred_label:
                    output_anyfile(f"{path}, {ans}, {pred_label}", "uncorrect")
                    false_negative["before"] += 1
                    false_positive["just"] += 1
                else:
                    correct += 1
                    output_anyfile(f"{path}, {ans}, {pred_label}", "correct")
                    true_positive["before"] += 1
                    true_negative["just"] += 1
            image_accum += 1
            output_anyfile(f"{path}, {ans}, {pred_label}", "combined")
        if not i: print("\n\n\n\n")
            
        cp.cprint(f"\033[4Acompleted : {i} / {datacount} ( {round(i/datacount*100, 3)} [%] ){' '*50}", "green")
        cp.cprint(f"accuracy  : {round(correct/image_accum*100, 3)} [%]{' '*50}", "cyan")
        cp.cprint(f"TP : {true_positive}, TN : {true_negative}{' '*10}", "cyan")
        cp.cprint(f"FP : {false_positive}, FN : {false_negative}{' '*10}", "cyan")
        
        # bar.update(1)
    
    print()
    
    accuracy = correct/image_accum
    
    before_f1 = calc.f1_score(
        calc.precision(
            tp=true_positive["before"],
            fp=false_positive["before"]
        ),
        calc.recall(
            tp=true_positive["before"],
            fn=false_negative["before"]
        )
    )
    
    just_f1 = calc.f1_score(
        calc.precision(
            tp=true_positive["just"],
            fp=false_positive["just"]
        ),
        calc.recall(
            tp=true_positive["just"],
            fn=false_negative["just"]
        )
    )
    
    survey_keys = ["accuracy", "F1(Before)", "F1(Just)", "TP(Before)", "TN(Before)", "FP(Before)", "FN(Before)", "TP(Just)", "TN(Just)", "FP(Just)", "FN(Just)"]
    survey_values = [accuracy, before_f1, just_f1,
                     true_positive["before"], true_negative["before"], false_positive["before"], false_negative["before"],
                     true_positive["just"], true_negative["just"], false_positive["just"], false_negative["just"]]
    
    output(" ".join(survey_keys))
    output(" ".join(map(str, survey_values)))
    
    return dict(zip(survey_keys, survey_values))
    

def main():
    result = test()
    cp.cprint(f"test result : {result}", "green")
    cp.cprint("- finished ! -", "cyan")
    

if (__name__ == "__main__"):
    main()
