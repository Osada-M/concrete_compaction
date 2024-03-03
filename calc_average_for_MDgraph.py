
DIR = "/media/nagalab/SSD1.7TB/nagalab/osada_ws/fullframe/result/540x540"
# SAVE_DIR = "/Volumes/DATA/Laboratory/work/graph"
# MACBOOK = "/Users/osadamasashi/Library/CloudStorage/GoogleDrive-osadamasashi.c@gmail.com/Other computers/iMac/"

# TARGET = "e-unet_4class_adam_dropout_20220805_AutoLearning"

# TARGET = "e-unet_4class_myself_20220826_AutoLearning"
# TARGET = "e-unet_4class_adam_dropout_0.025_20220829_AutoLearning"
# TARGET = "e-unet_4class_adam_dropout_0.2_20220829_AutoLearning"
# TARGET = "e-unet_metric_classifier_4class_20220914_AutoLearning"
# TARGET = "e-unet_4class_radam_dropout_20220805_AutoLearning"
# TARGET = "e-unet_without_pre-training_b4_e2"

# TARGET = "e-unet_4class_use-AE-input_20221014-1_flip_20221103_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_20221103_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip_rotate_20221110_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip-ud_rotate_20221110_AutoLearning"

# TARGET = "e-unet_after_use-AE-input_20221014_ssim_mse_flip_rotate_20221116_AutoLearning"
TARGET = "e-unet_after_use-AE-input_20221014_ssim_mse_20221122_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip-lr_rotate_20221110_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_flip-lr_20221122_AutoLearning"
# TARGET = "e-unet_4class_AllInOne_20221129_AutoLearning"
# TARGET = "e-unet_4class_use-AE-input_20221014_ssim_mse_rotate_limit-90-90_20221129_AutoLearning"

KEY = ["loss", "acc", "val_loss", "val_acc"]
COMPALATE = ["loss", "val_loss"]

# IS_MACBOOK = 1
IS_IOU = True
# SIZE = "540x540"
SIZE = "576x576"
IS_FOURCLASSES = 1
IS_QUANTIZED = 0
IS_FOURCLASSES_ACCURACY = 0

IS_AFTERCLASS = 0
AFTER_TEXTNAME = 1

if IS_FOURCLASSES_ACCURACY:
    BUF_KEY = ["acc", "f1_b", "f1_b-b", "f1_b-j", "f1_j",
            "tp_b", "tn_b", "fn_b", "fp_b",
            "tp_b-b", "tn_b-b", "fn_b-b", "fp_b-b",
            "tp_b-j", "tn_b-j", "fn_b-j", "fp_b-j",
            "tp_j", "tn_j", "fn_j", "fp_j",
            "iou_b", "iou_b-b", "iou_b-j", "iou_j"]
elif IS_AFTERCLASS:
    BUF_KEY = ["acc", "f1_b", "f1_j", "f1_a",
            "tp_b", "tn_b", "fn_b", "fp_b",  
            "tp_j", "tn_j", "fn_j", "fp_j",
            "tp_a", "tn_a", "fn_a", "fp_a",
            "iou_b", "iou_j", "iou_a"]
else:
    BUF_KEY = ["fold", "acc", "f1b", "f1j", "ioub", "iouj"]

if not IS_IOU: del BUF_KEY[-2:]
# if IS_MACBOOK:
#     DIR = DIR.replace("/Volumes/", MACBOOK)
#     SAVE_DIR = SAVE_DIR.replace("/Volumes/", MACBOOK)

make_path = lambda fold: f"{DIR}/{TARGET}_fold{fold[0]}_{SIZE}/{TARGET}_fold{fold[0]}_{SIZE}{'_quantized' if IS_QUANTIZED else ''}_testResult{fold[1]}{'_4classes' if IS_FOURCLASSES_ACCURACY else ''}{'_after_3classes' if IS_AFTERCLASS or AFTER_TEXTNAME else ''}.txt"


def extract():
    
    remove_index = [[None], [None]]
    
    if IS_FOURCLASSES_ACCURACY or IS_AFTERCLASS:
        numbers_data = dict()
        for i, m_key in enumerate(["pixel", "mesh"]):
            numbers_data[m_key] = dict()
            for f in range(1, 6):
                with open(make_path([f, ["", "_judgeMesh"][i]]), mode="r") as txt:
                    result = txt.read()
                    if len(result):
                        buf = result.split("\n")[1]
                        for key, val in zip(BUF_KEY, buf.split(" ")):
                            if not key in numbers_data[m_key].keys(): numbers_data[m_key][key] = 0
                            numbers_data[m_key][key] += float(val.replace("nan", "0"))/5.
                    else:
                        remove_index[i].append(i)
                        
        return numbers_data, remove_index
        
    else:
        data = [None]*10
        numbers_data = dict()
        for f in range(1, 6):
            for i, path_buf in enumerate(["", "_judgeMesh"]):
                try:
                    with open(make_path([f, path_buf]), mode="r") as txt:
                        result = txt.read()
                        if len(result):
                            buf = (result.split("\n"))[2]
                            buf = ((buf.split(": "))[1]).split("|")
                            del buf[0], buf[-1]
                            origin_data = dict()
                            for i, key in enumerate(BUF_KEY):
                                key += path_buf
                                if (key == f"fold{path_buf}"):
                                    origin_data[key] = f"FOLD{f}"
                                elif (key == f"None{path_buf}"):
                                    origin_data[str(f"None_{i}{path_buf}")] = f"-"
                                else:
                                    origin_data[key] = float(buf[i].replace("nan", "0"))
                                    numbers_data[key] = numbers_data[key]+(float(buf[i])/5) if key in numbers_data.keys() else float(buf[i])/5
                            data[f-1+(5*(len(path_buf)>0))] = origin_data
                        else:
                            remove_index[i].append(f-1)
                except:
                    remove_index[i].append(f-1)
            
            # if len(remove_index):
            #     for 
        
        # print(data)

        return data, numbers_data, remove_index


def main():
    # try: os.mkdir(f"{SAVE_DIR}/{TARGET}")
    # except: pass
    
    if IS_FOURCLASSES_ACCURACY:
        numbers, removes = extract()
        
        classes = ["_b", "_b-b", "_b-j", "_j"]
        data_buf = ["Acc", "Before", "B-Before", "B-Just", "Just"]
        data = {"mesh":dict(zip(data_buf, [None]*len(data_buf))), "pixel":dict(zip(data_buf, [None]*len(data_buf)))}
        matrix = ""
        
        for index, m_key in enumerate(["pixel", "mesh"]):
            for key, val in numbers[m_key].items():
                numbers[m_key][key] = val*5/(5-len(removes[index])+1)
            data[m_key]["Acc"] = round(numbers[m_key]["acc"], 5)
            for data_key, class_key in zip(data_buf[1:], classes):
                data[m_key][data_key] = {"f1" : numbers[m_key][f"f1{class_key}"],
                                         "tp" : numbers[m_key][f"tp{class_key}"],
                                         "tn" : numbers[m_key][f"tn{class_key}"],
                                         "fp" : numbers[m_key][f"fp{class_key}"],
                                         "fn" : numbers[m_key][f"fn{class_key}"],
                                         "iou" : numbers[m_key][f"iou{class_key}"]}
                for key, val in data[m_key][data_key].items():
                    data[m_key][data_key][key] = round(val, 5)
        
            matrix += f"""\
{m_key}(4class)
|-|Before|B-Before|B-Just|Just|
|:--:|:--:|:--:|:--:|:--:|
|Accuracy|**{data[m_key]["Acc"]}**||||
|F1|{data[m_key]["Before"]["f1"]}|{data[m_key]["B-Before"]["f1"]}|{data[m_key]["B-Just"]["f1"]}|{data[m_key]["Just"]["f1"]}|
|TP|{data[m_key]["Before"]["tp"]}|{data[m_key]["B-Before"]["tp"]}|{data[m_key]["B-Just"]["tp"]}|{data[m_key]["Just"]["tp"]}|
|TN|{data[m_key]["Before"]["tn"]}|{data[m_key]["B-Before"]["tn"]}|{data[m_key]["B-Just"]["tn"]}|{data[m_key]["Just"]["tn"]}|
|FP|{data[m_key]["Before"]["fp"]}|{data[m_key]["B-Before"]["fp"]}|{data[m_key]["B-Just"]["fp"]}|{data[m_key]["Just"]["fp"]}|
|FN|{data[m_key]["Before"]["fn"]}|{data[m_key]["B-Before"]["fn"]}|{data[m_key]["B-Just"]["fn"]}|{data[m_key]["Just"]["fn"]}|
|IoU|{data[m_key]["Before"]["iou"]}|{data[m_key]["B-Before"]["iou"]}|{data[m_key]["B-Just"]["iou"]}|{data[m_key]["Just"]["iou"]}|

"""
        
        print(matrix)
        
    if IS_AFTERCLASS:
        numbers, removes = extract()
        
        classes = ["_b", "_j", "_a"]
        data_buf = ["Acc", "Before", "Just", "After"]
        data = {"mesh":dict(zip(data_buf, [None]*len(data_buf))), "pixel":dict(zip(data_buf, [None]*len(data_buf)))}
        matrix = ""
        
        for index, m_key in enumerate(["pixel", "mesh"]):
            for key, val in numbers[m_key].items():
                numbers[m_key][key] = val*5/(5-len(removes[index])+1)
            data[m_key]["Acc"] = round(numbers[m_key]["acc"], 5)
            for data_key, class_key in zip(data_buf[1:], classes):
                data[m_key][data_key] = {"f1" : numbers[m_key][f"f1{class_key}"],
                                         "tp" : numbers[m_key][f"tp{class_key}"],
                                         "tn" : numbers[m_key][f"tn{class_key}"],
                                         "fp" : numbers[m_key][f"fp{class_key}"],
                                         "fn" : numbers[m_key][f"fn{class_key}"],
                                         "iou" : numbers[m_key][f"iou{class_key}"]}
                for key, val in data[m_key][data_key].items():
                    data[m_key][data_key][key] = round(val, 5)
        
            matrix += f"""\
{m_key}(3class)
|-|Before|Just|After|
|:--:|:--:|:--:|:--:|
|Accuracy|**{data[m_key]["Acc"]}**||||
|F1|{data[m_key]["Before"]["f1"]}|{data[m_key]["Just"]["f1"]}|{data[m_key]["After"]["f1"]}|
|TP|{data[m_key]["Before"]["tp"]}|{data[m_key]["Just"]["tp"]}|{data[m_key]["After"]["tp"]}|
|TN|{data[m_key]["Before"]["tn"]}|{data[m_key]["Just"]["tn"]}|{data[m_key]["After"]["tn"]}|
|FP|{data[m_key]["Before"]["fp"]}|{data[m_key]["Just"]["fp"]}|{data[m_key]["After"]["fp"]}|
|FN|{data[m_key]["Before"]["fn"]}|{data[m_key]["Just"]["fn"]}|{data[m_key]["After"]["fn"]}|
|IoU|{data[m_key]["Before"]["iou"]}|{data[m_key]["Just"]["iou"]}|{data[m_key]["After"]["iou"]}|

"""
        
        print(matrix)
        
    else:
        data, numbers, removes = extract()
        dim = len(BUF_KEY)
        key = iter(["pixel", "mesh"])
        index = -1
        
        for f, d in enumerate(data):
            if not f%5:
                print(f"\n{next(key)}")
                print("|item|Acc|F1(B)|F1(J)|IoU(B)|IoU(J)|")
                print("|:--:|:--:|:--:|:--:|:--:|:--:|")
                index += 1
                
            if f%5 in removes[index]: continue
            
            f += 1
            markdown = "|".join(map(lambda x: f"{x[1]}{'0'*(6-len(x[1])+(x[0]>1)) if 0 < x[0] < 4 else ''}", enumerate(list(map(str, d.values())))))
            print(f"|{markdown}|")
        
            if not f%5 and f:
                print(f"|AVERAGE|**{'**|**'.join(map(lambda x: str(round(x*5/(5-len(removes[index])+1), 5)), list(numbers.values())[(dim-1)*((f//5)-1):(dim-1)*(f//5)]))}**|")
                if (f<=5): print(f"|Others|||||{round((numbers['ioub'] + numbers['iouj']) / 2, 5)}|")
                else: print(f"|Others|||||{round((numbers['ioub_judgeMesh'] + numbers['iouj_judgeMesh']) / 2, 5)}|")
        
        print("\n==========\n")
        
        for f, d in enumerate(data):
            f += 1
            if not f%5 and f:
                mat = list(map(lambda x: round(x*5/(5-len(removes[index])+1), 5), list(numbers.values())[(dim-1)*((f//5)-1):(dim-1)*(f//5)]))
                print(f"|AVERAGE|{'|'.join(map(str, mat))}|")
                
                acc, f1_b, f1_j, iou_b, iou_j = mat
                print(f"Acc : {acc}, F1 : {round(f1_b, 5)}, {round(f1_j, 5)}, IoU : {round(iou_b, 5)}, {round(iou_j, 5)}")
                print(f"Acc : {acc}, Mean-F1 : {round((f1_b + f1_j) / 2, 5)}, Mean-IoU : {round((iou_b + iou_j) / 2, 5)}\n")


main()
