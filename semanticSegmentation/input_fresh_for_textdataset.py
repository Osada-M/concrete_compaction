import csv

from colorPrint import Cprint as cp


TEXT_DATASET_DIR = "/workspace/Dataset/fullframe/text_dataset"
# TEXT_DATASET_DIR = "/workspace/Dataset/semanticSegmentation/text_dataset"
ALL_TEXT_DATASET = f"{TEXT_DATASET_DIR}/all.txt"
FRESH_CSV = "/workspace/osada_ws/freshdata.csv"

IS_USE_ALLTEXT = False

IS_TRAIN_DATA_SKIP = False
SKIP_NUMBER = 5


text_path = lambda target: f"{TEXT_DATASET_DIR}/fold{target[0]}/{target[1]}{target[2]}.txt"
alltext_path = lambda target: f"{TEXT_DATASET_DIR}/all{target}.txt"


def append_fresh(fresh):
    """
    @機能：
    @引数：
    @戻値：
    """
    if IS_USE_ALLTEXT:
        try:
            with open(ALL_TEXT_DATASET, mode="r") as f:
                lines = f.readlines()
            with open(alltext_path("_include_fresh"), mode="w") as f:
                output = ""
                for l in map(lambda x: x.rstrip("\n"), lines):
                    buffer = l.split(" ")
                    while not len(buffer[0]):
                        buffer = buffer[1:]
                    image_path = buffer[0]
                    date = (image_path.split("/"))[-2]
                    if (date in ["before", "just"]):
                        date = (((image_path.split("/"))[-1]).split("_"))[0]
                        date = date[2:]
                    output += f"{l} {' '.join(fresh[date])}\n"
                f.write(output)
            print(f"all.txt {cp.colored('[ OK ]', 'green')}")
        except:
            print(f"all.txt {cp.colored('[ NG ]', 'red')}")
                
    for fold in range(1, 6):
        tmp = ""; print()
        try:
            for target in ["train", "test", "validation"]:
                with open(text_path([fold, target, ""]), mode="r") as f:
                    lines = f.readlines()
                with open(text_path([fold, target, "_include_fresh"]), mode="w") as f:
                    output = ""
                    for i, l in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                        if (target == "train") and IS_TRAIN_DATA_SKIP and (i%SKIP_NUMBER): continue
                        buffer = l.split(" ")
                        while not len(buffer[0]):
                            buffer = buffer[1:]
                        image_path = buffer[0]
                        date = (image_path.split("/"))[-2]
                        if (date in ["before", "just"]):
                            date = (((image_path.split("/"))[-1]).split("_"))[0]
                            date = date[2:]
                        output += f"{l} {' '.join(fresh[date])}\n"
                    f.write(output)
                cp.cprint(f"\033[1Acompleted : fold{fold} |{tmp} {target}.txt", "cyan")
                tmp += f" {target}.txt"
            print(f"fold{fold} {cp.colored('[ OK ]', 'green')}")
        except:
            print(f"fold{fold} {cp.colored('[ NG ]', 'red')}")


def read_fresh_csv(csv_file):
    """
    @機能：
    @引数：
    @戻値：
    """
    with open(csv_file, mode="r") as f:
        reader = [row for row in csv.reader(f)][1:]
        result = dict(zip(list(zip(*reader))[0], map(lambda x: x[1:], reader)))
    
    return result
        

if(__name__ == "__main__"):
    fresh = read_fresh_csv(FRESH_CSV)
    append_fresh(fresh)
