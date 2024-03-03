
## my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from line_sender import send_master
from image_path_timer import ImagePathTime as IPT


ALL_TEXT = "/workspace/mesh_dataset/all_4class.txt"
SPLIT_MATRIX = [0, 1, 2, 2, 2, 0, 1, 2, 2, 2]
SPLIT_KEY = ["test", "validation", "train"]


mask_image = lambda answer: f"/workspace/Dataset/semanticSegmentation/masked/class{answer}_270.png"
ngc_docker = lambda target: f"/workspace/mesh_dataset/fold{target[0]}/{target[1]}_4class_myself.txt"
semseg = lambda target: f"/workspace/Dataset/semanticSegmentation/text_dataset/fold{target[0]}/{target[1]}_4class_myself.txt"


def right_rotate(array:list):
    return [array[-1]] + array[:-1]


def adjust_matrix(fold:int):
    matrix = SPLIT_MATRIX.copy()
    for i in range(fold):
        matrix = right_rotate(matrix)
    
    return matrix


def diside_class():
    
    images = []
    answers = []
    freshes = []
    
    with open(ALL_TEXT, mode="r") as f:
        data = f.readlines()
    length = len(data); print()
    
    for i, line in enumerate(map(lambda x: x.rstrip("\n"), data)):
        buf = line.split(" ")
        if len(buf):
            images.append(buf[0])
            answers.append(int(buf[1]))
            freshes.append(" ".join(buf[2:]))
    
        print(f"\033[1A{i+1} / {length}")
    
    length = len(images)
    print()
    
    for fold in range(5):
        
        matrix = adjust_matrix(fold)
        fold += 1
        count = -1
        
        for target in SPLIT_KEY:
            with open(ngc_docker([fold, target]), mode="w"): pass
            with open(semseg([fold, target]), mode="w"): pass
        
        for i, (img, ans, frs) in enumerate(zip(images, answers, freshes)):
            
            image_index = int((img.replace(".jpg", "").replace("ab", "").split("/")[-1]).split("_")[1]) - 1
            target = SPLIT_KEY[matrix[image_index]]
            
            count += target == "train"
            if (target == "train") and (count%5): continue
            
            msk = mask_image(ans)
            
            with open(ngc_docker([fold, target]), mode="a") as ngc:
                with open(semseg([fold, target]), mode="a") as sem:
                    print(f"{img} {ans} {frs}", file=ngc)
                    print(f"{img} {msk} {frs}", file=sem)
    
            cp.cprint(f"\033[1Afold {fold}, {i+1} / {length}", "green")
        
        print()


def main():
    diside_class()


if(__name__ == "__main__"):
    main()