import cv2
import pickle

from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/semanticSegmentation"
SIZE = 270


images_path = lambda *target : f"/workspace/osada_ws/text_dataset/ngc_docker/fold{target[0]}/{target[1]}_4class.txt"
output_path = lambda *target : f"{DIR}/text_dataset/fold{target[0]}/{target[1]}_4class.txt"
mask_path = lambda class_num : f"{DIR}/masked/class{class_num}_{SIZE}.png"
targets = ["test", "train", "validation"]

# positive = f"{DIR}/masked/positive_{SIZE}.png"
# negative = f"{DIR}/masked/negative_{SIZE}.png"


def convert():
    with open("/workspace/mesh_dataset/fresh.pickle", mode="rb") as f:
        fresh_data = pickle.load(f)

    for fold in range(1, 6):
        cp.cprint(f"- fold : {fold} -", "green")
        for t in targets:
            text = images_path(fold, t)
            with open(text, mode="r") as f:
                paths = f.read().split("\n")
            length = len(paths)
            cp.cprint(f"target : {t}, length : {length}\n", "cyan")
            
            with open(output_path(fold, t), mode="w") as o: pass
            
            for i, p in enumerate(paths):
                lst = p.split(" ")
                if (len(lst) < 2): continue
                path, answer, *others = lst
                answer = int(answer)
                
                ## フレッシュ性状データの取得
                path_elements = path.split("/")
                _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
                element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
                date, place, time_id, mesh_id = element.split("_")
                key = f"{date}_{place}"
                fresh = fresh_data[key]
            
                output = f"{path} {mask_path(answer)} {' '.join(map(str, fresh))}"
            
                with open(output_path(fold, t), mode="a") as f:
                    print(output, file=f)
                    
                cp.cprint(f"\033[1Aconmpleted : {i+1}/{length}", "orange")

    print("- finished ! -")


def main():
    convert()


main()
