import numpy as np
import cv2

# my module
from colorPrint import Cprint as cp

echo = lambda *string : cp.cprint(", ".join(map(str, string)), "cyan")
print = lambda *string: cp.cprint(" ".join(map(str, string)), color="cyan")
# is_docker = True



def main():
    save_dir = "/workspace/luminance/edge_images/"

    texts = {"test" : "/workspace/osada_ws/dataset_generation/dataset_create/fold1/test.txt",
             "train" : "/workspace/osada_ws/dataset_generation/dataset_create/fold1/train.txt",
             "validation" : "/workspace/osada_ws/dataset_generation/dataset_create/fold1/validation.txt"}
    
    threshold = [50, 20]
    
    paths = list()
    
    for target in texts.keys():
        with open(texts[target], mode="r", encoding="utf8") as f:
            paths += list(map(lambda x : (x.split(" "))[0], f.read().split("\n")))
 
    edge_extraction(paths, threshold, save_dir)
    
    print("finished !")


# エッジ抽出
def edge_extraction(paths:list, threshold:list, save_dir):
    length = len(paths)-1
    print(f"th0 = {threshold[0]}, th1 = {threshold[1]}\n")
    for i, target in enumerate(paths):
        # if (i>2) : break
        try:
            if not target:
                cp.cprint(f"[!] warning : No.{i}, this path is blank\n", "orange")
                continue
            img = cv2.imread(target, 0)
            img = cv2.Canny(img, threshold[0], threshold[1])
            cv2.imwrite(f"{save_dir}edge_{i}.png", img)
            print(f"\033[1Aconpleted : {i} / {length}")
            
        except KeyboardInterrupt:
            cp.cprint("[!] accepted KeyboarInterrupt", "orange")
            break
        
        except:
            cp.cprint("[!] error", "red")
            cp.cprint(f"  > index : {i}", "red")
            cp.cprint(f"  > path  : {target}\n", "red")
    print()

if (__name__ == "__main__"):
    main()
