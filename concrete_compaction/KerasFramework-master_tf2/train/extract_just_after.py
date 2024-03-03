from PIL import Image
import numpy as np

from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/fullframe/text_dataset"
CLASSES = [0, 128, 255]


def make_2class():
    
    for buf in ["", "_rectified"]:
        for fold in range(1, 6):
            cp.cprint(f"fold {fold}", "orange")
            for target in ["train", "validation", "test"]:
                
                print()
                
                text = f"{DIR}/fold{fold}/{target}_after{buf}.txt"
                with open(text) as f:
                    lines = f.readlines()
                    
                length = len(lines)
                new_length = 0
                
                text = f"{DIR}/fold{fold}/{target}_after_wide{buf}.txt"
                with open(text, mode="w") as f:
                    for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                        
                        cp.cprint(f"\033[1A > {target} : {i+1} / {length}, {new_length}{' '*20}", "pink")
                        
                        val = line.split(" ")
                        img, msk, *frh = val
                        msk_img = np.array(Image.open(msk), dtype=np.uint8)
                        
                        ## include
                        if CLASSES[1] in msk_img or CLASSES[2] in msk_img:
                            f.write(f"{line}\n")
                            new_length += 1
                

def sarvey():
    
    mesh = None
    result = np.zeros((25), dtype=np.uint64)
    
    sarvey_text = f"{DIR}/after_wide_sarvey.txt"
    
    with open(sarvey_text, mode="w"): pass
    
    for fold in range(1, 6):
        cp.cprint(f"fold {fold}", "orange")
        print()
            
        after = np.zeros(25, dtype=np.uint64)
        
        text = f"{DIR}/fold{fold}/train_after_wide_rectified.txt"
        with open(text) as f:
            lines = f.readlines()
        length = len(lines)
        
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
            
            if len(line):

                val = line.split(" ")
                img, msk, *frh = val
                msk_img = np.array(Image.open(msk), dtype=np.uint8)[:, :, 0]
                
                if mesh is None:
                    mesh = msk_img.shape[0] * msk_img.shape[1] // 24
                
                index = np.sum(msk_img == CLASSES[-1]) // mesh
                after[index] += 1
                
                cp.cprint(f"\033[1A{i+1} / {length}, [ {' '.join(map(str, after))} ]", "pink")
                
        with open(sarvey_text, mode="a") as f:
            print(f"{fold}, {' '.join(map(str, after))}", file=f)
        
        result += after
        
    with open(sarvey_text, mode="a") as f:
        print(f"sum, {' '.join(map(str, result))}", file=f)
    
    
def make_all_text():
    
    for buf in["", "_rectified"]:
        all_text = f"{DIR}/all_after_wide{buf}.txt"
        
        with open(all_text, mode="w") as al:
            for fold in range(1, 6):
                text = f"{DIR}/fold{fold}/test_after_wide{buf}.txt"
                with open(text) as te:
                    al.write(te.read())


def main():
    
    # cp.cprint("YOU DON'T RUN THIS !", "red")
    # exit()
    
    # make_2class()
    sarvey()
    # make_all_text()


main()
