from PIL import Image

from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/fullframe/text_dataset"
TEXT = ["all_after.txt"]


def check():
    
    for text in TEXT:
        
        print(f"\ntext : {text}\n", "pink")
        with open(f"{DIR}/{text}") as f:
            lines = f.readlines()
        length = len(lines)
        error = []
        
        for i, line in enumerate(map(lambda x: x.rstrip("\n"), lines)):
            val = line.split(" ")
            img, msk, *_ = val
            
            try:
                Image.open(msk)
            except:
                error.append(msk)
            
            cp.cprint(f"\033[1A{i+1} / {length}, {len(error)}", "orange")

        cp.cprint(f"Found {len(error)} errors.", "red")
        
        print(error)

def main():
    
    check()


main()
