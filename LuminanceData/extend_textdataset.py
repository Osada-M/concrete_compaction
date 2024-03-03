
# my module
from colorPrint import Cprint as cp


## ================ config ================


# DATASET_DIR = "/workspace/Dataset/semanticSegmentation"
DATASET_DIR = "/workspace/Dataset/fullframe"
LIMIT = None

LUMINANCE_COEF = [0.5, 0.6, 0.7, 0.8, 0.9,
                  1.0, 1.1, 1.2, 1.3, 1.4,
                  1.5]


## ========================================


text = lambda target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}.txt"

all_text = lambda buf : f"{DATASET_DIR}/text_dataset/all{buf}.txt"

save_path = lambda target : f"{DATASET_DIR}/text_dataset/fold{target[0]}/{target[1]}_luminance.txt"


def datacounter(datapath):
    """
    @機能：テキストファイルの行数を数えるだけ
    @引数：数えたいテキストファイルのパス
    @戻値：行数
    """
    with open(datapath) as f:
        readlines = f.readlines()
        if LIMIT : readlines = readlines[:LIMIT]
        
    return len(readlines)


def extend():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    ## 
    for fold in range(6):
        
        ## 
        for mode in ["", "_include_fresh"]:
            cp.cprint(f"\n{f'fold{fold}' if fold else 'all'} {mode.replace('_', '')}", "green")
            
            ## 
            for target in ["train", "test", "validation"]:
                target += mode
                ## 
                if not fold:
                    # break
                    text_data = all_text(mode)
                    save_data = all_text(mode + "_luminance")
                    target = "all" + mode
                else:
                    text_data = text([fold, target])
                    save_data = save_path([fold, target])
                
                ## 
                with open(text_data, "r") as f:
                    lines = f.readlines()
                
                ## 
                datacount = datacounter(text_data)
                cp.cprint(f"{target} | completed : 0 / {datacount}", "cyan")
                
                ## 
                with open(save_data, "w") as f: pass
                with open(save_data, "a") as f:
                    ## 
                    for i, l in enumerate(map(lambda x: x.rstrip("\n"), lines)):
                        ## 
                        for luminance in LUMINANCE_COEF:
                            print(f"{l} {luminance}", file=f)
                            
                        cp.cprint(f"\033[1A{target} | completed : {i+1} / {datacount}{' '*50}", "cyan")
                
                if not fold: break


def main():
    extend()
    cp.cprint(f"\n- finished -", "cyan")


if(__name__ == "__main__"):
    main()
                            
                