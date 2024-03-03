from colorPrint import Cprint as cp


DIR = "/workspace/Dataset/fullframe/text_dataset"
ORIGIN = ["_4class_rectified", "masked_4class"]
REPLACE = ["_after_rectified", "masked_after"]


if (ORIGIN == REPLACE):
    print(f"Holy...  Must be ORIGIN != REPLACE.")
    exit()


def convert_label():
    
    try:
        with open(f"{DIR}/all{ORIGIN[0]}.txt", mode="r") as org:
            with open(f"{DIR}/all{REPLACE[0]}.txt", mode="w") as rep:
                rep.write((org.read()).replace(ORIGIN[1], REPLACE[1]))
        cp.cprint("Converted all.txt", "green")
    except:
        cp.cprint("Ignored all.txt", "red")
        
    
    for fold in range(1, 6):
        for target in ["train", "validation", "test"]:
            with open(f"{DIR}/fold{fold}/{target}{ORIGIN[0]}.txt", mode="r") as org:
                with open(f"{DIR}/fold{fold}/{target}{REPLACE[0]}.txt", mode="w") as rep:
                    rep.write((org.read()).replace(ORIGIN[1], REPLACE[1]))
            cp.cprint(f"Converted fold{fold}/{target}.txt", "green")


def main():
    
    print(f"{cp.colored(ORIGIN, 'pink')} => {cp.colored(REPLACE, 'cyan')}")
    convert_label()


main()
