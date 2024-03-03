## my module
from colorPrint import Cprint as cp
from MyUtils import Utils, Calc, TimeCounter
from MyUtils import ImageManager as im
from MyUtils import ModelManager as mm
from line_sender import send_master
from image_path_timer import ImagePathTime as IPT


TEXT = {"plain" : "",
        "resnet" : "_4class", 
        "myself" : "_4class_myself",
        "fresh" : "_include_fresh"}


ngc_docker = lambda target: f"/workspace/mesh_dataset/fold{target[0]}/test{target[1].replace(TEXT['fresh'], '')}.txt"
semseg = lambda target: f"/workspace/Dataset/semanticSegmentation/text_dataset/fold{target[0]}/test{target[1]}.txt"


# def make_fold_text()

def compare():
    
    for func in [ngc_docker, semseg]:
        print(f"\n{cp.colored('origin', 'cyan')}\t{cp.colored('chenged', 'orange')}")
        
        for fold in range(1, 6):
            
            print()
            
            with open(func([fold, TEXT["fresh"]]), mode="r") as org:
                org_length = len(org.readlines())
            print(f"\033[1A{cp.colored(f'{org_length}', 'cyan')}\t{cp.colored('-', 'orange')}")
            
            with open(func([fold, TEXT["myself"]]), mode="r") as chg:
                chg_length = len(chg.readlines())
            print(f"\033[1A{cp.colored(f'{org_length}', 'cyan')}\t{cp.colored(f'{chg_length}', 'orange')}")


def adjust(origin_key:str="fresh"):
    
    cp.cprint(f"origin_key : {origin_key}", "cyan")
    
    for func in [ngc_docker, semseg]:
        print()
        
        for fold in range(1, 6):
            
            with open(func([fold, TEXT[origin_key]]), mode="r") as org:
                origin_path = set()
                for line in map(lambda x: (x.rstrip("\n")).split(" "), org.readlines()):
                    if len(line):
                        origin_path.add(line[0])
            
            with open(func([fold, TEXT["myself"]]), mode="r") as chg:
                changed_line = []
                for line in map(lambda x: str(x).rstrip("\n"), chg.readlines()):
                    if len(line):
                        if (line.split(" "))[0] in origin_path:
                            changed_line.append(line)
            
            if not len(changed_line):
                cp.cprint("[!] \"changed_line\" has not any paths...\\(T~T)/\n", "red"); continue
            
            with open(func([fold, TEXT["myself"]]), mode="w") as chg: pass
            with open(func([fold, TEXT["myself"]]), mode="a") as chg:
                for line in changed_line:
                    print(line, file=chg)
            
            cp.cprint(f"\033[1Acompleted : {fold} / 5", "green")
            
            del origin_path, changed_line


def main():
    adjust("resnet")
    compare()


main()
