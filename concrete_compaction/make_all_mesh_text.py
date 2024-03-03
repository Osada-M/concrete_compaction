
path = lambda target : f"/workspace/osada_ws/text_dataset/ngc_docker/{target}.txt"


def make_text():
    
    with open(path("all"), mode="w") as f: f.write(str())
    with open(path("train"), mode="w") as f: f.write(str())
    # with open(path("validation"), mode="w") as f: f.write(str())
    with open(path("test"), mode="w") as f: f.write(str())
    
    # for target in ["train", "validation", "test"]:
    for target in ["test"]:
        for fold in range(5):
            text = path(f"fold{fold+1}/{target}")
            with open(text, mode="r") as f:
                data = f.readlines()
            
            # with open(path(target), mode="a") as f:
                # print("".join(data).replace("//", "/"), file=f, end="\n")
                
            # if (target in ["validation", "test"]):
                # data = data[::5]
            with open(path("all"), mode="a") as f:
                print("".join(data).replace("//", "/"), file=f, end="\n")
            
            if (fold+1 == 4):
                with open(path("test"), mode="a") as f:
                    print("".join(data).replace("//", "/"), file=f, end="\n")
            else:
                with open(path("train"), mode="a") as f:
                    print("".join(data).replace("//", "/"), file=f, end="\n")
                
            
            print(f"{target}, {fold+1}")
    
    # for target in ["train", "validation", "test", "all"]:
    for target in ["train", "test", "all"]:
        print(target)
        
        with open(path(target), "r") as f:
            readlines = list(sorted(f.readlines()))
        
        print(readlines[:10])
        
        for i, val in enumerate(readlines):
            if (val != "\n"): break
        if i: del readlines[:i]
        
        print(readlines[:10])
        
        with open(path(target), "w") as f:
            f.write("".join(readlines))
            

def main():
    make_text()


main()
