from MyUtils import ImageManager as im


FULLFRAME = lambda fold: f"/workspace/Dataset/fullframe/text_dataset/fold{fold}/test_4class_rectified.txt"
# FULLFRAME = lambda fold: f"/workspace/Dataset/semanticSegmentation/text_dataset/fold{fold}/test_4class.txt"
MESH = lambda fold: f"/workspace/mesh_dataset/fold{fold}/test.txt"



def mesh_test():
    data = [None]*5
    length = 0
    
    for fold in range(1, 6):
        with open(MESH(fold), mode="r") as f:
            readlines = [y for y in map(lambda x: x.rstrip("\n"), f.readlines()) if len(y)]
            data[fold-1] = list(sorted(set(map(lambda x: im.get_image_key(x.split(" ")[0], without_mesh_id=True), readlines))))
            length += len(readlines)

    print(f"{'='*10} mesh {'='*10}")
    for fold, val in enumerate(data):
        print(fold+1, val)
    print(f"length : {length}")


def fullframe_test():
    data = [None]*5
    length = 0
    
    for fold in range(1, 6):
        with open(FULLFRAME(fold), mode="r") as f:
            readlines = [y for y in map(lambda x: x.rstrip("\n"), f.readlines()) if len(y)]
            data[fold-1] = list(sorted(set(map(lambda x: im.get_image_key(x.split(" ")[0], without_mesh_id=True, is_fullframe=True), readlines))))
            length += len(readlines)
    
    print(f"{'='*10} fullframe {'='*10}")
    for fold, val in enumerate(data):
        print(fold+1, val)
    print(f"length : {length}")


def main():
    
    mesh_test()
    print()
    fullframe_test()


main()
