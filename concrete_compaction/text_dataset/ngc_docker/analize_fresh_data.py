import pickle


DIR = "/media/nagalab/SSD1.7TB/nagalab/osada_ws/concrete_compaction/text_dataset/ngc_docker"


def extract_data():

    data = dict()
    with open(f"{DIR}/all.txt", mode="r") as f:
        readlines = f.readlines()
    for line in map(lambda x: x.rstrip("\n"), readlines):
        if line:
            path, ans, *fresh = line.split(" ")
            path_elements = path.split("/")
            _0, _1, _2, _3, _4, _5, ans, element, *_end = path_elements
            element = element.replace(".jpg", "").replace("ab", "").replace("CM", "")
            date, place, time_id, mesh_id = element.split("_")
            # key = int(date)
            key = f"{date}_{place}"
            
            if not key in data.keys():
                data[key] = list(map(float, fresh))
                # data[key] = set()
            # data[key].add(" ".join(fresh))
    
    return data


def write_data(data):
    
    with open(f"{DIR}/fresh.pickle", mode="wb") as f:
        pickle.dump(data, f)


def main():
    data = extract_data()
    print(data)
    print(len(data.keys()))
    write_data(data)
    

main()
