import matplotlib.pyplot as plt
import numpy as np

# my module
from colorPrint import Cprint as cp

echo = lambda *string : cp.cprint(", ".join(map(str, string)), "cyan")



### データの読み取り


input_text = "/workspace/luminance/luminance.txt"
output_img = "/workspace/luminance/graph_1.png"

KEYS = "answer max min average red_max red_min red_average green_max green_min green_average blue_max blue_min blue_average".split(" ")


with open(input_text, mode="r", encoding="utf8") as f:
    input_data = list(map(lambda x : list(map(float, (x.split(" "))[1:])), f.read().split("\n")[1:]))

echo("- read answer data -")

data = [[0]*(len(input_data)-1) for _ in KEYS]

for i, d in enumerate(input_data):
    for j, val in enumerate(d):
        data[j][i] = val

data = dict(zip(KEYS, data))

echo("- analyzed answer data -")



### グラフの描画


yrange_avallable = True
xrange_avallable = False


if yrange_avallable:
    graph_yrange = [(0, 85000), (0, 85000),
                    (0, 17500), (0, 17500),
                    (0, 17500), (0, 17500)]
else:
    graph_yrange = [(0, 85000), (0, 85000),
                    (0, 85000), (0, 85000),
                    (0, 85000), (0, 85000)]

if xrange_avallable:
    graph_xrange = [(100, 140), (100, 140),
                    (50, 220), (50, 220),
                    (30, 200), (30, 200)]
else:
    graph_xrange = [(0, 255), (0, 255),
                    (0, 255), (0, 255),
                    (0, 255), (0, 255)]


quartile_color = ["#aaaaaa", "#aaaa22", "#ff3333"]

aggregation_positive = dict(zip("max min average".split(" "), [[0]*256 for _ in range(3)]))
aggregation_negative = dict(zip("max min average".split(" "), [[0]*256 for _ in range(3)]))

quartile_positive = dict(zip("max min average".split(" "), [0]*3))
quartile_negative = dict(zip("max min average".split(" "), [0]*3))


for target in "max min average".split(" "):
    for i, val in enumerate(data[target]):
        if (data["answer"][i]):
            aggregation_positive[target][int(val)] += 1
        else:
            aggregation_negative[target][int(val)] += 1
            
    accumlation_positive = list()
    accumlation_negative = list()
    
    for i, (ps, ng) in enumerate(zip(aggregation_positive[target], aggregation_negative[target])):
        if ps: accumlation_positive += [i]*ps
        if ng: accumlation_negative += [i]*ng
    
    quartile_positive[target] = list(np.percentile(accumlation_positive, [0, 25, 50, 75, 100]))
    quartile_negative[target] = list(np.percentile(accumlation_negative, [0, 25, 50, 75, 100]))


plt.title("aa")
fig = plt.figure(figsize=(40, 20))

# 完了、平均
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot([quartile_positive["average"][0], quartile_positive["average"][0]], graph_yrange[0], c=quartile_color[0], label="0%, 100%")
ax1.plot([quartile_positive["average"][1], quartile_positive["average"][1]], graph_yrange[0], c=quartile_color[1], label="25%, 75%")
ax1.plot([quartile_positive["average"][2], quartile_positive["average"][2]], graph_yrange[0], c=quartile_color[2], label="50%")
ax1.plot([quartile_positive["average"][3], quartile_positive["average"][3]], graph_yrange[0], c=quartile_color[1])
ax1.plot([quartile_positive["average"][4], quartile_positive["average"][4]], graph_yrange[0], c=quartile_color[0])
ax1.legend()
ax1.bar(range(256), aggregation_positive["average"])
ax1.set_title("positive - average")
ax1.set_ylabel("number of sheets")
ax1.set_xlabel("luminance")
ax1.set_ylim(graph_yrange[0])
ax1.set_xlim(graph_xrange[0])

# 完了、最大値
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot([quartile_positive["max"][0], quartile_positive["max"][0]], graph_yrange[2], c=quartile_color[0], label="0%, 100%")
ax2.plot([quartile_positive["max"][1], quartile_positive["max"][1]], graph_yrange[2], c=quartile_color[1], label="25%, 75%")
ax2.plot([quartile_positive["max"][2], quartile_positive["max"][2]], graph_yrange[2], c=quartile_color[2], label="50%")
ax2.plot([quartile_positive["max"][3], quartile_positive["max"][3]], graph_yrange[2], c=quartile_color[1])
ax2.plot([quartile_positive["max"][4], quartile_positive["max"][4]], graph_yrange[2], c=quartile_color[0])
ax2.legend()
ax2.bar(range(256), aggregation_positive["max"])
ax2.set_title("positive - max")
ax2.set_ylabel("number of sheets")
ax2.set_xlabel("luminance")
ax2.set_ylim(graph_yrange[2])
ax2.set_xlim(graph_xrange[2])

# 完了、最小値
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot([quartile_positive["min"][0], quartile_positive["min"][0]], graph_yrange[4], c=quartile_color[0], label="0%, 100%")
ax3.plot([quartile_positive["min"][1], quartile_positive["min"][1]], graph_yrange[4], c=quartile_color[1], label="25%, 75%")
ax3.plot([quartile_positive["min"][2], quartile_positive["min"][2]], graph_yrange[4], c=quartile_color[2], label="50%")
ax3.plot([quartile_positive["min"][3], quartile_positive["min"][3]], graph_yrange[4], c=quartile_color[1])
ax3.plot([quartile_positive["min"][4], quartile_positive["min"][4]], graph_yrange[4], c=quartile_color[0])
ax3.legend()
ax3.bar(range(256), aggregation_positive["min"])
ax3.set_title("positive - min")
ax3.set_ylabel("number of sheets")
ax3.set_xlabel("luminance")
ax3.set_ylim(graph_yrange[4])
ax3.set_xlim(graph_xrange[4])

# 未完了、平均
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot([quartile_negative["average"][0], quartile_negative["average"][0]], graph_yrange[1], c=quartile_color[0], label="0%, 100%")
ax4.plot([quartile_negative["average"][1], quartile_negative["average"][1]], graph_yrange[1], c=quartile_color[1], label="25%, 75%")
ax4.plot([quartile_negative["average"][2], quartile_negative["average"][2]], graph_yrange[1], c=quartile_color[2], label="50%")
ax4.plot([quartile_negative["average"][3], quartile_negative["average"][3]], graph_yrange[1], c=quartile_color[1])
ax4.plot([quartile_negative["average"][4], quartile_negative["average"][4]], graph_yrange[1], c=quartile_color[0])
ax4.legend()
ax4.bar(range(256), aggregation_negative["average"])
ax4.set_title("negative - average")
ax4.set_ylabel("number of sheets")
ax4.set_xlabel("luminance")
ax4.set_ylim(graph_yrange[1])
ax4.set_xlim(graph_xrange[1])

# 未完了、最大値
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot([quartile_negative["max"][0], quartile_negative["max"][0]], graph_yrange[3], c=quartile_color[0], label="0%, 100%")
ax5.plot([quartile_negative["max"][1], quartile_negative["max"][1]], graph_yrange[3], c=quartile_color[1], label="25%, 75%")
ax5.plot([quartile_negative["max"][2], quartile_negative["max"][2]], graph_yrange[3], c=quartile_color[2], label="50%")
ax5.plot([quartile_negative["max"][3], quartile_negative["max"][3]], graph_yrange[3], c=quartile_color[1])
ax5.plot([quartile_negative["max"][4], quartile_negative["max"][4]], graph_yrange[3], c=quartile_color[0])
ax5.legend()
ax5.bar(range(256), aggregation_negative["max"])
ax5.set_title("negative - max")
ax5.set_ylabel("number of sheets")
ax5.set_xlabel("luminance")
ax5.set_ylim(graph_yrange[3])
ax5.set_xlim(graph_xrange[3])

# 未完了、最小値
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot([quartile_negative["min"][0], quartile_negative["min"][0]], graph_yrange[5], c=quartile_color[0], label="0%, 100%")
ax6.plot([quartile_negative["min"][1], quartile_negative["min"][1]], graph_yrange[5], c=quartile_color[1], label="25%, 75%")
ax6.plot([quartile_negative["min"][2], quartile_negative["min"][2]], graph_yrange[5], c=quartile_color[2], label="50%")
ax6.plot([quartile_negative["min"][3], quartile_negative["min"][3]], graph_yrange[5], c=quartile_color[1])
ax6.plot([quartile_negative["min"][4], quartile_negative["min"][4]], graph_yrange[5], c=quartile_color[0])
ax6.legend()
ax6.bar(range(256), aggregation_negative["min"])
ax6.set_title("negative - min")
ax6.set_ylabel("number of sheets")
ax6.set_xlabel("luminance")
ax6.set_ylim(graph_yrange[5])
ax6.set_xlim(graph_xrange[5])


fig.tight_layout()
fig.savefig(output_img)

echo("- drew graph - ")
