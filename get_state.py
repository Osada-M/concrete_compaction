import time
import datetime

# my module
from concrete_compaction.colorPrint import Cprint as cp


OBSERVED_TEXT = "/media/nagalab/SSD1.7TB/nagalab/osada_ws/concrete_compaction/state.txt"
SLEEP = 5


def main():
    with open(OBSERVED_TEXT, "w") as f:
        f.write("\nobserve process start")
        
    while True:
        try:
            with open(OBSERVED_TEXT, "r") as f:
                string = f.read().rstrip("\n")
            with open(OBSERVED_TEXT, "w") as _: pass
            if string:
                now = datetime.datetime.now()           
                cp.cprint(f"> {now.strftime('%m/%d %H:%M:%S')}{' '*50}", "black", "white")
                buffer = string.split("\n")
                color, value = buffer[0], buffer[1:]
                if not color: color = "white"
                cp.cprint("\n".join(value), color, end="\n\n")
                
            time.sleep(SLEEP)
        except:
            break
    
    cp.cprint("\n- finished -", "cyan")


if(__name__ == "__main__"):
    main()
