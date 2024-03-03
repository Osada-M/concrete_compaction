import random
import time

from colorPrint import Cprint as cp


eye = lambda color : cp.colored("0", color)

def choice():
    color = ""
    for i in range(3):
        c = hex(random.randint(0, 255))[2:]
        if (len(c) == 1):
            c = f"0{c}"
        color += c
    return color

color = choice()

print("\n")

while 1:
    try:
        if random.random() > 0.2:
            print(f"\033[1A( {eye(color)} _ {eye(color)} ) < {cp.colored(f'#{color}', color)}")
        else:
            print("\033[1A( - _ - )")
            color = choice()
        time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"\033[2A\n( {eye(color)} _ < ) < bye !          ")
        break
