# print = lambda *string : cp.cprint(", ".join(map(str, string)), "cyan")

class Cprint:
    
    colors = {
            "white"     : "ffffff",
            "lightgray" : "cccccc",
            "gray"      : "aaaaaa",
            "darkgray"  : "666666",
            "black"     : "000000",
            "red"       : "ff4444",
            "orange"    : "ff8844",
            "yellow"    : "ffff33",
            "green"     : "22ff22",
            "lightgreen": "bbffaa",
            "cyan"      : "33eeff",
            "blue"      : "3399ff",
            "purple"    : "ff55dd",
            "pink"      : "ffaaff",
            "brown"     : "bb7733",
        }


    def cprint(string:str="", color:str=None, background:str=None, end:str="\n", bloom:str=None, **kwargs):
        
        string = Cprint.colored(string, color, background, bloom)
        print(string, end=end, **kwargs)


    def colored(string:str="", color:str=None, background:str=None, bloom:str=None):
        
        try:
            before, after = "", ""
            
            if(color in Cprint.colors.keys()):
                color = Cprint.colors[color]
            if(background in Cprint.colors.keys()):
                background = Cprint.colors[background]
                
            if color and background:
                red, green, blue = Cprint.bloomTrans(list(map(lambda x: int(x, 16), [color[2*i:2*i+2] for i in range(3)])), bloom)
                red_back, green_back, blue_back = Cprint.bloomTrans(list(map(lambda x: int(x, 16), [background[2*i:2*i+2] for i in range(3)])), bloom)
                before, after = f"\033[48;2;{red_back};{green_back};{blue_back}m\033[38;2;{red};{green};{blue}m", f"\033[0m\033[0m"
            elif color:
                red, green, blue = Cprint.bloomTrans(list(map(lambda x: int(x, 16), [color[2*i:2*i+2] for i in range(3)])), bloom)
                before, after = f"\033[38;2;{red};{green};{blue}m", f"\033[0m"
            elif background:
                red_back, green_back, blue_back = Cprint.bloomTrans(list(map(lambda x: int(x, 16), [background[2*i:2*i+2] for i in range(3)])), bloom)
                before, after = f"\033[48;2;{red_back};{green_back};{blue_back}m", f"\033[0m"
            
            return f"{before}{string}{after}"
        
        except:
            Cprint.cprint(f"color input error ! ( color={color}, background={background} )", "ff4444")
            return string


    def colorMulti(color, coef, masterCoef:float=30):
        color = map(lambda c: int(coef*masterCoef) if ((coef>1) and (c<=0)) else c, color)
        color = map(lambda c: int(c*coef) if (int(c*coef) <= 255) else 255, color)
        return color


    def bloomTrans(color, bloom:str=None):
        if bloom is None:
            return color
        
        try:
            if(bloom in ["light", "l"]): coef = 1.2
            elif(bloom in ["dark", "d"]): coef = 0.5
            else: coef = float(bloom)
        except:
            coef = 1
        
        return Cprint.colorMulti(color, coef)
