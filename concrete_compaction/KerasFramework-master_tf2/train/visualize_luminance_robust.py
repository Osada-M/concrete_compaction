import os
from PIL import Image
import numpy as np
import colorsys

## my modules
from MyUtils import ImageManager as im
from MyUtils import Utils


ORIGINAL_IMG = "/workspace/Dataset/fullframe/image/190802/06_0150.png"
ORIGINAL_MSK = "/workspace/Dataset/fullframe/masked_4class/190802/masked_06_0150.png"
SAVE_DIR = "/workspace/visualization/luminance"

tests = [
    # ["rgb", "area_minmax", False, "linear"],
    # ["rgb", "default", True, "linear"],
    # ["rgb", "default", False, "linear"],
    # ["rgb", "default", False, "tanh"],
    # ["rgb", "default", 1, "linear", "20221004"],
    # ["rgb", "default", 1, "linear", "20221005"],
    # ["rgb", "default", 1, "linear", "20221025_skip-connection-0"],
    # ["rgb", "default", 1, "linear", "20221025_skip-connection-1"],
    # ["rgb", "default", 1, "linear", "20221025_skip-connection-2"],
    # ["rgb", "default", 1, "linear", "20221011"],
    # ["rgb", "default", 1, "linear", "20221014_ssim"],
    # ["rgb", "default", 1, "linear", "20221014_ssim_2x"],
    # ["rgb", "default", 1, "linear", "20221014_ssim_mse"],
    ["rgb", "default", 1, "linear", "20221014-1", [1]],
    ["rgb", "default", 1, "linear", "20221014-1", [2]],
    ["rgb", "default", 1, "linear", "20221014-1", [3]],
    ["rgb", "default", 1, "linear", "20221014_ssim_mse", [1]],
    ["rgb", "default", 1, "linear", "20221014_ssim_mse", [2]],
    ["rgb", "default", 1, "linear", "20221014_ssim_mse", [3]],
    ["rgb", "default", 1, "linear", "20221014_ssim_2x", [1]],
    ["rgb", "default", 1, "linear", "20221014_ssim_2x", [2]],
    ["rgb", "default", 1, "linear", "20221014_ssim_2x", [3]],
    
    # ["rgb", "default", 1, "tanh", "20221004"],
    # ["rgb", "default", 1, "tanh", "20221005"],
    # ["rgb", "default", 1, "tanh", "20221025_skip-connection-0"],
    # ["rgb", "default", 1, "tanh", "20221025_skip-connection-1"],
    # ["rgb", "default", 1, "tanh", "20221025_skip-connection-2"],
    # ["rgb", "default", 1, "tanh", "20221011"],
    # ["rgb", "default", 1, "tanh", "20221014_ssim"],
    # ["rgb", "default", 1, "tanh", "20221014_ssim_2x"],
    # ["rgb", "default", 1, "tanh", "20221014_ssim_mse"],
    ["rgb", "default", 1, "tanh", "20221014-1", [1]],
    ["rgb", "default", 1, "tanh", "20221014-1", [2]],
    ["rgb", "default", 1, "tanh", "20221014-1", [3]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_mse", [1]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_mse", [2]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_mse", [3]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_2x", [1]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_2x", [2]],
    ["rgb", "default", 1, "tanh", "20221014_ssim_2x", [3]],
    
    # ["hsv", "default", False, "linear"],
    # ["hsv", "area_minmax", False, "linear"],
    
    # ["lbp", "default", False, "linear"],
    # ["lbp-3c", "default", False, "linear"],
    
    # ["hog", "default", False, "linear"],
    # ["hog-3c", "default", False, "linear"],
    ]

def HSVColor(img):
    if isinstance(img, Image.Image):
        r, g, b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            Hdat.append(int(h * 255.))
            Sdat.append(int(s * 255.))
            Vdat.append(int(v * 255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB', (r, g, b))
    else :
        return None

def make_img_and_save():
    
    original_img = Image.open(ORIGINAL_IMG).resize((576, 576))
    original_msk = Image.open(ORIGINAL_MSK).resize((576, 576))
    
    original_img = np.array([np.array(original_img)])
    original_msk = np.array([np.array(original_msk)])
    
    for test in tests:
        color_type, normalization, use_AE_input, noise_type, AE_id, flip = test
        save_dir = f"{SAVE_DIR}/{'flip-'+''.join(map(str, flip))+'_' if len(flip) else ''}{color_type}_{normalization}{f'_use-AE-input_{AE_id}' if use_AE_input else ''}{f'-2' if use_AE_input==2 else ''}_{noise_type}"
        Utils.makedir(save_dir)
        
        im.set_AE_id(AE_id)
        
        for le_mode in ["all", "circle", "half", "slant"]:
            for le_const in [100 - (25*i) for i in range(9)]:# if (i != 4)]:
                
                img, *_ = im.adjust_data(
                    np.copy(original_img), np.copy(original_msk), is_fullframe=True, is_use_average_image=False, size=(576, 576), classification="fourclasses", num_classes=4,
                    is_use_LE=True, LE_mode=le_mode, LE_const=le_const, color_type=color_type, normalization=normalization,
                    # autoencoder=bool(len(noise_type)), use_AE_input=use_AE_input, noise=bool(len(noise_type)), noise_type=noise_type
                    use_AE_input=use_AE_input, noise_type=noise_type, is_flip=bool(len(flip)), flip_list=flip
                    )

                # pred_img = np.uint8(np.clip(img[0]*255, 0, 255))
                pred_img = np.uint8(img[0]*255.)
                del img
                img = Image.fromarray(pred_img)
                
                # img = np.copy(img[0])
                # if (np.min(img) < 0):
                #     img = (img+1.)/2.
                # img *= 255.
                # img = np.clip(img, 0, 255)
                # img = Image.fromarray(np.uint8(img))
                
                if (color_type == "hsv"):
                    # image = Image.new("HSV",(10, 10))
                    img = HSVColor(img)
                                
                img.save(f"{save_dir}/{le_mode}_{str(le_const).replace('-', 'in')}.png")
                
        print(test)


def main():
    make_img_and_save()


main()
