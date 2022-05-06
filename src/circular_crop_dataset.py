#!python3

import os
import re
import glob
import shutil
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Pool
from timeit import default_timer as timer


def process_circular_crop(img_path, output_dir):
    # open input image and convert to array
    img = Image.open(img_path)   
    img_arr = np.array(img)
    
    # create a new RGB image that is a white circle inscribed in the input
    # image dimensions
    height, width = img.size
    if img.mode == "RGB":
        lum_img = Image.new(mode="RGB", size=(height, width), color=(0, 0, 0))
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice([(0, 0), (height, width)], 0, 360, fill=(255, 255, 255),
                      outline="white")
    elif img.mode == "RGBA":
        lum_img = Image.new(mode="RGBA", size=(height, width), color=(0, 0, 0, 1))
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice([(0, 0), (height, width)], 0, 360, fill=(255, 255, 255, 1),
                      outline="white")

    # where the new image is black, keep it black, and where the new image is
    # white use the input image
    lum_img_arr = np.array(lum_img)
    try:
        final_img_arr = np.where(lum_img_arr == 0, 0, img_arr)
    except ValueError as e:
        print(img_path)
        raise e
        
    dirname = img_path.split('/')[-2]
    output_path = os.path.join(output_dir, dirname)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    # print(output_filename)
    Image.fromarray(final_img_arr).save(output_filename)


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


if __name__ == "__main__":
    data_dir = "/home/sdan8/wake_data_augmentation/data/original_dataset/"
    output_dir = "/home/sdan8/wake_data_augmentation/data/circular_crop_dataset/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filenames = glob_re(r'.*(Cband|Sband|Xband|masks)',
                        glob.glob(data_dir + "*/*.png"))

    n = 4
    start = timer()
    with Pool(n) as p:
        p.starmap(process_circular_crop, [(filename, output_dir)
                                          for filename in filenames])
    print(f"Took {timer()-start:.4f} seconds with {n} processes")

    # copy the results csv files
    for band_dir in ["Cband", "Sband", "Xband", ""]:
        source = os.path.join(data_dir, band_dir, "results.csv")
        destination = os.path.join(output_dir, band_dir, "results.csv")
        shutil.copy(source, destination)
    
    print("Done.")

