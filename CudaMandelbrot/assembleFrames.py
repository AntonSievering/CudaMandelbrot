from PIL import Image, warnings
import os
from readfile import load_values
from colorize import colorize
import time

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

folder_name = "frames/"

filenames = [file for file in os.listdir(os.path.join(folder_name)) if file.endswith('.frame')]
print("found {} frame files".format(len(filenames)))



def gen_string(n, len):
    strn = str(n)
    str0 = ""
    for _ in range(0, len - strn.__len__()):
        str0 += "0"
    return str0 + strn


tStart = time.time()

for file in filenames:
    frame = load_values(folder_name + file)
    i = int(file[5:10])
    
    colorize(frame, folder_name + "frame" + gen_string(i, 5) + ".png")
    os.system("del frames\\" + file)
    
    print("{} files done".format(i))

tEnd = time.time()
dt = tEnd - tStart

print("done. took {} seconds".format(dt))