import numpy as np
import os
import os
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

def find():
    unique_images = dict()
    for path, _, files in os.walk('.'):
        print(path)
        for file in files:
            pathToFile = path +"\\" + file
            if pathToFile.endswith(".jpg"): 
                img = Image.open(pathToFile)
                sum = np.sum(np.array(img))
                if sum in unique_images:
                    print("DUPLICITA")
                    print(pathToFile)
                    print(unique_images[sum])
                    return
                else:
                    unique_images[sum] = pathToFile
    print(unique_images)
find()
print("DONE")