import cv2
import colour
import numpy as np

COLOR_DICT = {
    'black': [0, 128, 128], 
    'blue': [113, 159,  53], 
    'bronze': [217, 126, 141], 
    'brown': [147, 147, 159], 
    'green': [161,  66, 185], 
    'grey': [174, 128, 128], 
    'lightblue': [212, 112, 113], 
    'orange': [171, 172, 193], 
    'pink': [160, 201, 111], 
    'purple': [117, 178,  96], 
    'red': [129, 201, 180], 
    'white': [255, 128, 128], 
    'yellow': [239, 113, 220]
}

COLOR_RGB_DICT = {
    'black': [0, 0, 0], 
    'blue': [0, 0, 255], 
    'bronze': [219 ,212 ,188], 
    'brown': [185, 125, 86], 
    'green': [29, 177, 35], 
    'grey': [166, 166, 166], 
    'lightblue': [153, 217, 234], 
    'orange': [255, 127, 39], 
    'pink': [255, 84, 183], 
    'purple': [163, 73, 164], 
    'red': [255, 0, 0], 
    'white': [255, 255, 255], 
    'yellow': [255, 242, 0]
}

COLOR_PALETTE = ["black", "lightblue", "blue", "brown", "green", "grey", "orange", "red", "pink", "white", "yellow"] # We use these color to check distance per pixel
COLOR_LABEL = ["black", "blue", "blue", "brown", "green", "grey", "orange", "red", "red", "white", "yellow"] # We call them as these

def pixel_norm (pixel):
    colors_distance = [colour.delta_E(COLOR_DICT[c], pixel) for c in COLOR_PALETTE]
    nearest_index = colors_distance.index(min(colors_distance))
    return COLOR_RGB_DICT[COLOR_LABEL[nearest_index]]

def normalize (image: cv2.Mat):
    result = []
    resize = cv2.resize(image, (16, 16), interpolation = cv2.INTER_NEAREST)
    for row in range(len(resize)):
        for column in range(len(resize[row])):
            repl_pixel = pixel_norm(resize[row][column])
            result.extend(repl_pixel)
    return result

def raw_palette (image: cv2.Mat):
    result = []
    resize = cv2.resize(image, (16, 16), interpolation = cv2.INTER_NEAREST)
    for row in range(len(resize)):
        for column in range(len(resize[row])):
            result.extend(resize[row][column])
    return result
