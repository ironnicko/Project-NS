from sys import setrecursionlimit
from time import time
from base64 import b64encode
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
from bisect import bisect_right
from keras import backend as K
import numpy as np
from collections import deque

# CONSTANTS AND HELPER FUNCTIONS


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0)/(K.sum(y_true_f)+K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def input_image(img):
    img = resize(img[:, :, :IMG_C], (IMG_H, IMG_W),
                 mode='constant', preserve_range=True)
    resized_frame[0] = img

    return resized_frame


IMG_H = 128
IMG_W = 128
IMG_C = 3

SIZE_OF_GRID = 128

resized_frame = np.zeros((1, IMG_H, IMG_W, IMG_C), dtype=np.uint8)

model = load_model("./Attention_resunet.h5", custom_objects={
                   'jacard_coef': jacard_coef})


def BFS(X, Y, img, visited):
    IMG_H, IMG_W = img.shape
    """
        We will be calling a BFS call for each direction from our current postition thus we traverse:
                               ^  ^  ^
                                \ | /
                              < - o - >
                                / | \\
                               v  v  v

            UP, DOWN, LEFT, RIGHT, UP-RIGHT, UP-LEFT, DOWN-RIGHT, DOWN-LEFT
    """

    pixel = 0
    stack = deque([(X, Y)])

    while stack:
        x, y = stack.popleft()
        if x < 0 or y < 0 or x >= IMG_H or y >= IMG_W:
            continue
        if visited[x][y] or not img[x][y]:
            continue
        visited[x][y] = 1
        pixel += 1

        for dx in [1, 0, -1]:
            for dy in [1, 0, -1]:
                if dx == dy and dx == 0:
                    continue
                stack.append((x + dx, y + dy))
    return pixel


def connectedComponents(img):
    IMG_H, IMG_W = img.shape
    visited = [[0 for _ in range(IMG_H)]
               for __ in range(IMG_W)]
    nuclei_count = 0
    pixels = []
    for X in range(IMG_H):
        for Y in range(IMG_W):
            if img[X][Y] and not visited[X][Y]:
                nuclei_count += 1
                pixel = BFS(X, Y, img, visited)
                if pixel:
                    pixels.append(pixel)
    pixels.sort()

    Q1 = np.percentile(pixels, 25)
    Q3 = np.percentile(pixels, 75)

    IRQ = Q3 - Q1

    up_idx = bisect_right(pixels, Q3)
    low_idx = bisect_right(pixels, Q1)

    out = pixels[low_idx: up_idx+1]
    mean = sum(out) / max(1, len(out))

    adj_nuclei_count = up_idx + 1

    for i in range(up_idx+1, len(pixels)):
        adj_nuclei_count += pixels[i] / mean
    adj_nuclei_count = int(adj_nuclei_count)
    return nuclei_count, adj_nuclei_count


def main(img="input.png"):
    frame = cv2.imread(img)
    resized_frame = input_image(frame)
    resized_frame = resized_frame[int(resized_frame.shape[0]*0.9):]
    segmented = model.predict(
        resized_frame[int(resized_frame.shape[0]*0.9):])
    seg = np.squeeze((segmented > 0.5).astype(np.uint8))
    nuclei_count, adj_nuclei_count = connectedComponents(seg)
    l, r = min(nuclei_count, adj_nuclei_count), max(
        nuclei_count, adj_nuclei_count)
    cv2.putText(frame, f"Nuclei Count : {l} - {r}", (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)
    cv2.imwrite("123.png", frame)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
