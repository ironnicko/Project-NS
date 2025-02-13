from collections import deque
from bisect import bisect_right
import numpy as np
import torch
import cv2


def BFS(X, Y, img, visited):
    IMG_H, IMG_W = img.shape
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
    visited = [[0 for _ in range(IMG_W)] for __ in range(IMG_H)]
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

    out = pixels[low_idx: up_idx + 1]
    mean = sum(out) / max(1, len(out))

    adj_nuclei_count = up_idx + 1

    for i in range(up_idx + 1, len(pixels)):
        adj_nuclei_count += pixels[i] / mean
    adj_nuclei_count = int(adj_nuclei_count)
    return nuclei_count, adj_nuclei_count


def postProcess(frame, outputs):
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.1).astype(np.uint8)
    nuclei_count, adj_nuclei_count = connectedComponents(medsam_seg)
    medsam_seg = cv2.resize(
        medsam_seg, (frame.shape[1], frame.shape[0]))

    color = np.array([30/255, 144/255, 255/255])
    mask_image = (medsam_seg[:, :, None] *
                  color * 255).astype(np.uint8)
    frame_with_mask = cv2.addWeighted(frame, 0.5, mask_image, 0.5, 0)

    frame_with_mask_rgb = cv2.cvtColor(
        frame_with_mask, cv2.COLOR_BGR2RGB)
    return frame_with_mask_rgb, nuclei_count, adj_nuclei_count
