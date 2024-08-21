#!/usr/bin/env python
import cv2
import math
import numpy as np
import random
import torch
import torch.nn.functional as F


def draw_image(image, corners, color=(4, 111, 241), thickness=2):
    face_idx = [[0,1,5,4],
                [1,2,6,5],
                [2,3,7,6],
                [3,0,4,7]]

    highest_sum = 0
    highest_point = (0,0)

    for ind_f in range(4):
        f = face_idx[ind_f]
        for j in range(4):
            pt1 = (int(corners[f[j]][0]), int(corners[f[j]][1]))
            pt2 = (int(corners[f[(j + 1) % 4]][0]), int(corners[f[(j + 1) % 4]][1]))
            cv2.line(image, pt1, pt2, color, thickness)

            if (pt1[0]) ** 2 + (pt1[1]) ** 2  > highest_sum:
                highest_sum = ((pt1[0]) ** 2) + ((pt1[1]) ** 2)

                highest_point = pt1

            if (pt2[0]) ** 2 + (pt2[1]) ** 2  > highest_sum:
                highest_sum = (pt2[0])** 2 + (pt2[1])**2
                highest_point = pt2

    return highest_point
def rotate_point(x, y, center_x, center_y, theta):
    x = x - center_x
    y = y - center_y
    nx = int(center_x + x * math.cos(theta) - y * math.sin(theta))
    ny = int(center_y + x * math.sin(theta) + y * math.cos(theta))
    return nx, ny    

def draw_bev_rect(image, rect, color = (4, 111, 241), thickness=2):  # default color = yellow
    center_x = rect[0]
    center_y = rect[1]
    w = rect[2]
    h = rect[3]
    theta = rect[4]
    x1 = center_x - 0.5 * w 
    x2 = center_x + 0.5 * w 
    y1 = center_y - 0.5 * h 
    y2 = center_y + 0.5 * h 

    point_list = []
    point_list.append(rotate_point(x1, y1, center_x, center_y, theta))
    point_list.append(rotate_point(x1, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y1, center_x, center_y, theta))

    red = (0, 0, 200)
    yellow = (4, 239, 242)
    orange = (4,111,241)

    cv2.line(image, point_list[0], point_list[1], color, thickness)
    cv2.line(image, point_list[1], point_list[2], color, thickness)
    cv2.line(image, point_list[2], point_list[3], red, thickness)
    cv2.line(image, point_list[3], point_list[0], color, thickness)

    highest_sum = 0
    highest_point = (0,0)
    for pt1 in point_list :

        if (pt1[0]) ** 2 + (pt1[1]) ** 2 > highest_sum:
            highest_sum = ((pt1[0]) ** 2) + ((pt1[1]) ** 2)

            highest_point = pt1


    return highest_point

def init_bev(x_offset=250, scale=10, bev_size=500, dis_interval=5, thickness=2):
    image_bev = np.zeros((bev_size, bev_size, 3), np.uint8)
    for i in range(1, 20):
        cv2.circle(image_bev, (x_offset, bev_size), scale * i * dis_interval, (255, 255, 255), thickness)
    return image_bev

def draw_bev(image_bev, x3d, z3d, l3d, w3d, ry3d, color=(4, 111, 241), x_offset=250, scale=10, bev_size=500):
    bev_rect = [0, 0, 0, 0, 0]
    bev_rect[0] = x3d * scale + x_offset
    bev_rect[1] = bev_size - z3d * scale
    bev_rect[2] = l3d * scale
    bev_rect[3] = w3d * scale
    bev_rect[4] = ry3d
    point_list = draw_bev_rect(image_bev, bev_rect, color)

    return point_list