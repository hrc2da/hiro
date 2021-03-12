import random

import cv2
import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)
from wordnet.net import WordDetectorNet
from wordnet.eval import infer_one
import torch

#probably shouldn't have this here...
net = WordDetectorNet()
net.load_state_dict(torch.load('SimpleHTR/wordnet/model/weights', map_location='cpu'))
net.eval()
net.to('cpu')


def cropRectangle(img):
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(img, 11, 17, 17)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    
    _,threshed = cv2.threshold(dilation, 170, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    filtered_rects = [rect for rect in rects if (rect[2] > 500 and rect[2] < 750 and rect[3] > 200 and rect[3] < 400 and rect[2]/rect[3] > 2)]
    
    if filtered_rects == False or len(filtered_rects) != 1:
        print("couldn't find the rectangle!")
    box = filtered_rects[0]
    print(box)
    img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    return img


def ceil32(val):
    if val % 32 == 0:
        return val
    val = (val // 32 + 1) * 32
    return val

def process_jpg_for_net(item, max_side_len=1024):
    orig = item

    f = min(max_side_len / orig.shape[0], max_side_len / orig.shape[0])
    if f < 1:
        orig = cv2.resize(orig, dsize=None, fx=f, fy=f)
    img = np.ones((ceil32(orig.shape[0]), ceil32(orig.shape[1])), np.uint8) * 255
    img[:orig.shape[0], :orig.shape[1]] = orig

    img = (img / 255 - 0.5).astype(np.float32)
    imgs = img[None, None, ...]
    imgs = torch.from_numpy(imgs).to('cpu')
    return imgs

def get_scale_factor(img, max_side_len=1024):
    f = min(max_side_len / img.shape[0], max_side_len / img.shape[0])
    return f if f < 1 else 1

def preprocess(img, imgSize, dataAugmentation=False):
    # "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros(imgSize[::-1])

    # data augmentation
    # img = img.astype(np.float)
    if dataAugmentation:
        # photometric data augmentation
        if random.random() < 0.25:
            rand_odd = lambda: random.randint(1, 3) * 2 + 1
            img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
        if random.random() < 0.25:
            img = cv2.dilate(img, np.ones((3, 3)))
        if random.random() < 0.25:
            img = cv2.erode(img, np.ones((3, 3)))
        if random.random() < 0.5:
            img = img * (0.25 + random.random() * 0.75)
        if random.random() < 0.25:
            img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 50), 0, 255)
        if random.random() < 0.1:
            img = 255 - img

        # geometric data augmentation
        wt, ht = imgSize
        h, w = img.shape
        f = min(wt / w, ht / h)
        fx = f * np.random.uniform(0.75, 1.25)
        fy = f * np.random.uniform(0.75, 1.25)

        # random position around center
        txc = (wt - w * fx) / 2
        tyc = (ht - h * fy) / 2
        freedom_x = max((wt - fx * w) / 2, 0) + wt / 10
        freedom_y = max((ht - fy * h) / 2, 0) + ht / 10
        tx = txc + np.random.uniform(-freedom_x, freedom_x)
        ty = tyc + np.random.uniform(-freedom_y, freedom_y)

        # map image into target image
        M = np.float32([[fx, 0, tx], [0, fy, ty]])
        target = np.ones(imgSize[::-1]) * 255 / 2
        img = cv2.warpAffine(img, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)


    # no data augmentation
    else:
        # center image
        # img = img[190:340,360:800]
        orig_img = img
        loaded_img_scale = 0.25
        
        padding = 40
        square_img = img[:,127:127+768]
        # net_img = cv2.resize(square_img, dsize=net.input_size, fx=loaded_img_scale, fy=loaded_img_scale)
        # net_img = (net_img/ 255 - 0.5)
        net_img = process_jpg_for_net(square_img)
        # net_img = cv2.resize(img, dsize=input_size)
        wordnetboxes = infer_one(net, net_img, max_aabbs=1000)
        f = get_scale_factor(net_img)
        wordnetboxes = [aabb.scale(1 / f, 1 / f) for aabb in wordnetboxes]
        wordboxes = [orig_img[int(aabb.ymin) : int(aabb.ymax), int(aabb.xmin) + 127: int(aabb.xmax) + 127] for aabb in wordnetboxes]

        img = cropRectangle(img)
        img = img[padding:-padding,padding:-padding]
        img = img.astype(np.float)
        

        wt, ht = imgSize
        h, w = img.shape
        f = min(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2
        # from matplotlib import pyplot as plt
        # plt.imshow(img,cmap='gray')
        # import pdb; pdb.set_trace()
        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones(imgSize[::-1]) * 255 / 2
        # import pdb; pdb.set_trace()
        # _,image = cv2.threshold(img,140, 255, cv2.THRESH_BINARY)                    
        # image = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,121,2)
        '''
        image = img
        # img = cv2.warpAffine(image, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        # increase contrast
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
        # import pdb; pdb.set_trace()
        # increase line width
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
        image = imgMorph
        '''
        img = cv2.warpAffine(img, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # transpose for TF
    img = cv2.transpose(img)
    # convert to range [-1, 1]
    
    # import pdb; pdb.set_trace()
    # img = img/127.5 - 1
    img = img / 255 - 0.5
    wordboxes.append(img)
    return wordboxes


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # img = cv2.imread('data/reading_imgs/0.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('SimpleHTR/data/5.jpg', cv2.IMREAD_GRAYSCALE)
    img_aug = preprocess(img, (128, 32), False)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(cv2.transpose(img_aug), cmap='gray')
    plt.show()
