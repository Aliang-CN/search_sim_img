# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : emb_engine.py
# Time       ：2022/11/16 下午4:27
# Author     ：Aliang
# Description：主要实现图片的向量生产和图片的相似度计算的代码
"""

import glob
import os
import numpy as np
from collections import defaultdict
from torchvision import models, transforms
from tqdm import tqdm
import lpips
import cv2
import torch
from urllib import request
import requests
import logging
import time

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class ImgEmbEngine(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.init_model()

    def read_img_path(self):
        img_paths = [i for i in glob.glob('../data/*')]
        return img_paths

    def download_img_from_url(self, url):
        """
        通过url下载图片
        :param url:
        :return:
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 ",
            'Connection': 'close'
        }
        r = requests.get(url, headers=headers)
        r.close()
        return r.content

    def deal_img(self, img_url):
        r = self.download_img_from_url(img_url)
        img_array = np.array(bytearray(r), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        if len(img.shape) == 2:
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.uint8)
        my_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        my_tensor = my_transforms(img)
        my_tensor = my_tensor.resize_(1, 3, 224, 224)
        my_tensor = my_tensor.to('cpu')
        return my_tensor

    def init_model(self):
        # 加载模型
        self.model = models.vgg16(pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
  
    def gen_emb(self, img_urls):
        img_emb_list = []
        for img_url in img_urls:
            img = self.deal_img(img_url)
            norm_feat = self.model(img)  # 得到特征，就可以改变网络
            norm_feat = norm_feat.view(norm_feat.size(1))
            emb = norm_feat.detach().numpy()
            img_emb_list.append(list(emb))
        return img_emb_list


class ImgSimEngine(object):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg', model_path=self.model_path)

    def deal_img(self, img_url):
        response = request.urlopen(img_url)
        img_array = np.array(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        img
        if len(img.shape) == 2:
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.uint8)
        my_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        my_tensor = my_transforms(img)
        my_tensor = my_tensor.resize_(1, 3, 224, 224)
        my_tensor = my_tensor.to('cpu')
        return my_tensor

    def cal_sim(self, img1, img2):
        s = self.loss_fn_alex(img1, img2)
        s = s.detach().numpy()
        s = np.squeeze(s)
        s = 1-s
        return s

    def batch_cal_sim(self, data, threshold=0.95):
        batch_img_id, batch_img_url, batch_sim_img_list, batch_sim_img_url_list = data[0], data[1], data[2], data[3]
        sim_img_dict_result = {k: [] for k in batch_img_id}
        for img_id, img_url, sim_img_list, sim_img_url_list in zip(batch_img_id, batch_img_url, batch_sim_img_list, batch_sim_img_url_list):
            for sim_img, sim_img_url in zip(sim_img_list, sim_img_url_list):
                s = self.cal_sim(self.deal_img(img_url), self.deal_img(sim_img_url))
                logging.info("img_id:{}, sim_img_id:{}, score:{}".format(img_id, sim_img, s))
                if s > threshold:
                    sim_img_dict_result[img_id].append(sim_img)
        return sim_img_dict_result

    def single_cal_sim(self, data, threshold=0.95):
        img_id, img_url, sim_img_list, sim_img_url_list = data[0], data[1], data[2], data[3]
        sim_img_dict_result = {img_id: []}
        for sim_img, sim_img_url in zip(sim_img_list, sim_img_url_list):
            s = self.cal_sim(self.deal_img(img_url), self.deal_img(sim_img_url))
            logging.info("img_id:{}, sim_img_id:{}, score:{}".format(img_id, sim_img, s))
            if s > threshold:
                sim_img_dict_result[img_id].append(sim_img)
        return sim_img_dict_result

