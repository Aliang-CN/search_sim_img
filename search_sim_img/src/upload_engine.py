# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : upload_engine.py
# Time       ：2022/11/16 下午3:36
# Author     ：Aliang
# Description：
"""

import requests
import json


class UploadEngine(object):
    def __init__(self, mode='test'):
        pass

    def write_sim_img(self, img_id: str, img_id_list: list):
        img_id_list = ','.join(img_id_list)
        params = {'appid': '',
                  'sign': '',
                  'img_id': img_id,
                  'img_id_list': img_id_list}
        r = requests.get(f'{self.api}/api/writeSimImg', params=params)
        r.close()
        errCode = json.loads(r.content)['errCode']
        return errCode

    def write_vec_img_info(self, data):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 ",
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        params = {'appid': '',
                  'sign': '',
                  'data': data}
        r = requests.post(f'{self.api}/api/batchWriteVecImg', headers=headers, params=params)
        r.close()
        errCode = json.loads(r.content)['errCode']
        return errCode
