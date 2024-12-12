# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : sing_main.py.py
# Time       ：2022/11/21 下午7:40
# Author     ：Aliang
# Description：
"""

import json
import time
import os
import sys
sys.path.append("../")
from search_sim_img.src.download_engine import DowndloadEngine
from search_sim_img.src.upload_engine import UploadEngine
from vec_engine.handle.milvus_handle import MilvusHandle
from search_sim_img.src.img_engine import ImgEmbEngine, ImgSimEngine

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class BatchEmbGen(object):
    """
    实现获取批图片并且写入milvus引擎的功能
    """
    def __init__(self, mode):
        self.download_engine = DowndloadEngine(mode)
        self.upload_engine = UploadEngine(mode)
        self.milvus = MilvusHandle(host='127.0.0.1', port='19530', collection_name='sim_img')
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.img_emb_engine = ImgEmbEngine(os.path.join(self.abs_path, 'model/vgg16-397923af.pth'))

    def pipeline(self):
        """
        1、获取批id和url
        2、下载图片，并且生成emb
        3、插入emb到milvus
        4、上传emb到接口
        :return:
        """
        indexs = [10432]
        for index in indexs:
            batch_img_id, batch_img_url,  last_index, flag = self.download_engine.get_batch_img_id(index=index, limit=1)
            if flag == 1:
                break
            for i in range(5):
                try:
                    batch_img_emb = self.img_emb_engine.gen_emb(batch_img_url)
                    batch_vec_id = self.milvus.insert(batch_img_emb)
                    data = [{"img_id": img_id, "vec_id": vec_id, "embedding": "0"} for img_id, vec_id in
                            zip(batch_img_id, batch_vec_id)]
                    data = json.dumps(data)
                    logging.info("write vec info: {}".format(data))
                    status = self.upload_engine.write_vec_img_info(data)
                    logging.info("write vec result: {}".format(status))
                    logging.info("last_id: {}".format(index))
                    break
                except Exception as e:
                    if i >=5 :
                        break
                    else:
                        logging.info(f"err index {index}")
                        logging.info(e)
                        time.sleep(3)

        logging.info("batch embe gen done !!!")


class BatchSimCal(object):
    def __init__(self, topk=100, threshold=0.95, mode='test'):
        self.topk = topk
        self.threshold = threshold
        self.download_engine = DowndloadEngine(mode)
        self.upload_engine = UploadEngine(mode)
        self.milvus = MilvusHandle(host='127.0.0.1', port='19530', collection_name='sim_img')
        self.img_emb_engine = ImgEmbEngine()
        self.img_sim_engine = ImgSimEngine()

    def merge_result(self, batch_img_id, alike_result, sim_result):
        result = {}
        for img_id in batch_img_id:
            alike_img_list = alike_result[img_id]
            sim_img_list = sim_result[img_id]
            result[img_id] = alike_img_list + sim_img_list
        return result

    def pipeline(self):
        """
        1、获取vec索引
        2、图片召回
        3、下载召回图片
        4、图片打分排序
        5、图片相似结果写入
        """
        index = 10431
        for i in range(5):
            try:
                batch_img_id, batch_img_url, last_index, flag = self.download_engine.get_batch_img_id(index=index, limit=1)
                logging.info(batch_img_id)
                index = last_index
                batch_img_emb = self.img_emb_engine.gen_emb(batch_img_url)
                search_result = self.milvus.search(batch_img_emb, top_k=self.topk)

                batch_alike_vec_list = [[id for id, dis in zip(id_list, distance_list) if dis <= 20.0]
                                      for id_list, distance_list in zip(search_result.id_array,
                                                                        search_result.distance_array)]

                batch_sim_vec_list = [[id for id, dis in zip(id_list, distance_list) if dis < 500.0 and dis > 20.0]
                                      for id_list, distance_list in zip(search_result.id_array,
                                                                        search_result.distance_array)]

                batch_alike_img_list = [self.download_engine.get_img_id_by_vec_id(l) for l in batch_alike_vec_list]

                alike_result = {img_id: alike_img_list
                                for img_id, alike_img_list in zip(batch_img_id, batch_alike_img_list)}

                batch_sim_img_list = [self.download_engine.get_img_id_by_vec_id(l) for l in batch_sim_vec_list]
                batch_sim_img_url_list = [[self.download_engine.get_url_from_img_id(i) for i in sim_img]
                                          for sim_img in batch_sim_img_list]
                data = (batch_img_id, batch_img_url, batch_sim_img_list, batch_sim_img_url_list)
                sim_result = self.img_sim_engine.batch_cal_sim(data, self.threshold)

                result = self.merge_result(batch_img_id, alike_result, sim_result)
                if len(result) != 0:
                    for img_id, sim_id_list in result.items():
                        status = self.upload_engine.write_sim_img(img_id, sim_id_list)
                        logging.info("write sim img result: {}".format(status))
                break
            except Exception as e:
                if i >= 5:
                    break
                else:
                    logging.info(f"err index {index}")
                    logging.info(e)
                    time.sleep(3)

            logging.info("batch cal img done !!!")


def main():
    # 生成embedding
    beg = BatchEmbGen(mode='pre-online')
    beg.pipeline()

    # 计算相似度
    bsc = BatchSimCal(mode='pre-online')
    bsc.pipeline()


if __name__ == '__main__':
    main()