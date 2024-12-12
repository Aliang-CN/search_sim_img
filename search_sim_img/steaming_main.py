# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : steaming_main.py
# Time       ：2022/11/16 上午10:00
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
from data_sources.consumer.kafka_consumer import KafkaConsumer
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class SteamingPipeline(object):
    """
    实现获取批图片并且写入milvus引擎的功能
    """
    def __init__(self, topk=100, threshold=0.95, mode='test'):
        self.topk = topk
        self.threshold = threshold
        self.download_engine = DowndloadEngine(mode)
        self.upload_engine = UploadEngine(mode)
        self.milvus = MilvusHandle(host='172.31.160.150', port='19530', collection_name='sim_img')
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.img_emb_engine = ImgEmbEngine(os.path.join(self.abs_path, 'model/vgg16-397923af.pth'))
        self.img_sim_engine = ImgSimEngine(os.path.join(self.abs_path, 'model/vgg16-397923af.pth'))
        if mode == "test":
            self.k_client = KafkaConsumer(topics=['adc_creative_data'], group_id='sim_img', data_source='default')
        elif mode == "pre-online":
            self.k_client = KafkaConsumer(topics=['adc_creative_data'], group_id='sim_img', data_source='online')
        elif mode == "online":
            self.k_client = KafkaConsumer(topics=['adc_creative_data'], group_id='sim_img', data_source='online')
        else:
            raise Exception('err mode, please use test or online')

    def merge_result(self, batch_img_id, alike_result, sim_result):
        result = {}
        for img_id in batch_img_id:
            alike_img_list = alike_result[img_id]
            sim_img_list = sim_result[img_id]
            result[img_id] = alike_img_list + sim_img_list
        return result

    def pipeline(self):
        """
        1、从kafka中获取信息
        2、下载图片，并且生成emb
        3、插入emb到milvus
        4、上传emb到接口
        5、获取vec索引
        6、图片召回
        7、图片打分排序
        8、图片相似结果写入
        :return:
        """
        while True:
            img_id, flag = self.k_client.pull()
            if flag == 0:
                time.sleep(5)
                continue

            img_id = img_id[0]
            img_url = self.download_engine.get_url_from_img_id(img_id=img_id)
            batch_img_id = [img_id]
            batch_img_url = [img_url]
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
                    logging.info("last_id: {}".format(img_id))
                    break
                except Exception as e:
                    if i >= 5:
                        break
                    else:
                        logging.info(f"err index {img_id}")
                        logging.info(e)
                        time.sleep(3)

            for i in range(5):
                try:
                    img_emb = self.img_emb_engine.gen_emb([img_url])
                    batch_img_emb = img_emb
                    search_result = self.milvus.search(batch_img_emb, top_k=self.topk)

                    batch_alike_vec_list = [[id for id, dis in zip(id_list, distance_list) if dis <= 20.0]
                                            for id_list, distance_list in zip(search_result.id_array,
                                                                              search_result.distance_array)]

                    batch_sim_vec_list = [
                        [id for id, dis in zip(id_list, distance_list) if dis < 250.0 and dis > 20.0]
                        for id_list, distance_list in zip(search_result.id_array,
                                                          search_result.distance_array)]

                    batch_alike_img_list = [self.download_engine.get_img_id_by_vec_id(l) for l in
                                            batch_alike_vec_list]

                    alike_result = {img_id: alike_img_list
                                    for img_id, alike_img_list in zip(batch_img_id, batch_alike_img_list)}

                    batch_sim_img_list = [self.download_engine.get_img_id_by_vec_id(l) for l in
                                          batch_sim_vec_list]
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
                        logging.info(f"err index {img_id}")
                        logging.info(e)
                        time.sleep(3)


def main():
    # 生成embedding
    beg = SteamingPipeline(mode='pre-online')
    beg.pipeline()


if __name__ == '__main__':
    main()