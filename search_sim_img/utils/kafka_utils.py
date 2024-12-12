# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : kafka_utils.py
# Time       ：2022/11/15 下午5:30
# Author     ：Aliang
# Description：
"""

from confluent_kafka import Consumer as KFConsumer
from confluent_kafka import Producer as kafkaProducer
from confluent_kafka import TopicPartition, KafkaError

