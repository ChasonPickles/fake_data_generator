import sys
import sdv
from sdv.tabular import GaussianCopula, CTGAN
from ctgan import CTGANSynthesizer
from kafka import KafkaConsumer
from kafka import KafkaProducer
from json import loads
from json import dumps
import pandas as pd
import fcntl
from threading import Thread, Lock
import redis
import random
import datetime
from functools import reduce
import numpy as np
import json
import os

KAFKA_SEND_TOPIC = "ad-events-1"
#KAFKA_SEND_TOPIC = "stragglers-2"
KAFKA_RECEIVE_TOPIC = "stragglers"

GAUSSIAN_MODEL_FILE = 'gaussian_model.pkl'
GAN_MODEL_FILE = 'gan_model.pkl'
SAMPLE_AMOUNT = 800

kafka_port = 9092
kafka_broker = "localhost"

server = kafka_broker + ":" + str(kafka_port)
bootstrap_servers = [server]

consumer_regular = KafkaConsumer(
    KAFKA_RECEIVE_TOPIC,
    bootstrap_servers=server,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group-' + str(random.randint(0, 10)),
    value_deserializer=lambda x: loads(x.decode('utf-8')))

producer_regular = KafkaProducer(
    bootstrap_servers = server,
    key_serializer = lambda x: (x.encode('utf-8')),
    value_serializer = lambda x: (x.encode('utf-8')))

HEADERS = ["ad_id", "ad_type", "event_type"]
lock = Lock()


def create_fake_data_model(df):
    start_time = datetime.datetime.now()
    # create a new model here
    ctgan_model = GaussianCopula(field_transformers={
        'ad_id': 'label_encoding'
    })
    ctgan_model.fit(df)

    lock.acquire()
    ctgan_model.save(GAUSSIAN_MODEL_FILE)
    lock.release()

    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print("End Training Model")
    execution_time = round(time_diff.seconds / 60, 4)
    print("Training took " + str(execution_time) + "min")


if __name__ == "__main__":
    with open("exampleDataPoints.txt", "w+") as fh:
        tmp_data = []
        curr_data_points = 0
        for msg in consumer_regular:
            event = msg.value
            num_to_sample = event.get("EventsDiscardedSinceLastWatermark")
            if (num_to_sample is None):  # is straggler event
                curr_data_points = curr_data_points + 1
                tmp_dict = {}
                tmp_dict.update({"ad_id" : event["ad_id"]})
                tmp_dict.update({"event_type" : event["event_type"]})
                tmp_dict.update({"ad_type" : event["ad_type"]})
                tmp_data.append(tmp_dict)
            else:  # is watermark event
                sdv_instance = sdv.SDV()
                lock.acquire()
                model = sdv_instance.load(GAUSSIAN_MODEL_FILE)
                lock.release()
                sampled_data = model.sample(int(num_to_sample))
                for sampled_data_point in sampled_data.iterrows():
                    new_data_point = {}
                    new_data_point["user_id"] = "-"
                    new_data_point["page_id"] = "-"
                    new_data_point["ad_id"] = sampled_data_point[1]["ad_id"]
                    new_data_point["ad_type"] = sampled_data_point[1]["ad_type"]
                    new_data_point["event_type"] = sampled_data_point[1]["event_type"]
                    new_data_point["event_time"] = event["lastWatermark"]
                    new_data_point["ip_address"] = "-"
                    # send to Kafka
                    fh.write(str(new_data_point) + "\n")
                    producer_regular.send(KAFKA_SEND_TOPIC, dumps(new_data_point), dumps(new_data_point))

                if curr_data_points > 10000:
                    print("starting new thread creating new model\n")
                    df = pd.DataFrame(tmp_data, columns=["ad_id", "event_type", "ad_type"])
                    thread = Thread(target=create_fake_data_model, args=(df,))
                    thread.start()
                    curr_data_points = 0
                    tmp_data = []
