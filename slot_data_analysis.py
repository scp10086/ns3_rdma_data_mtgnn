import re
from typing import List, Iterator
import pandas as pd
import math
import os
import json

# read data according to the time slot
def slot_cluster_read(folder_path: str,batch_size: int, slot_order: int):
    
    openfile_name =  f"file_"+str(slot_order)+".txt"
    openpath_name = os.path.join(folder_path, openfile_name)
    folder_data = []
    with open(openpath_name, "r") as f:
        for line in f:
            item = json.loads(line)
            folder_data.append(item)
    return folder_data

def slot_to_one(src_path: str, out_path: str,node_number:int,feature_number:int,time_length:int,batchsize:int):
    # read file
    slot_cluster_read(src_path, batchsize,slot_order)
    # setup graph data
    
    # update graph data
    
    # save file
    return
if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 输出当前工作目录
    print("当前工作目录：", current_dir)

    # 更改当前工作目录
    os.chdir('/home/zhanghua/qiaojing')

    # 输出更改后的工作目录
    print("更改后的工作目录：", os.getcwd())

    read_batch_size = 20
    total_time = 0.1
    start_time = 2
    time_slot = 0.0001
    folder_path = "slot_cluster_totaltime_"+str(total_time)+"_time_slot_"+str(time_slot)
    slot_order = 0
    outfolder_path = "slot_out"+str(total_time)+"_time_slot_"+str(time_slot)
    node_number = 4
    feature_number = 1
    time_length = math.ceil(total_time/time_slot)
    slot_to_one(folder_path, outfolder_path,node_number,feature_number,time_length,read_batch_size)