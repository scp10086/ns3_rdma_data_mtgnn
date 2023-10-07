import re
from typing import List, Iterator
import pandas as pd
import math
import os
import json
import numpy as np
import torch
import route_reader
import pickle

# Function to convert IP from binary representation to dotted notation only the node number is blow 255
def convert_ip_to_id(bin_str):
    if isinstance(bin_str, str):
        ip =  f"{int(bin_str[0:2], 16)}.{int(bin_str[2:4], 16)}.{int(bin_str[4:6], 16)}.{int(bin_str[6:], 16)}"
        id = int(bin_str[4:6], 16)
    return id

# read data according to the time slot
def slot_cluster_read(folder_path: str,batch_size: int, slot_order: int):
    
    openfile_name =  f"file_"+str(slot_order)+".txt"
    openpath_name = os.path.join(folder_path, openfile_name)
    
    with open(openpath_name, "r") as f:
        while True:
            folder_data = []
            for _ in range(batch_size):
                line = f.readline() 
                if not line:
                    break
                item = json.loads(line)
                folder_data.append(item)
            while len(folder_data) < batch_size:
                folder_data.append(None)
            yield folder_data
def updata_arr_function(arr,switch_node_list_array,read_list_data):
    nhop = read_list_data['line2']['nhop']
    
    for hopid in range(int(nhop)):
        if read_list_data['line1']['firstflag'] == "not first RTT":
            temp = read_list_data['part4'][hopid]['now_ack_byte']
            temp = int(temp)
            switchid = switch_node_list_array[hopid]
            if temp >= arr[switchid][0]:
                arr[switchid][0]= temp   
        elif read_list_data['line1']['firstflag'] == "first RTT":
            temp = read_list_data['part4'][hopid]['byte']
            temp = int(temp)
            switchid = switch_node_list_array[hopid]
            if temp >= arr[switchid][0]:
                arr[switchid][0]= temp   

def slot_to_one(route_src_path: str,cluster_src_path: str, out_path: str,node_number:int,feature_number:int,time_length:int,batchsize:int):
    # read route file
    with open(route_src_path, 'rb') as f:
        data_loaded = pickle.load(f)
    #print(type(data_loaded[0]))
    paths_df = pd.DataFrame(data_loaded)
    #print(paths_df.head())

    # 创建一个文件夹
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # read cluster file    
    for t in range(time_length):
        cluster_read = slot_cluster_read(cluster_src_path, batchsize,t)
        # setup graph data
        arr = np.zeros((node_number, feature_number))
        end_flag = False
        while True:            
            read_list = next(cluster_read)
            for i in range(batchsize):
                if read_list[i] == None:
                    end_flag = True
                    break
            # get route path
            # use source and dst to find switch node / get coordinate
                src = read_list[i]['line3']['sip']
                src_id = convert_ip_to_id(src)
                dst = read_list[i]['line3']['dip']
                dst_id = convert_ip_to_id(dst)
                path = paths_df.loc[(paths_df['source'] == src_id) & (paths_df['destination'] == dst_id), 'path'].values[0]
                switch_node_list = path[1:-1]
                base_number = 8
                switch_node_list_array = [x - base_number for x in switch_node_list]
                print(switch_node_list_array)
                # update graph data by value feature upate function
                updata_arr_function(arr,switch_node_list_array,read_list[i])
            if end_flag == True:
                break
        # save array path
        nyp_path = out_path +'_'+str(t)+'.npy'
        save_array_path = os.path.join(out_path, nyp_path)
        np.save(save_array_path, arr)    
        print(t)
    
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
    route_file_path = "routing_path.pkl"
    slot_order = 0
    outfolder_path = "slot_out"+str(total_time)+"_time_slot_"+str(time_slot)
    node_number = 4
    feature_number = 1
    time_length = math.ceil(total_time/time_slot)
    slot_to_one(route_file_path,folder_path, outfolder_path,node_number,feature_number,time_length,read_batch_size)