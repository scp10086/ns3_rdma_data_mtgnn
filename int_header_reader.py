import re
from typing import List, Iterator
import pandas as pd
import math
import os
import json
line1_base = r"qp->hp.m_lastUpdateSeq (?P<notation>[!=]+) 0 (?P<firstflag>\w+\s+\w+(\s+\w+)?)"
line2_base = r"ih.nhop (?P<nhop>\d+)"
line3_base = r"(?P<time_step>\d+) (?P<update_type>\w+) (?P<sip>\w+) (?P<dip>\w+) (?P<sport>\d+) (?P<dport>\d+) \[(?P<m_lastUpdateSeq>\d+),(?P<ack_seq>\d+),(?P<next_seq>\d+)\]"
part4_first_base = r"(?P<qlen>\d+) (?P<byte>\d+) (?P<time>\d+)"
part4_not_first_base = r"(?P<now_ack_qlen>\d+)\((?P<last_ack_qlen>\d+)\) (?P<now_ack_byte>\d+)\((?P<last_ack_byte>\d+)\) (?P<now_ack_time>\d+)\((?P<last_ack_time>\d+)\)"
# Function to create an iterator over the trace file to read and parse lines in batches
def intheader_file_iterator(file_path: str, batch_size: int) -> Iterator[List[dict]]:
    line1_base_compile = re.compile(line1_base)
    line2_base_compile = re.compile(line2_base)
    line3_base_compile = re.compile(line3_base)
    part4_first_base_compile = re.compile(part4_first_base)
    part4_not_first_base_compile = re.compile(part4_not_first_base)    
    with open(file_path, 'r') as file:
        while True:
            batch_data = []
            for _ in range(batch_size):
                # read file and parse
                line = file.readline() 
                # Stop the iteration if we reached the end of the file                        
                if not line:
                    break         
                line1 = line.strip()
                #line1 first or not first
                                
                base1 = line1_base_compile.match(line1) 
                data1 = base1.groupdict()
                #print(data1)                
                if base1 == None:
                    break  
                #line 2 nhop

                line2 = file.readline() 
                line2 = line2.strip()
                
                base2 = line2_base_compile.match(line2)
                data2 = base2.groupdict()
                #print(data2)
                if base2 == None:
                    break                  
                #line 3 other data
                line3 = file.readline() 
                line3 = line3.strip()
                
                base3 = line3_base_compile.match(line3)
                data3 = base3.groupdict()
                #print(data3)
                if base3 == None:
                    break                
                        
                #part 4 int header

                Nhop = int(data2['nhop'])
                data4_nhop = []
                for i in range(Nhop):
                    line4 = file.readline()
                    line4 = line4.strip()
                    if data1['firstflag'] == "not first RTT":
                        base4 = part4_not_first_base_compile.match(line4)
                    if data1['firstflag'] == "first RTT":
                        base4 = part4_first_base_compile.match(line4)
                    data4 = base4.groupdict()
                    data4_nhop.append(data4)
                #print(data4_nhop)
                total_data = {"line1":data1,"line2":data2,"line3":data3,"part4":data4_nhop}
                batch_data.append(total_data)
                #print(total_data)
            # Fill the batch with None if it is smaller than batch_size
            while len(batch_data) < batch_size:
                batch_data.append(None)
            
            yield batch_data
            # Stop the iteration if we reached the end of the file
            if not line:
                break
# rearrange data according to the time slot and write to file
def slot_cluster(file_path: str,batch_size: int,total_time: float,start_time: float, time_slot: float):
    folder_name = "slot_cluster_totaltime_"+str(total_time)+"_time_slot_"+str(time_slot)
    batch_size = 200
    trace_iterator = intheader_file_iterator(file_path, batch_size)
    # make a folder
    total_slot_number = math.ceil(total_time/time_slot)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    #make  total_slot_number txt
    for i in range(total_slot_number):
        file_name = f"file_"+str(i)+".txt"
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "w") as f:
            pass
    while True:
        first_batch_data = next(trace_iterator)
        for i in range(batch_size):
            if first_batch_data[i] == None:
                return
            time_record = int(first_batch_data[i]["line3"]["time_step"])
            time_record = float(time_record/1e9)
            time_difference = time_record - start_time
            time_slot_order = math.floor( time_difference / time_slot)
            openfile_name =  f"file_"+str(time_slot_order)+".txt"
            openpath_name = os.path.join(folder_name, openfile_name)
            with open(openpath_name, "a+") as f:
                json.dump(first_batch_data[i], f)
                f.write('\n')

if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 输出当前工作目录
    print("当前工作目录：", current_dir)
    prefix_folder_path = "/originaldata/8node_4router_testtopo_link_250M/"
    base_path = "/home/zhanghua/qiaojing/data_analysis"
    os_path = base_path + prefix_folder_path
    # 更改当前工作目录
    os.chdir(os_path)

    # 输出更改后的工作目录
    print("更改后的工作目录：", os.getcwd())
    # Test the final implementation with the first batch of data
    #file_path = 'rdmahpccint.txt'
    file_path = 'rdmahpccint.txt'
    
    batch_size = 20
    total_time = 1
    start_time = 2
    time_slot = 0.0001
    # # Create the iterator
    # trace_iterator = intheader_file_iterator(file_path, batch_size)
    # # Get the first batch of data
    # while True:
    #     first_batch_data = next(trace_iterator)
    
    slot_cluster(file_path, batch_size,total_time,start_time,time_slot)

