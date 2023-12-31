import re
import pandas as pd
import numpy as np
import torch
import os
import json
import pickle
def read_routing_table(file_path):
    routing_table = []
    
    with open(file_path, 'r') as file:
        # Skip the first line as it contains the word "route"
        file.readline()
        
        for line in file:
            # Using regex to extract the necessary details from each line
            source = re.search(r'source: (\d+)', line).group(1)
            source_ip = re.search(r'source_ip: (\d+)', line).group(1)
            dst = re.search(r'dst: (\d+)', line).group(1)
            dst_ip = re.search(r'dst_ip: (\d+)', line).group(1)
            via = re.search(r'via: (\d+)', line).group(1)
            via_ip = re.search(r'via_ip: (.*?);', line).group(1)
            
            # Creating a dictionary with the extracted details
            route_details = {
                'source': int(source),
                'source_ip': int(source_ip),
                'dst': int(dst),
                'dst_ip': int(dst_ip),
                'via': int(via),
                'via_ip': via_ip
            }
            
            # Adding the dictionary to the list
            routing_table.append(route_details)
            
    return routing_table

def save_tensor(tensor, file_path):
    """Save the PyTorch tensor to a file."""
    torch.save(tensor, file_path)
    print(f"Tensor saved to {file_path}")

def load_tensor(file_path):
    """Load the PyTorch tensor from a file."""
    tensor = torch.load(file_path)
    print(f"Tensor loaded from {file_path}")
    return tensor

def get_full_path(routing_df):
    # this code is only suitable for hop no more than 2.
    # Create a dictionary to store the mappings from source to destination with the full path
    path_dict = {}

    # Initialize the dictionary with direct paths from the routing table
    for index, row in routing_df.iterrows():
        path_dict[(row['source'], row['dst'])] = [row['source'], row['via']]

    # Loop to construct the full path from source to destination
    # In each iteration, we extend the path with the next 'via' node in the route
    for _ in range(len(routing_df)):
        updated_path_dict = path_dict.copy()
        for (src, dst), path in path_dict.items():
            next_via = path[-1]  # Get the last node in the current path
            # Find the next 'via' node in the route and extend the path
            for index, row in routing_df[routing_df['source'] == next_via].iterrows():
                if row['dst'] == dst and row['via'] not in path:
                    updated_path_dict[(src, dst)].append(row['via'])
        
        # Update the path dictionary with the extended paths
        path_dict = updated_path_dict
    return path_dict    

def get_path_r(src, dst, routing_df, path_dict):
    """
    递归函数，用于计算从源节点到中间节点的路径
    """
    # 从路由表中查找下一个'via'节点，其'source'值等于当前节点，且'dst'值等于目标节点
    next_via = routing_df[(routing_df['source'] == src) & (routing_df['dst'] == dst)]['via'].iloc[0]

    # 如果中间节点就是目标节点，则返回路径中只包含源节点和目标节点
    if next_via == dst:
        return [src, dst]

    # 递归计算从中间节点到目标节点的路径
    path = get_path_r(next_via, dst, routing_df, path_dict)

    # 将从源节点到中间节点的路径和从中间节点到目标节点的路径组合起来
    full_path = [src] + path

    # 将路径添加到字典中
    path_dict[(src, dst)] = full_path

    return full_path

def get_full_path_r(routing_df):
    """
    递归函数，用于计算从源节点到目标节点的完整路径
    """
    path_dict = {}

    # 遍历每个源节点和目标节点的组合，计算从源节点到目标节点的完整路径
    for src in routing_df['source'].unique():
        for dst in routing_df['dst'].unique():
            # 如果源节点和目标节点相同，则路径只包含源节点和目标节点
            if src == dst:
                pass
                #path_dict[(src, dst)] = [src, dst]
            else:
                # 计算从源节点到中间节点的路径
                path = get_path_r(src, dst, routing_df, path_dict)
                # 将路径添加到字典中
                path_dict[(src, dst)] = path

    # 将字典转换为Pandas DataFrame，输出到控制台上
    paths_data = [{'source': src, 'destination': dst, 'path': path} for (src, dst), path in path_dict.items()]
    paths_df = pd.DataFrame(paths_data)
    
    return path_dict
def get_adj_matrix_from_route_table(paths_data):
    nodenumber = compute_number_node(paths_data)
    adj_matrix = np.zeros((nodenumber, nodenumber))
    for item in paths_data:
        path = item['path']
        for i in range(len(path)-1):
            adj_matrix[path[i]][path[i+1]] = 1
    #print(adj_matrix)
    return adj_matrix
def compute_number_node(paths_data):
    source_node_list = []
    dst_node_list = []
    for item in paths_data:
        source = item  ['source']
        dst = item['destination']
        path = item['path']
        # compute different number of node in source node and  dst node according to the path
        if source not in source_node_list:
            source_node_list.append(source)
        if dst not in dst_node_list:
            dst_node_list.append(dst)
    #print("source_node_list: ", source_node_list)
    #print("dst_node_list: ", dst_node_list)
    total_node_list = []
    # combine the source_node_list and dst_node_list without duplicate in both list 
    total_node_list = list(set(source_node_list).union(set(dst_node_list)))
    #print("total_node_list: ", total_node_list)
    return len(total_node_list)
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
    file_path = "hpccroute_8smaller.txt"
    # Read the routing table from the file
    routing_table = read_routing_table(file_path)
    # Convert the routing table list of dictionaries to a Pandas DataFrame
    routing_df = pd.DataFrame(routing_table)

    # Display the first few rows of the DataFrame
    routing_df.head()

    # Drop the 'via_ip' column as it is empty
    routing_df = routing_df.drop(columns=['via_ip'])

    # Convert the Pandas DataFrame to a NumPy array
    routing_np = routing_df.to_numpy()

    # Convert the NumPy array to a PyTorch tensor
    routing_tensor = torch.from_numpy(routing_np)

    # Display the first few rows of the tensor
    #print(routing_tensor[:5])
    
    # Define the paths to save and load the tensor
    save_path = 'routing_tensor.pt'

    # Save the tensor to a file
    save_tensor(routing_tensor, save_path)
    
    path_dict = get_full_path_r(routing_df)
    openpath_name = 'routing_path.pkl'

    # Create a DataFrame to store the full paths from source to destination
    paths_data = [{'source': src, 'destination': dst, 'path': path} for (src, dst), path in path_dict.items()]
    # write the paths_data to the  txt file
    pd.DataFrame(paths_data).to_csv('paths_data.csv', index=False)
    adj_matrix = get_adj_matrix_from_route_table(paths_data)
    # store adj_matrix as numpy array
    np.save('adj_matrix.npy', adj_matrix)
    path_dict_bytes = pickle.dumps(paths_data)
    with open(openpath_name, "wb") as f:
        f.write(path_dict_bytes)

    # Display the first few rows of the DataFrame
    #print(paths_df.head())
    #print(paths_df)