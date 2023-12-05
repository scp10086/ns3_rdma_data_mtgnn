import route_reader
import int_header_reader
import slot_data_analysis
import os
import pandas as pd
import torch
import numpy as np
import pickle
if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 输出当前工作目录
    print("当前工作目录：", current_dir)
    if not os.path.exists('/home/zhanghua/qiaojing/data_analysis/originaldata/expriment48_after'):
        os.makedirs('/home/zhanghua/qiaojing/data_analysis/originaldata/expriment48_after')
    # 更改当前工作目录
    topo_list = ["fattree","random"]
    numberofhost_list = [8,16,24,32]
    numberofswitch_list = [10,20,30,40]
    numberoftotalnode_list = [18,36,54,72]
    cdf_list = ["WebSearch_distribution.txt","FbHdp_distribution.txt"]
    load_list = [0.3,0.5,0.7]
    bandwidth = ["25G"]
    time_list = [0.1]
    base_path = "/home/zhanghua/qiaojing/data_analysis/originaldata"
    save_route_file_base_path = "/home/zhanghua/qiaojing/data_analysis/originaldata/expriment48_after"
    for numberofswitch in numberofswitch_list:
            for cdf in cdf_list:
                for load in load_list:
                    for time in time_list:
                        for topo in topo_list:
                            cdf_split = cdf.split(".")
                            cdf = cdf_split[0]
                            numberofhost = int(numberofswitch*4/5)
                            total_node = numberofhost + numberofswitch
                            task_name_path = "expriment48/" + "node"+ '_'+str(total_node) +'_'+ "switch"+'_'+str(numberofswitch) + '_'+ "topo" + '_'+  str(topo) + '_' + "load"+'_' + str(load)+'_' +"cdf"+'_'+str(cdf)+'_'+"time"+'_' + str(time)
                            prefix_folder_path = task_name_path
                            task_path = base_path + '/' + prefix_folder_path
                            new_hpccroute_name = str(topo) + '_' + str(numberofswitch) + '_' + str(load) +'_'+str(cdf)+'_' + str(time)+ "_hpccroute.txt"
                            new_rdmahpccint_name = str(topo) + '_' + str(numberofswitch) + '_' + str(load) +'_'+str(cdf)+'_' + str(time)+ "_rdmahpccint.txt"
                            route_file_path = task_path+'/'+new_hpccroute_name
                            
                            # Define the paths to save and load the tensor
                            task_prefix = str(topo) + '_' + str(numberofswitch) + '_' + str(load) +'_'+str(cdf)+'_' + str(time)
                            save_route_path = task_path+'/'+task_prefix +  '_routing_tensor.pt'
                            # Save the tensor to a file
                            if not os.path.exists(save_route_path):
                                # Read the routing table from the file
                                routing_table = route_reader.read_routing_table(route_file_path)
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
                                route_reader.save_tensor(routing_tensor, save_route_path)

                            save_csv_route_path = task_path+'/'+task_prefix +  '_routing_path.csv'
                            # write the paths_data to the  txt file
                            if not os.path.exists(save_csv_route_path):
                                path_dict = route_reader.get_full_path_r(routing_df)
                                # Create a DataFrame to store the full paths from source to destination
                                paths_data = [{'source': src, 'destination': dst, 'path': path} for (src, dst), path in path_dict.items()]
                                pd.DataFrame(paths_data).to_csv(save_csv_route_path, index=False)
                            
                            # store adj_matrix as numpy array
                            save_adj_matrix_path = task_path+'/'+task_prefix +  '_adj_matrix.npy'
                            if not os.path.exists(save_adj_matrix_path):
                                adj_matrix = route_reader.get_adj_matrix_from_route_table(paths_data)
                                np.save(save_adj_matrix_path, adj_matrix)
                            
                            save_pickle_route_path = task_path+'/'+task_prefix +  '_routing_path.pkl'
                            if not os.path.exists(save_pickle_route_path):
                                path_dict_bytes = pickle.dumps(paths_data)
                                with open(save_pickle_route_path, "wb") as f:
                                    f.write(path_dict_bytes)
                            # read the int header    
                            batch_size = 5000
                            total_time = 0.1
                            start_time = 2
                            time_slot = 0.00001
                            int_header_file_path = task_path+'/'+new_rdmahpccint_name
                            prefix_path = task_path
                            enable_slot_cluster = 0
                            if enable_slot_cluster:
                                int_header_reader.slot_cluster(int_header_file_path, batch_size,total_time,start_time,time_slot,prefix_path)

                            # make time slot
                            
                            # normalize the data








# mean std normalization

# min max normalization

# write the slot data to the file
