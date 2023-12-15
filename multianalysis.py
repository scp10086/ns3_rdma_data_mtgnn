import route_reader
import int_header_reader
import slot_data_analysis
import os
import pandas as pd
import torch
import numpy as np
import pickle
import math
from sklearn.preprocessing import StandardScaler
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
                            print(task_name_path)
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
                            # read the int header and make time slot
                            batch_size = 5000
                            total_time = 0.1
                            start_time = 2
                            time_slot = 0.00001
                            int_header_file_path = task_path+'/'+new_rdmahpccint_name
                            prefix_path = task_path
                            enable_slot_cluster = 0
                            if enable_slot_cluster:
                                int_header_reader.slot_cluster(int_header_file_path, batch_size,total_time,start_time,time_slot,prefix_path)
                        
                            # read adj matrix
                            adj_matrix = np.load(save_adj_matrix_path)
                            #print(adj_matrix)
                            read_batch_size = 5000
                            total_time = 0.1
                            start_time = 2
                            time_slot = 0.00001
                            
                            route_file_path = save_pickle_route_path
                            outfolder_path = "slot_out"+str(total_time)+"_time_slot_"+str(time_slot)
                            host_number = numberofhost
                            switch_number = numberofswitch
                            # later optimize this parameter
                            feature_number = 1
                            folder_after_name = "slot_cluster_totaltime_"+str(total_time)+"_time_slot_"+str(time_slot)
                            folder_name = prefix_path + '/' + folder_after_name 
                            files = os.listdir(folder_name)
                            num_files = len(files)
                            time_length = num_files
                            # save matrix, arr, diff_matrix, diff_arr
                            enable_slot_analysis = 0
                            if enable_slot_analysis:
                                slot_data_analysis.slot_to_one(route_file_path,folder_name, outfolder_path,switch_number,host_number,feature_number,time_length,read_batch_size,adj_matrix)
                            # normalize the data
                            cluster_src_path = folder_name
                            raw_storedvalue_adjacency_matrix_path = cluster_src_path + '/'+'raw_storedvalue_adjacency_matrix_path.npy'
                            raw_diff_storedvalue_adjacency_matrix_path = cluster_src_path + '/'+'raw_diff_storedvalue_adjacency_matrix_path.npy'
                            raw_arr_path = cluster_src_path + '/'+'raw_arr_path.npy'
                            raw_diff_arr_path = cluster_src_path + '/'+'raw_diff_arr_path.npy'
                            
                            raw_storedvalue_adjacency_matrix = np.load(raw_storedvalue_adjacency_matrix_path)
                            raw_diff_storedvalue_adjacency_matrix = np.load(raw_diff_storedvalue_adjacency_matrix_path)
                            raw_arr = np.load(raw_arr_path)
                            raw_diff_arr = np.load(raw_diff_arr_path)
                            raw_storedvalue_adjacency_matrix = np.nan_to_num(raw_storedvalue_adjacency_matrix, nan=0)
                            raw_diff_storedvalue_adjacency_matrix = np.nan_to_num(raw_diff_storedvalue_adjacency_matrix, nan=0)
                            raw_arr = np.nan_to_num(raw_arr, nan=0)
                            raw_diff_arr = np.nan_to_num(raw_diff_arr, nan=0)
                            # mean std normalization
                            
                            # min max normalization
                            max_values_raw_storedvalue_adjacency_matrix = np.max(raw_storedvalue_adjacency_matrix)
                            min_values_raw_storedvalue_adjacency_matrix = np.min(raw_storedvalue_adjacency_matrix)
                            max_values_raw_diff_storedvalue_adjacency_matrix = np.max(raw_diff_storedvalue_adjacency_matrix)
                            min_values_raw_diff_storedvalue_adjacency_matrix = np.min(raw_diff_storedvalue_adjacency_matrix)
                            max_values_raw_arr = np.max(raw_arr)
                            min_values_raw_arr = np.min(raw_arr)
                            max_values_raw_diff_arr = np.max(raw_diff_arr)
                            min_values_raw_diff_arr = np.min(raw_diff_arr)
                            if max_values_raw_diff_storedvalue_adjacency_matrix == 0:
                                raw_diff_storedvalue_adjacency_matrix = slot_data_analysis.generate_diff_matrix(raw_storedvalue_adjacency_matrix)
                                # save the diff matrix
                                np.save(raw_diff_storedvalue_adjacency_matrix_path,raw_diff_storedvalue_adjacency_matrix)
                                # get the max value
                                max_values_raw_diff_storedvalue_adjacency_matrix = np.max(raw_diff_storedvalue_adjacency_matrix)
                                min_values_raw_diff_storedvalue_adjacency_matrix = np.min(raw_diff_storedvalue_adjacency_matrix)
                            if max_values_raw_diff_arr == 0:
                                raw_diff_arr = slot_data_analysis.generate_diff_arr(raw_arr)
                                np.save(raw_diff_arr_path,raw_diff_arr)
                                # get the max value
                                max_values_raw_diff_arr = np.max(raw_diff_arr)
                                min_values_raw_diff_arr = np.min(raw_diff_arr)
                            enable_min_max_normalize = 0
                            if enable_min_max_normalize:                                
                                # min_max_ normalize the data
                                min_max_normalize_raw_storedvalue_adjacency_matrix = (raw_storedvalue_adjacency_matrix - min_values_raw_storedvalue_adjacency_matrix)/(max_values_raw_storedvalue_adjacency_matrix - min_values_raw_storedvalue_adjacency_matrix)
                                min_max_normalize_raw_diff_storedvalue_adjacency_matrix = (raw_diff_storedvalue_adjacency_matrix - min_values_raw_diff_storedvalue_adjacency_matrix)/(max_values_raw_diff_storedvalue_adjacency_matrix - min_values_raw_diff_storedvalue_adjacency_matrix)
                                min_max_normalize_raw_arr = (raw_arr - min_values_raw_arr)/(max_values_raw_arr - min_values_raw_arr)
                                min_max_normalize_raw_diff_arr = (raw_diff_arr - min_values_raw_diff_arr)/(max_values_raw_diff_arr - min_values_raw_diff_arr)
                                # min_max_ save the normalized data
                                min_max_normalize_raw_storedvalue_adjacency_matrix_path = prefix_path + '/'+'min_max_normalize_raw_storedvalue_adjacency_matrix.npy'
                                min_max_normalize_raw_diff_storedvalue_adjacency_matrix_path = prefix_path + '/'+'min_max_normalize_raw_diff_storedvalue_adjacency_matrix.npy'
                                min_max_normalize_raw_arr_path = prefix_path + '/'+'min_max_normalize_raw_arr.npy'
                                min_max_normalize_raw_diff_arr_path = prefix_path + '/'+'min_max_normalize_raw_diff_arr.npy'
                                np.save(min_max_normalize_raw_storedvalue_adjacency_matrix_path, min_max_normalize_raw_storedvalue_adjacency_matrix)
                                np.save(min_max_normalize_raw_diff_storedvalue_adjacency_matrix_path, min_max_normalize_raw_diff_storedvalue_adjacency_matrix)
                                np.save(min_max_normalize_raw_arr_path, min_max_normalize_raw_arr)
                                np.save(min_max_normalize_raw_diff_arr_path, min_max_normalize_raw_diff_arr)
                                # save min value and max value
                                min_max_normalize_raw_storedvalue_adjacency_matrix_min_path = prefix_path + '/'+'min_max_normalize_raw_storedvalue_adjacency_matrix_min_max_value.txt'
                                min_max_normalize_raw_diff_storedvalue_adjacency_matrix_min_path = prefix_path + '/'+'min_max_normalize_raw_diff_storedvalue_adjacency_matrix_min_max_value.txt'
                                min_max_normalize_raw_arr_min_path = prefix_path + '/'+'min_max_normalize_raw_arr_min_max_value.txt'
                                min_max_normalize_raw_diff_arr_min_path = prefix_path + '/'+'min_max_normalize_raw_diff_arr_min_max_value.txt'
                                min_max_normalize_raw_storedvalue_adjacency_matrix_min_max_value = np.array([min_values_raw_storedvalue_adjacency_matrix,max_values_raw_storedvalue_adjacency_matrix])
                                min_max_normalize_raw_diff_storedvalue_adjacency_matrix_min_max_value = np.array([min_values_raw_diff_storedvalue_adjacency_matrix,max_values_raw_diff_storedvalue_adjacency_matrix])
                                min_max_normalize_raw_arr_min_max_value = np.array([min_values_raw_arr,max_values_raw_arr])
                                min_max_normalize_raw_diff_arr_min_max_value = np.array([min_values_raw_diff_arr,max_values_raw_diff_arr])
                                np.savetxt(min_max_normalize_raw_storedvalue_adjacency_matrix_min_path,min_max_normalize_raw_storedvalue_adjacency_matrix_min_max_value)
                                np.savetxt(min_max_normalize_raw_diff_storedvalue_adjacency_matrix_min_path,min_max_normalize_raw_diff_storedvalue_adjacency_matrix_min_max_value)
                                np.savetxt(min_max_normalize_raw_arr_min_path,min_max_normalize_raw_arr_min_max_value)
                                np.savetxt(min_max_normalize_raw_diff_arr_min_path,min_max_normalize_raw_diff_arr_min_max_value)
                            enable_StandardScaler_normalize = 1
                            if enable_StandardScaler_normalize:
                                # StandardScaler normalize the data
                                raw_storedvalue_adjacency_matrix_scaler = StandardScaler()
                                # reshape the data to 2D （-1,1）
                                raw_storedvalue_adjacency_matrix_StandardScaler = raw_storedvalue_adjacency_matrix.reshape(-1,1)
                                raw_storedvalue_adjacency_matrix_scaler.fit(raw_storedvalue_adjacency_matrix_StandardScaler)
                                # transform the data
                                raw_storedvalue_adjacency_matrix_normalized = raw_storedvalue_adjacency_matrix_scaler.transform(raw_storedvalue_adjacency_matrix_StandardScaler)
                                # reshape the data to 3D
                                raw_storedvalue_adjacency_matrix_normalized = raw_storedvalue_adjacency_matrix_normalized.reshape(raw_storedvalue_adjacency_matrix.shape)
                                # save the normalized data
                                raw_storedvalue_adjacency_matrix_normalized_path = prefix_path + '/'+'StandardScaler_raw_storedvalue_adjacency_matrix_normalized.npy'
                                np.save(raw_storedvalue_adjacency_matrix_normalized_path, raw_storedvalue_adjacency_matrix_normalized)
                                # save the scaler
                                raw_storedvalue_adjacency_matrix_scaler_path = prefix_path + '/'+'StandardScaler_raw_storedvalue_adjacency_matrix_scaler.pkl'
                                with open(raw_storedvalue_adjacency_matrix_scaler_path, 'wb') as f:
                                    pickle.dump(raw_storedvalue_adjacency_matrix_scaler, f)
                                    
                                
                                
                                # StandardScaler normalize the data
                                raw_diff_storedvalue_adjacency_matrix_scaler = StandardScaler()
                                # reshape the data to 2D （-1,1）
                                raw_diff_storedvalue_adjacency_matrix_StandardScaler = raw_diff_storedvalue_adjacency_matrix.reshape(-1,1)
                                raw_diff_storedvalue_adjacency_matrix_scaler.fit(raw_diff_storedvalue_adjacency_matrix_StandardScaler)
                                # transform the data
                                raw_diff_storedvalue_adjacency_matrix_normalized = raw_diff_storedvalue_adjacency_matrix_scaler.transform(raw_diff_storedvalue_adjacency_matrix_StandardScaler)
                                # reshape the data to 3D
                                raw_diff_storedvalue_adjacency_matrix_normalized = raw_diff_storedvalue_adjacency_matrix_normalized.reshape(raw_diff_storedvalue_adjacency_matrix.shape)
                                # save the normalized data
                                raw_diff_storedvalue_adjacency_matrix_normalized_path = prefix_path + '/'+'StandardScaler_raw_diff_storedvalue_adjacency_matrix_normalized.npy'
                                np.save(raw_diff_storedvalue_adjacency_matrix_normalized_path, raw_diff_storedvalue_adjacency_matrix_normalized)
                                # save the scaler
                                raw_diff_storedvalue_adjacency_matrix_scaler_path = prefix_path + '/'+'StandardScaler_raw_diff_storedvalue_adjacency_matrix_scaler.pkl'
                                with open(raw_diff_storedvalue_adjacency_matrix_scaler_path, 'wb') as f:
                                    pickle.dump(raw_diff_storedvalue_adjacency_matrix_scaler, f)
                                
                                # StandardScaler normalize the data
                                raw_arr_scaler = StandardScaler()
                                # reshape the data to 2D （-1,1）
                                raw_arr_StandardScaler = raw_arr.reshape(-1,1)
                                raw_arr_scaler.fit(raw_arr_StandardScaler)
                                # transform the data
                                raw_arr_normalized = raw_arr_scaler.transform(raw_arr_StandardScaler)
                                # reshape the data to 3D
                                raw_arr_normalized = raw_arr_normalized.reshape(raw_arr.shape)
                                # save the normalized data
                                raw_arr_normalized_path = prefix_path + '/'+'StandardScaler_raw_arr_normalized.npy'
                                np.save(raw_arr_normalized_path, raw_arr_normalized)
                                # save the scaler
                                raw_arr_scaler_path = prefix_path + '/'+'StandardScaler_raw_arr_scaler.pkl'
                                with open(raw_arr_scaler_path, 'wb') as f:
                                    pickle.dump(raw_arr_scaler, f)
                                    
                                
                                
                                # StandardScaler normalize the data
                                raw_diff_arr_scaler = StandardScaler()
                                # reshape the data to 2D （-1,1）
                                raw_diff_arr_StandardScaler = raw_diff_arr.reshape(-1,1)
                                raw_diff_arr_scaler.fit(raw_diff_arr_StandardScaler)
                                # transform the data
                                raw_diff_arr_normalized = raw_diff_arr_scaler.transform(raw_diff_arr_StandardScaler)
                                # reshape the data to 3D
                                raw_diff_arr_normalized = raw_diff_arr_normalized.reshape(raw_diff_arr.shape)
                                # save the normalized data
                                raw_diff_arr_normalized_path = prefix_path + '/'+'StandardScaler_raw_diff_arr_normalized.npy'
                                np.save(raw_diff_arr_normalized_path, raw_diff_arr_normalized)
                                # save the scaler
                                raw_diff_arr_scaler_path = prefix_path + '/'+'StandardScaler_raw_diff_arr_scaler.pkl'
                                with open(raw_diff_arr_scaler_path, 'wb') as f:
                                    pickle.dump(raw_diff_arr_scaler, f)
                                    
                                
                            
                                
                                

                                   
                                
                            








# write the slot data to the file
