import torch
 
 
def gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        info += f'CUDA:{id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
    return info[:-1]
 
 
if __name__ == '__main__':
    print(gpu_info())


    # 获取可用的 GPU 数量
    device_count = torch.cuda.device_count()
    print(f'Number of available GPUs: {device_count}')

    # 设置 PyTorch 的当前设备为第二张 GPU
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    # 查看当前 PyTorch 的设备
    print(f'Current device: {torch.cuda.current_device()}')
