import torch
import neck
import sys
neckruns = "/home/ubuntu/nfs/neckruns"
folder = sys.argv[1]
for t_type in neck.structs.TType:
    pt = torch.load(f"{neckruns}/{folder}/{t_type}.pt", weights_only=False)
    log_file = open(f"{neckruns}/{folder}/{t_type}.log", "w")
    print(f"previewing {t_type} tensors")
    for c_name, tensor_info in pt.items():
        print(tensor_info, end='', file=log_file)
