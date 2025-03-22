import torch
import neck
import sys
from pathlib import Path
from neck.structs import TType

neckruns = Path("/home/ubuntu/nfs/neckruns/")

run_name = sys.argv[1]
log_folder = neckruns / run_name
if not log_folder.exists():
    print(f"folder {run_name} does not exist")
    sys.exit(1)
for t_type in [TType.param] + list(TType):
    txt_file = log_folder / f"{t_type}.log"
    if txt_file.exists():
        continue
    pt_file = log_folder / f"{t_type}.pt"
    if not pt_file.exists():
        continue
    pt = torch.load(pt_file, weights_only=False)
    txt = open(txt_file, "w")
    print(f"previewing {t_type} tensors")
    for c_name, tensor_info in pt.items():
        print(tensor_info, end='', file=txt)
