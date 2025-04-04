import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
import invoke
from neck.structs import TType

invoke.run(f"bash examples/reproduce-issues/vpp-zero-bug.sh 0")
invoke.run("rm -rf /home/ubuntu/efs/zhenyus/neckruns/vpp-zero-bug-1")
invoke.run(f"bash examples/reproduce-issues/vpp-zero-bug.sh 1")
neck.merge_distr_logs(f"/home/ubuntu/efs/zhenyus/neckruns/vpp-zero-bug-1", t_type=set(TType)-{TType.g_param_main})
neck.conduct_differential_test("/home/ubuntu/efs/zhenyus/neckruns/vpp-zero-bug-0", f"/home/ubuntu/efs/zhenyus/neckruns/vpp-zero-bug-1", f"diff_vpp-zero-bug")