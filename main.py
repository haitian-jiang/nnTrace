import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
import invoke

invoke.run("rm -rf /home/ubuntu/efs/zhenyus/neckruns/wrong_fp8_comm_group_fast-0")
invoke.run(f"bash examples/reproduce-issues/wrong_fp8_comm_group_fast.sh 0")

invoke.run("rm -rf /home/ubuntu/efs/zhenyus/neckruns/wrong_fp8_comm_group_fast-1")
invoke.run(f"bash examples/reproduce-issues/wrong_fp8_comm_group_fast.sh 1")
neck.merge_distr_logs(f"/home/ubuntu/efs/zhenyus/neckruns/wrong_fp8_comm_group_fast-1")

invoke.run("rm -rf diff_wrong_fp8_comm_group_fast")
neck.conduct_differential_test("/home/ubuntu/efs/zhenyus/neckruns/wrong_fp8_comm_group_fast-0", f"/home/ubuntu/efs/zhenyus/neckruns/wrong_fp8_comm_group_fast-1", f"diff_wrong_fp8_comm_group_fast")

invoke.run("grep -v 'batch.0' diff_wrong_fp8_comm_group_fast/*.log | awk '$NF > 20'")
