import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
import invoke


configs = [
  {'dp': 1, 'pp': 1, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
]
for config in configs:
  dp = config["dp"]
  pp = config["pp"]
  vpp = config["vpp"]
  tp = config["tp"]
  sp = config["sp"]
  cp = config["cp"]

  neck_run_name=f"bug_fp8_pp_recompute_ga_pp{pp}"
  print(f"{neck_run_name=}")
  os.environ["NECK_RUN_NAME"] = neck_run_name
  invoke.run(f"rm -rf /home/ubuntu/efs/zhenyus/neckruns/{neck_run_name}")
  invoke.run(f"bash examples/llama.sh {dp} {pp} {vpp} {tp} {sp} {cp}")

neck.merge_distr_logs(f"/home/ubuntu/efs/zhenyus/neckruns/bug_fp8_pp_recompute_ga_pp{2}")
invoke.run(f"rm -rf diff_bug_fp8_pp_recompute_ga")
neck.conduct_differential_test(f"/home/ubuntu/efs/zhenyus/neckruns/bug_fp8_pp_recompute_ga_pp{1}", f"/home/ubuntu/efs/zhenyus/neckruns/bug_fp8_pp_recompute_ga_pp{1}", f"diff_bug_fp8_pp_recompute_ga")
invoke.run("grep -v 'batch.0' diff_bug_fp8_pp_recompute_ga/*.log | awk '$NF > 20'")