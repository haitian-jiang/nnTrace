import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
import invoke
import json

# configs = []
# for dp in (1, 2):
#   for pp, vpp in (
#       (1, 1),
#       (2, 1),
#       (2, 2),
#      ):
#     for tp, sp in (
#        (1, 0),
#        (2, 0),
#        (2, 1),
#        ):
#         for cp in (1, 2):
#            if dp == 1 and pp == 1 and tp == 1:
#               # Skip as tested
#               continue
#            if dp == 1 and pp == 1 and sp == False and cp == 1:
#               # Skip as tested
#               continue
#            if dp == 2 and pp == 2 and tp == 2 and cp == 2:
#               continue 
#            configs.append({
#               "dp": dp,
#               "pp": pp,
#               "vpp": vpp,
#               "tp": tp,
#               "sp": sp,
#               "cp": cp,
#            })
# 
# print(len(configs))
# for i in configs:
#   print(i)

configs = [
  {'dp': 1, 'pp': 1, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 2},
  {'dp': 1, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 1},
  # {'dp': 1, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 2}, # broken
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 2},
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 2},
  {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 1},
  # {'dp': 1, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 2}, # broken
  {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 1, 'sp': 0, 'cp': 2},
  {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 0, 'cp': 1},
  {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 0, 'cp': 2},
  {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 1, 'cp': 1},
  # {'dp': 1, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 1, 'cp': 2}, # broken
  {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 2},
  {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 1},
  # {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 2}, # broken
  {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 1},
  # {'dp': 2, 'pp': 1, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 2}, # broken
  {'dp': 2, 'pp': 2, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 2, 'pp': 2, 'vpp': 1, 'tp': 1, 'sp': 0, 'cp': 2},
  {'dp': 2, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 0, 'cp': 1},
  {'dp': 2, 'pp': 2, 'vpp': 1, 'tp': 2, 'sp': 1, 'cp': 1},
  {'dp': 2, 'pp': 2, 'vpp': 2, 'tp': 1, 'sp': 0, 'cp': 1},
  {'dp': 2, 'pp': 2, 'vpp': 2, 'tp': 1, 'sp': 0, 'cp': 2},
  {'dp': 2, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 0, 'cp': 1},
  {'dp': 2, 'pp': 2, 'vpp': 2, 'tp': 2, 'sp': 1, 'cp': 1},
]
for config in configs:
  dp = config["dp"]
  pp = config["pp"]
  vpp = config["vpp"]
  tp = config["tp"]
  sp = config["sp"]
  cp = config["cp"]

  neck_run_name=f"dp{dp}-pp{pp}-vpp{vpp}-tp{tp}-sp{sp}-cp{cp}"
  print(f"{neck_run_name=}")
  os.environ["NECK_RUN_NAME"] = neck_run_name
  try:
    invoke.run(f"rm -rf /home/ubuntu/efs/zhenyus/neckruns/{neck_run_name}")
    invoke.run(f"bash examples/llama.sh {dp} {pp} {vpp} {tp} {sp} {cp}")
    neck.merge_distr_logs(f"/home/ubuntu/efs/zhenyus/neckruns/{neck_run_name}")
    invoke.run(f"rm -rf diff_{neck_run_name}")
    neck.conduct_differential_test(f"/home/ubuntu/efs/zhenyus/neckruns/dp{1}-pp{1}-vpp{1}-tp{1}-sp{0}-cp{1}", f"/home/ubuntu/efs/zhenyus/neckruns/{neck_run_name}", f"diff_{neck_run_name}")
  except Exception as e:
    print(f"Error: {e}")
    continue
invoke.run("grep -v 'batch.0' diff*/*.log | awk '$NF > 20'")
invoke.run("grep -v 'batch.0' diff*/*.log | awk '$NF > 20'|awk '!seen[substr($0,1,29)]++ {print substr($0,1,29)}'")
print("Done")