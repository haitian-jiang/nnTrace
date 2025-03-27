import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
 
for TP, CP in (
  (1, 2), (1, 4),
  (1, 8),
  (2, 1), (4, 1), (8, 1), 
  ):
  neck.conduct_differential_test("/home/ubuntu/efs/zhenyus/neckruns/oracle", f"/home/ubuntu/efs/zhenyus/neckruns/tp{TP}_cp{CP}", f"diff_tp{TP}_cp{CP}")