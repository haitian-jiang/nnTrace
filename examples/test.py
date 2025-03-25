import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
neck.conduct_differential_test("/home/ubuntu/efs/zhenyus/neckruns/oracle", "/home/ubuntu/efs/zhenyus/neckruns/specimen", "diff_dir")