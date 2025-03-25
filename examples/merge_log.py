import os
os.environ["NECK_CONFIG_PATH"] = "/home/ubuntu/efs/zhenyus/NECK/config/gpt.yaml"
import neck
print(os.environ.get('NECK_CONFIG_PATH'))
neck.merge_distr_logs("/home/ubuntu/efs/zhenyus/neckruns/specimen")