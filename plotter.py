import os
import json
import matplotlib.pyplot as plt

log = "work_dir/20240510_170139.log.json"
save_dir = "plots/Baseline_SegViT_19epochs/"

with open(log) as file:
    data = [json.loads(line) for line in file][1:]

def graph_metric(metric, train_val, data, file_name):
    x_axis = "iter" if train_val == "val" else "iter"
    if train_val == "train":
        iters = [d[x_axis] for d in data if d["mode"] == train_val and d[x_axis] % 1000 == 0]
        values = [d[metric] for d in data if d["mode"] == train_val and d[x_axis] % 1000 == 0]
    else: 
        iters = [i * 40000 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        values = [d[metric] for d in data if d["mode"] == train_val]
    
    plt.plot(iters, values)
    plt.xlabel(x_axis)
    plt.ylabel(metric)
    plt.title(train_val + " " + metric) 
    full_file_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(full_file_path)
    plt.close()
    
graph_metric("loss", "train", data, "train_loss")
graph_metric("mIoU", "val", data, "val_mIoU")
graph_metric("mAcc", "val", data, "val_mAcc")
graph_metric("aAcc", "val", data, "val_aAcc")
graph_metric("lr", "train", data, "lr")