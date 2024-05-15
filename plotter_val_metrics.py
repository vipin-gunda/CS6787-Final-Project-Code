import os
import json
import matplotlib.pyplot as plt

log = "work_dir/20240510_170139.log.json"
save_dir = "plots/Baseline_SegViT_19epochs/"

with open(log) as file:
    data = [json.loads(line) for line in file][1:]
acc_values = {}
iou_values = {}

iters = [i * 40000 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

for d in data:
    if d["mode"] == "val":
        for key, value in d.items():
            if key.startswith("Acc"): 
                if key not in acc_values:
                    acc_values[key] = []  
                acc_values[key].append(value)
            elif key.startswith("IoU"):
                if key not in iou_values:
                    iou_values[key] = [] 
                iou_values[key].append(value)

# Plotting and saving Accuracy plot
plt.figure(figsize=(12, 4))
for key in acc_values:
    plt.plot(iters, acc_values[key], label=key)
plt.xlabel("Iters")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # Two columns for legend
plt.tight_layout()
acc_plot_path = os.path.join(save_dir, "val_acc_perclass.png")
plt.savefig(acc_plot_path)
plt.close()

# Plotting and saving IoU plot
plt.figure(figsize=(12, 4))
for key in iou_values:
    plt.plot(iters, iou_values[key], label=key)
plt.xlabel("Iters")
plt.ylabel("IoU")
plt.title("Validation IoU")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # Two columns for legend
plt.tight_layout()
iou_plot_path = os.path.join(save_dir, "val_iou_perclass.png")
plt.savefig(iou_plot_path)
plt.close()
