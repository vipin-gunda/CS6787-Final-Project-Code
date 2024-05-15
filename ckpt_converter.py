import torch
import torch.nn as nn

model = torch.load("/home/vg245/SegVit/generic_130000.pth")

# model["meta"]["config"] = model["meta"]["config"].replace("num_classes=32","num_classes=34")

print(model["meta"]["config"])
model["meta"]["CLASSES"] = ('airport', 'amfootball', 'baseball', 'beach', 'bridge', 'cemetary', 'commercialarea', 'dam', 'farmland', 'forest', 'golf', 'highway', 'marina', 'garage', 'park', 'parkinglot', 'pond', 'railroad', 'residential', 'river', 'roundabout', 'sand', 'shootingrange', 'soccer', 'supermarket', 'swimming', 'tennis', 'university', 'warehouse', 'wetland', 'mansion', 'road')

model["meta"]["palette"] =[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7]][:32]

# model["state_dict"]["decode_head"]["class_embed"] = nn.Linear(512, 33)
loaded_weights = torch.load("/home/vg245/SegVit/google_earth_checkpoints/mansion-all.pth")

extended_weights = torch.cat([loaded_weights, torch.zeros(1, 512)], dim=0)
model["state_dict"]["decode_head.class_embed.weight"] = torch.nn.Parameter(extended_weights, requires_grad=False)
model["state_dict"]["decode_head.class_embed.bias"] = torch.zeros(33, 1)

torch.save(model, "/home/vg245/SegVit/generic_130000_eval.pth")
    
# # model meta classes
# # model meta palette
# # model state dict
# # model decodehead classembed
# # model decodehead.encoder?


