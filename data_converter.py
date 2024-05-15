import os
import numpy as np
from PIL import Image
mask_dir = "segdataset/masks/"
img_dir = "segdataset/images/"
save_img_dir = "segdataset_converted/images/"
save_mask_dir = "segdataset_converted/masks/"
num_classes = 32
test_classes = [7, 8, 9, 12, 14, 20, 22, 25, 32]
train_index = [i for i in range(1, num_classes + 1) if i not in test_classes]
num_train, num_val, num_test = 0, 0, 0


for root, _, files in os.walk(img_dir):
    for file in files:
        if file == ".DS_Store":
            continue
        training = np.random.choice(
            ["training", "validation", "test"], p=[0.745, 0.115, 0.14]
        )
        file_name = os.path.splitext(file)[0]
        img_mask_dir = os.path.join(mask_dir, os.path.basename(root), file_name)
        annotation = np.zeros((224, 224))
        if len(
            [
                mask
                for mask in os.listdir(img_mask_dir)
                if int(os.path.splitext(mask)[0]) + 1 in test_classes
            ]
        ) == len(os.listdir(img_mask_dir)):
            training = "test"
        for mask in os.listdir(img_mask_dir):
            class_idx = int(os.path.splitext(mask)[0]) + 1
            if training != "test" and class_idx in test_classes:
                continue
            im = np.array(Image.open(os.path.join(img_mask_dir, mask)).convert("L"))
            if training == "test":
                annotation[im == 255] = class_idx
            else:
                annotation[im == 255] = train_index.index(class_idx) + 1
        if training == "training":
            num_train += 1
        if training == "validation":
            num_val += 1
        if training == "test":
            num_test += 1
        os.makedirs(os.path.join(save_mask_dir, training), exist_ok=True)
        os.makedirs(os.path.join(save_img_dir, training), exist_ok=True)
        im = Image.fromarray(np.uint8(annotation))
        im.save(os.path.join(save_mask_dir, training, f"{file_name}.png"))
        os.rename(os.path.join(root, file), os.path.join(save_img_dir, training, file))
print(f"{num_train} {num_val} {num_test}")
total = num_train + num_val + num_test
print(
    f"{num_train/total*100:.2f}% Training, {num_val/total*100:.2f}% Validation, {num_test/total*100:.2f}% Testing,"
)