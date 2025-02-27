from torchvision.datasets import VOCDetection

import torch

class VOCDataset(VOCDetection):
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    def __init__(self, root, year='2012', image_set='train', download=False,
                 transform=None):
        super().__init__(root, year, image_set, download, transform)
        self.class_to_index = {cls: idx + 1 for idx, cls in enumerate(self.VOC_CLASSES)}

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        boxes = []
        labels = []
        for obj in target["annotation"]["object"]:
            try:
                bndbox = obj['bndbox']
                label = obj['name']
                xmax = int(bndbox["xmax"])
                xmin = int(bndbox["xmin"])
                ymax = int(bndbox["ymax"])
                ymin = int(bndbox["ymin"])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_index[label])
            except (KeyError, ValueError) as e:
                print(f"skipping invalid object: {e}")

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target