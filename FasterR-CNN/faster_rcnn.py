import os
import torch

from process_data import VOCDataset

from pprint import pprint
from huggingface_hub.utils.tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, \
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.transforms.v2 import ToTensor

def collate_fn(batch):
    """
        change how DataLoader collate data from
        (image1, {boxes1, labels1})
        (image2, {boxes2, labels2})
        ...
        to
        (image1, image2,...)
        ({boxes1, labels1}), {boxes2, labels2},...)
    """
    # model expects list of tensor
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets

def save_checkpoint(epoch, model, optimizer, scheduler, best_map, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_map': best_map
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_map']

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader)
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        losses = model(images, targets)

        total_loss = sum([loss for loss in losses.values()])
        progress_bar.set_description(f"Loss: {total_loss:.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return total_loss

def evaluate(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [image.to(device) for image in images]
            predictions = model(images)
            #avoid accumulate GPU usage in metric, covert back to cpu
            predictions = [{k: v.to("cpu") for k, v in pred.items()} for pred in predictions]
            metric.update(predictions, targets)

    return metric.compute()

def train_model():
    batch_size = 1
    epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.005
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )

    #Load data
    transform = transforms.Compose([
        ToTensor()
    ])
    train_data = VOCDataset(root="../data", image_set='train', transform=transform)
    val_data = VOCDataset(root="../data", image_set='val', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)

    num_classes = len(train_data.class_to_index) + 1 #FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training state
    start_epoch = 0
    best_map = 0.0

    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        start_epoch, best_map = load_checkpoint(model, optimizer, lr_scheduler,
                                              checkpoint_path)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate
        eval_metrics = evaluate(model, val_loader, device)
        pprint(eval_metrics)
        current_map = eval_metrics['map']

        # Save best model
        if current_map > best_map:
            best_map = current_map
            save_checkpoint(epoch, model, optimizer, lr_scheduler, best_map,
                          os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New best model, mAP: {best_map:.4f}")

        # Save last checkpoint
        save_checkpoint(epoch, model, optimizer, lr_scheduler, best_map,
                       os.path.join(checkpoint_dir, 'last_checkpoint.pth'))

        # Update learning rate
        lr_scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"Completed! Best mAP: {best_map:.4f}")
    return model


if __name__ == '__main__':
    model = train_model()


