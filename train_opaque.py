
import os
import torch
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import pandas as pd
import numpy as np
from tifffile import tifffile
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
class Img_data(Dataset):
    def __init__(self, root=None, transform = None, split = "train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        train = pd.read_pickle(os.path.join(self.root, 'class2images_train.p'))
        labelmapping = list(train.keys())
        self.labelmapping = labelmapping
        if split == "train":
            data = train
        elif split == "val":
            data = pd.read_pickle(os.path.join(self.root, 'class2images_val.p'))
        elif split == "test":
            data = pd.read_pickle(os.path.join(self.root, 'class2images_test.p'))
        self.data = []
        if "mimic_cxr" in self.root.split("/"):
            for i in labelmapping:
                if i in data:
                    for j in data[i]:
                        self.data.append((j, np.array([float(z) for z in i])))
        else:
            for i in range(len(labelmapping)):
                if labelmapping[i] in data:
                    for j in data[labelmapping[i]]:
                        self.data.append((j, i))
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            with Image.open(os.path.join(self.root, "images", sample[0])) as img:
                image = self.transform(img)
        except:
            print(sample)
        label = sample[1]
        return image, label

from torchvision import transforms, utils
import torchvision
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--backbone', default="densenet")
    parser.add_argument('--trial', default="")
    parser.add_argument('--root', default="")
    parser.add_argument('--ckpt_dir', default="")
    args = parser.parse_args()
    root = args.root

    data = args.root
    traintransform = torchvision.transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=5),
                transforms.ColorJitter(),
                transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None),
                transforms.CenterCrop(size=(224, 224)),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
    testtransform = torchvision.transforms.Compose([
                transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None),
                transforms.CenterCrop(size=(224, 224)),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])

    train_ds = Img_data(root=root, split="train", transform=traintransform)
    val_ds = Img_data(root=root, split="val", transform=testtransform)
    test_ds = Img_data(root=root, split="test", transform=testtransform)
    train_dl = DataLoader(train_ds, batch_size=16, num_workers=8, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=16, num_workers=8, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=16, num_workers=8, shuffle=False, drop_last=False)
    if args.backbone == "densenet":
        net = torchvision.models.densenet121(weights='DEFAULT')
        if args.data != "mimic":
            net.classifier = torch.nn.Linear(1024, len(train_ds.labelmapping))
        else:
            net.classifier = torch.nn.Linear(1024, 13)
    elif args.backbone == "resnet":
        if args.data != "mimic":
            net = torchvision.models.resnet18(weights='DEFAULT')
            net.fc = torch.nn.Linear(512, len(train_ds.labelmapping))
        else:
            net = torchvision.models.resnet50(weights='DEFAULT')
            net.fc = torch.nn.Linear(2048, 13)
    elif args.backbone == "convnext":
        if args.data != "mimic":
            net = torchvision.models.convnext_tiny(weights='DEFAULT')
            net.classifier[2] = torch.nn.Linear(768, len(train_ds.labelmapping))
        else:
            net = torchvision.models.convnext_base(weights='DEFAULT')
            net.classifier[2] = torch.nn.Linear(1024, 13)
    elif args.backbone == "vit":
        if args.data != "mimic":
            net = torchvision.models.vit_b_16(weights='DEFAULT')
            net.heads.head = torch.nn.Linear(768, len(train_ds.labelmapping))
        else:
            net = torchvision.models.vit_b_16(weights='DEFAULT')
            net.heads.head = torch.nn.Linear(768, 13)
    net.cuda()
    if args.data != "mimic":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss()
    best_acc = 0
    optimizer = torch.optim.AdamW([i for i in net.parameters() if i.requires_grad], lr=1e-4)
    m = torch.nn.Sigmoid()
    for epoch in range(args.epoch):
        net.train()
        for idx, (image, label) in enumerate(train_dl):
            label = label.cuda()
            image = image.cuda()
            logits = net(image)
            if args.data == "mimic":
                logits = m(logits)
                loss = criterion(logits, label.float())
            else:
                loss = criterion(logits, label.long())
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        tot = 0
        correct = 0
        net.eval()
        with torch.no_grad():
            for idx, (image, label) in enumerate(val_dl):
                label = label.cuda()
                image = image.cuda()
                if args.data == "mimic":
                    logits = net(image)
                    logits = m(logits)
                    pred = logits > 0.5
                    tot += len(image)
                    correct += (torch.sum((pred == label))/13).item()
                else:
                    logits = net(image).argmax(dim=1)
                    tot += len(image)
                    correct += torch.sum(logits == label).item()
            acc = correct / tot
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), os.path.join(f"{args.ckpt_dir}/best_{args.data}_{args.backbone}_{args.trial}.pt"))
            print(f"Train Epoch: {epoch} \t"
                  f"Accuracy: {(acc):.6f} \t"
                  f"Best Accuracy: {(best_acc):.6f}", flush=True)

    net.load_state_dict(torch.load(os.path.join(f"{args.ckpt_dir}/best_{args.data}_{args.backbone}_{args.trial}.pt")))
    net.eval()
    tot = 0
    correct = 0
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_dl):
            label = label.cuda()
            image = image.cuda()
            if args.data == "mimic":
                logits = net(image)
                logits = m(logits)
                pred = logits > 0.5
                tot += len(image)
                correct += (torch.sum((pred == label))/13).item()
            else:
                logits = net(image).argmax(dim=1)
                tot += len(image)
                correct += torch.sum(logits == label).item()
        acc = correct / tot
    print(args)
    print(f"Test Accuracy: {(acc):.6f}")