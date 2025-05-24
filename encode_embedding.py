
import os
import torch
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import numpy as np
import pickle
from tifffile import tifffile
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
        for i in range(len(labelmapping)):
            if labelmapping[i] in data:
                for j in data[labelmapping[i]]:
                    self.data.append((j, i))
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        with Image.open(os.path.join(self.root, "images", sample[0])) as img:
            image = self.transform(img)
        label = sample[1]
        return image, label

from torchvision import transforms, utils
import torchvision
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    parser.add_argument('--trial', default="")
    parser.add_argument('--backbone', default="densenet")
    parser.add_argument('--root', default="")
    parser.add_argument('--result_dir', default="")
    args = parser.parse_args()

    root = args.root

    testtransform = torchvision.transforms.Compose([
                transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None),
                transforms.CenterCrop(size=(224, 224)),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])

    train_ds = Img_data(root=root, split="train", transform=testtransform)
    val_ds = Img_data(root=root, split="val", transform=testtransform)
    test_ds = Img_data(root=root, split="test", transform=testtransform)
    train_dl = DataLoader(train_ds, batch_size=16, num_workers=8, shuffle=False, drop_last=False)
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
    net.load_state_dict(torch.load(os.path.join(f"./ckpt/best_{args.data}_{args.backbone}_{args.trial}.pt")))
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0

    with torch.no_grad():
        concept_emb = []
        rep_emb = []
        label_train = []
        for idx, (image, label) in enumerate(train_dl):
            label = label.cuda()
            image = image.cuda()
            #logits = net(image)
            if args.backbone == "densenet":
                features = net.features(image)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
            elif args.backbone == "convnext":
                features = net.features(image)
                out = F.adaptive_avg_pool2d(features, (1, 1))
                out = net.classifier[:2](out)
            elif args.backbone == "resnet":
                x = net.conv1(image)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                x = net.layer4(x)
                out = net.avgpool(x)
            elif args.backbone == "vit":
                x = net._process_input(image)
                n = x.shape[0]
                batch_class_token = net.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = net.encoder(x)
                out = x[:, 0]
                #out = net.heads(x)
            rep = torch.flatten(out, 1).detach().cpu().numpy()
            rep_emb.append(rep)
            label_train.extend(label.cpu().tolist())
        #concept_emb = np.concatenate(concept_emb, axis=0)
        rep_train = np.concatenate(rep_emb, axis=0)

        rep_emb = []
        label_val = []
        for idx, (image, label) in enumerate(val_dl):
            label = label.cuda()
            image = image.cuda()
            if args.backbone == "densenet":
                features = net.features(image)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
            elif args.backbone == "convnext":
                features = net.features(image)
                out = F.adaptive_avg_pool2d(features, (1, 1))
                out = net.classifier[:2](out)
            elif args.backbone == "resnet":
                x = net.conv1(image)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                x = net.layer4(x)
                out = net.avgpool(x)
            elif args.backbone == "vit":
                x = net._process_input(image)
                n = x.shape[0]
                batch_class_token = net.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = net.encoder(x)
                out = x[:, 0]
                #out = net.heads(x)
            rep = torch.flatten(out, 1).detach().cpu().numpy()
            rep_emb.append(rep)
            label_val.extend(label.cpu().tolist())
        rep_val = np.concatenate(rep_emb, axis=0)

        rep_emb = []
        label_test = []
        test_rep = []
        for idx, (image, label) in enumerate(test_dl):
            label = label.cuda()
            image = image.cuda()
            if args.backbone == "densenet":
                features = net.features(image)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
            elif args.backbone == "convnext":
                features = net.features(image)
                out = F.adaptive_avg_pool2d(features, (1, 1))
                out = net.classifier[:2](out)
            elif args.backbone == "resnet":
                x = net.conv1(image)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                x = net.layer4(x)
                out = net.avgpool(x)
            elif args.backbone == "vit":
                x = net._process_input(image)
                n = x.shape[0]
                batch_class_token = net.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = net.encoder(x)
                out = x[:, 0]
                #out = net.heads(x)
            rep = torch.flatten(out, 1).detach().cpu().numpy()
            rep_emb.append(rep)
            label_test.extend(label.cpu().tolist())
        rep_test = np.concatenate(rep_emb, axis=0)
        #test_rep = np.concatenate(test_rep, axis=0)

    with open(f"{args.result_dir}/{args.data}/target.pkl", "wb") as f:
        pickle.dump((label_train, label_val, label_test), f)
    os.makedirs(f"{args.result_dir}/{args.data}/{args.backbone}", exist_ok=True)
    os.makedirs(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}", exist_ok=True)
    np.save(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_train.npy", rep_train)
    np.save(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_val.npy", rep_val)
    np.save(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_test.npy", rep_test)
