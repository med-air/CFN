import numpy as np
import torch
import os.path as osp
from dncbm.custom_pipeline import Pipeline
import os
from pathlib import Path
import torch
import numpy as np
import math
import datetime
import sys
sys.path.append('./sparse_autoencoder/')
from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
import wandb
from time import time

from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init
import os.path as osp
from sklearn.svm import SVC
from tqdm import tqdm
import argparse
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    parser.add_argument('--backbone', default="densenet")
    parser.add_argument('--trial', default="")
    parser.add_argument('--save_dir', default="")
    args = parser.parse_args()


    data_dir_activations = {}
    os.makedirs(osp.join('.', 'activations_img'), exist_ok=True)
    os.makedirs(osp.join('.', 'activations_img', args.data), exist_ok=True)
    os.makedirs(osp.join('.', 'activations_img', args.data, args.backbone), exist_ok=True)
    os.makedirs(osp.join('.', 'activations_img', args.data, args.backbone, f'out{args.trial}'), exist_ok=True)
    data_dir_activations["img"] = osp.join('.', 'activations_img', args.data, args.backbone, f'out{args.trial}')
    save_dir_activations = f"{args.save_dir}/{args.data}/{args.backbone}/{args.trial}/"
    train = torch.from_numpy(np.load(save_dir_activations + "img_emb_train.npy"))
    torch.save(train, osp.join(data_dir_activations["img"], "train"))
    train_val = torch.from_numpy(np.load(save_dir_activations + "img_emb_val.npy"))
    torch.save(train_val, osp.join(data_dir_activations["img"], "train_val"))
    val = torch.from_numpy(np.load(save_dir_activations + "img_emb_test.npy"))
    torch.save(val, osp.join(data_dir_activations["img"], "val"))

    start_time = time()
    autoencoder_input_dim: int = train.shape[1]
    n_learned_features = int(autoencoder_input_dim * 8)
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                    n_learned_features=n_learned_features, n_components=1).cuda()
    print(f"Autoencoder created at {time() - start_time} seconds")
    print(f"------------Getting Image activations from model: {'densenet121'}")

    # We use a loss reducer, which simply adds up the losses from the underlying loss functions.
    loss = LossReducer(LearnedActivationsL1Loss(
        l1_coefficient=float(3e-5), ), L2ReconstructionLoss(), )
    print(f"Loss created at {time() - start_time} seconds")

    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=float(5e-4),
        betas=(0.9,
               0.999),
        eps=1e-8,
        weight_decay=0.0,
        has_components_dim=True,
    )

    print(f"Optimizer created at {time() - start_time} seconds")
    actual_resample_interval = 1
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=math.inf,
        n_learned_features=n_learned_features, resample_epoch_freq=500000,
        resample_dataset_size=819200,
    )

    print(f"Activation resampler created at {time() - start_time} seconds")

    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(data_dir_activations["img"]),
        loss=loss,
        optimizer=optimizer,
        device="cuda",
        args={},
    )
    print(f"Pipeline created at {time() - start_time} seconds")

    fnames = os.listdir(data_dir_activations["img"])

    train_fnames = []
    train_val_fnames = []
    for fname in fnames:
        if fname.startswith(f"train_val"):
            train_val_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))
        elif fname.startswith(f"train"):
            train_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))

    print(f"Train and Train_val fnames created at {time() - start_time} seconds")

    # It takes the train activations and inside split it into train_activations and train_val_activations
    pipeline.run_pipeline(
        train_batch_size=int(4096),
        checkpoint_frequency=500000,
        val_frequency=50000,
        num_epochs=200,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=500000,
    )

    print(f"-------total time taken------ {np.round(time() - start_time, 3)}")

    index = []
    learned_activations = []
    for batch_idx in range(len(train) // 512 + 1):
        train_batch = train[batch_idx * 512: (batch_idx + 1) * 512].cuda()
        learned_activations_batch = autoencoder(train_batch).learned_activations[:, 0, :].detach().cpu()
        concept_batch = autoencoder(train_batch)
        index_batch = torch.abs(concept_batch.learned_activations).sum(dim=[0, 1]).cpu() > 0
        index.append(index_batch.unsqueeze(0))
        learned_activations.append(learned_activations_batch)
    learned_activations = torch.concat(learned_activations, dim=0)
    index = torch.concat(index, dim=0)
    index = index.sum(dim=0) > 0

    np.save(save_dir_activations + "/learned_activation.npy", learned_activations.detach().cpu().numpy())

