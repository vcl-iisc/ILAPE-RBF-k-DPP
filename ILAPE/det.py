"""Script for multi-gpu training for incremental learing."""
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy
from animal_data_loader import AnimalDatasetCombined, ToTensor

from det_sampling import Buffer

from utils import *
from models import *

from pose_resnet import *

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

outputs = []


def train(opt, train_loader, m, criterion, optimizer, writer, phase="Train"):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, _) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        if cfg.LOSS.TYPE == "SmoothL1":
            loss = criterion(output.mul(label_masks), labels.mul(label_masks))

        if cfg.LOSS.get("TYPE") == "MSELoss":
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))

        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(
                writer, loss_logger.avg, acc_logger.avg, opt.trainIters, phase
            )

        # Debug
        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)

        # TQDM
        train_loader.set_description(
            "loss: {loss:.8f} | acc: {acc:.4f}".format(
                loss=loss_logger.avg, acc=acc_logger.avg
            )
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(m, val_loader, opt, cfg, writer, criterion, batch_size=1):
    loss_logger_val = DataLogger()
    acc_logger = DataLogger()

    m.eval()

    val_loader = tqdm(val_loader, dynamic_ncols=True)

    for inps, labels, label_masks, _ in val_loader:
        if isinstance(inps, list):
            inps = [inp.cuda() for inp in inps]

        else:
            inps = inps.cuda()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        loss = criterion(output.mul(label_masks), labels.mul(label_masks))
        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        loss_logger_val.update(loss, batch_size)
        acc_logger.update(acc, batch_size)

        # TQDM
        val_loader.set_description(
            "Loss: {loss:.4f} acc: {acc:.4f}".format(
                loss=loss_logger_val.avg, acc=acc_logger.avg
            )
        )

    val_loader.close()
    return loss_logger_val.avg, acc_logger.avg


def train_kd(
    opt,
    old_train_loader,
    new_train_loader,
    m,
    m_prev,
    criterion,
    optimizer,
    writer,
    phase="Train",
):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    m_prev.eval()

    train_loader = tqdm(zip(old_train_loader, new_train_loader), dynamic_ncols=True)
    for j, (data1, data2) in enumerate(train_loader):
        inps_old, labels_old, label_masks_old, _ = data1
        inps_new, labels_new, label_masks_new, _ = data2

        if isinstance(inps_old, list):
            inps_old = [inp.cuda().requires_grad_() for inp in inps_old]
        else:
            inps_old = inps_old.cuda().requires_grad_()

        if isinstance(inps_old, list):
            inps_new = [inp.cuda().requires_grad_() for inp in inps_new]
        else:
            inps_new = inps_new.cuda().requires_grad_()

        labels_old = labels_old.cuda()
        label_masks_old = label_masks_old.cuda()

        labels_new = labels_new.cuda()
        label_masks_new = label_masks_new.cuda()

        output_old = m(inps_old)

        output_new = m(inps_new)

        output_teacher = m_prev(inps_old)

        loss_orig_old = 0.5 * criterion(
            output_old.mul(label_masks_old), output_teacher.mul(label_masks_old)
        )
        loss_orig_new = 0.5 * criterion(
            output_new.mul(label_masks_new), labels_new.mul(label_masks_new)
        )

        loss_kd = 0.5 * criterion(
            output_old.mul(label_masks_new), labels_old.mul(label_masks_new)
        )

        acc = (
            calc_accuracy(
                output_old.mul(label_masks_old), labels_old.mul(label_masks_old)
            )
            + calc_accuracy(
                output_new.mul(label_masks_new), labels_new.mul(label_masks_new)
            )
        ) / 2

        loss = loss_orig_old + loss_orig_new + loss_kd

        # loss = loss_kd

        if isinstance(inps_new, list):
            batch_size = inps_new[0].size(0) * 2
        else:
            batch_size = inps_new.size(0) * 2

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(
                writer, loss_logger.avg, acc_logger.avg, opt.trainIters, phase
            )

        # Debug
        if opt.debug and not j % 10:
            debug_writing(writer, output_new, labels_new, inps_new, opt.trainIters)

        # TQDM
        train_loader.set_description(
            "loss: {loss:.8f} | acc: {acc:.4f}".format(
                loss=loss_logger.avg, acc=acc_logger.avg
            )
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def main():
    logger.info("******************************")
    logger.info(opt)
    logger.info("******************************")
    logger.info(cfg)
    logger.info("******************************")

    # List of keypoints used
    # 0-2, 1-2, 0-3, 1-3, 5-13, 13-6, 8-14, 14-7, 11-15, 15-9, 12-16, 16-10  => 12 limbs

    keypoint_names = [
        "L_Eye",
        "R_Eye",
        "Nose",
        "L_EarBase",
        "R_EarBase",
        "L_F_Elbow",
        "L_F_Paw",
        "R_F_Paw",
        "R_F_Elbow",
        "L_B_Paw",
        "R_B_Paw",
        "L_B_Elbow",
        "R_B_Elbow",
        "L_F_Knee",
        "R_F_Knee",
        "L_B_Knee",
        "R_B_Knee",
    ]

    # Model Initialize
    # m = preset_model(cfg)
    m = get_pose_net(cfg, True, logger)
    m = nn.DataParallel(m).cuda()

    # Register forward hook
    # m.module.suffle1.register_forward_hook(model_hook)

    if cfg.MODEL.PRETRAINED:
        logger.info(f"Loading model from {cfg.MODEL.PRETRAINED}...")
        m.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))

    if len(cfg.ANIMAL_CLASS_INCREMENTAL) % cfg.INCREMENTAL_STEP != 0:
        print(
            "Number of classes for incremental step is not a multiple of the number of incremental steps!"
        )
        return

    if cfg.LOSS.TYPE == "SmoothL1":
        criterion = nn.SmoothL1Loss().cuda()
    else:
        criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            m.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    elif cfg.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR
    )

    writer = SummaryWriter(".tensorboard/{}-{}".format(opt.exp_id, cfg.FILE_NAME))

    # generating base data loaders
    annot_df = pd.read_csv(cfg.DATASET.ANNOT)

    train_datasets = []
    val_datasets = []

    classes_till_now = []

    filename_list_classes = {}

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    buffer = Buffer(cfg.MEMORY, device)

    for animal_class in cfg.ANIMAL_CLASS_BASE:
        classes_till_now.append(animal_class)
        temp_df = annot_df.loc[annot_df["class"] == animal_class]

        images_list = np.array(temp_df["filename"])
        np.random.seed(121)
        np.random.shuffle(images_list)

        train_images_list = images_list[: int(0.9 * len(images_list))]
        val_images_list = images_list[int(0.9 * len(images_list)) :]

        buffer.add_data(train_images_list)

        train_tempset = AnimalDatasetCombined(
            cfg.DATASET.IMAGES,
            cfg.DATASET.ANNOT,
            train_images_list,
            input_size=(512, 512),
            output_size=(128, 128),
            transforms=torchvision.transforms.Compose([ToTensor()]),
            train=True,
        )

        val_tempset = AnimalDatasetCombined(
            cfg.DATASET.IMAGES,
            cfg.DATASET.ANNOT,
            val_images_list,
            input_size=(512, 512),
            output_size=(128, 128),
            transforms=torchvision.transforms.Compose([ToTensor()]),
            train=False,
        )

        train_datasets.append(train_tempset)
        val_datasets.append(val_tempset)

        filename_list_classes[animal_class] = []
        for i in train_images_list:
            filename_list_classes[animal_class].append(i)

    base_trainset = torch.utils.data.ConcatDataset(train_datasets)
    base_train_loader = torch.utils.data.DataLoader(
        base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True
    )

    base_valset = torch.utils.data.ConcatDataset(val_datasets)
    base_val_loader = torch.utils.data.DataLoader(
        base_valset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE
    )

    opt.trainIters = 0
    opt.val_iters = 0

    best_acc = 0.0
    best_model_weights = deepcopy(m.state_dict())
    logger.info(
        f"############# Starting Base Training with base classes {cfg.ANIMAL_CLASS_BASE} ########################"
    )

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        logger.info(
            f"############# Starting Epoch {opt.epoch} | LR: {current_lr} #############"
        )

        # Training

        train_loss, train_acc = train(
            opt, base_train_loader, m, criterion, optimizer, writer, phase="Base_Train"
        )
        logger.epochInfo("Base_Train", opt.epoch, train_loss, train_acc)

        lr_scheduler.step()

        # Prediction Test
        with torch.no_grad():
            val_loss, val_acc = validate(
                m,
                base_val_loader,
                opt,
                cfg,
                writer,
                criterion,
                batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
            )
            logger.info(
                f"##### Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####"
            )

            # Tensorboard
            if opt.board:
                board_writing(writer, val_loss, val_acc, opt.val_iters, "Base_Val")

            opt.val_iters += 1

        if val_acc > best_acc:
            # Save best weights
            best_model_weights = deepcopy(m.state_dict())
            best_acc = val_acc

        # Time to add DPG
        if i == cfg.TRAIN.DPG_MILESTONE:
            torch.save(
                best_model_weights,
                "./exp/{}-{}/model_{}.pth".format(opt.exp_id, cfg.FILE_NAME, "Base"),
            )
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.TRAIN.LR

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1
            )

            base_trainset = torch.utils.data.ConcatDataset(train_datasets)
            base_train_loader = torch.utils.data.DataLoader(
                base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True
            )

    torch.save(
        best_model_weights,
        "./exp/{}-{}/model_{}.pth".format(opt.exp_id, cfg.FILE_NAME, "Base"),
    )

    m.load_state_dict(best_model_weights)
    m = nn.DataParallel(m).cuda()
    m_prev = deepcopy(m)
    m_prev = nn.DataParallel(m_prev).cuda()

    # Time to do incremental learning
    val_datasets_incremental = []

    val_datasets_incremental.append(base_valset)

    for inc_step in range(
        int(len(cfg.ANIMAL_CLASS_INCREMENTAL) / cfg.INCREMENTAL_STEP)
    ):
        if cfg.TRAIN_INCREMENTAL.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
        elif cfg.TRAIN_INCREMENTAL.OPTIMIZER == "rmsprop":
            optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
        elif cfg.TRAIN.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(m.parameters(), lr=cfg.TRAIN.LR)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN_INCREMENTAL.LR_STEP,
            gamma=cfg.TRAIN_INCREMENTAL.LR_FACTOR,
        )

        animal_classes = cfg.ANIMAL_CLASS_INCREMENTAL[
            inc_step * cfg.INCREMENTAL_STEP : (inc_step + 1) * cfg.INCREMENTAL_STEP
        ]

        curr_train_datasets = []
        non_augmented_datasets = []
        curr_val_datasets = []

        images_list = buffer.get_data(cfg.MEMORY)

        old_train_datasets = AnimalDatasetCombined(
            cfg.DATASET.IMAGES,
            cfg.DATASET.ANNOT,
            images_list,
            input_size=(512, 512),
            output_size=(128, 128),
            transforms=torchvision.transforms.Compose([ToTensor()]),
            train=True,
        )
        old_train_loader = torch.utils.data.DataLoader(
            old_train_datasets,
            batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE,
            shuffle=True,
        )

        new_train_datasets = []
        for animal_class in animal_classes:
            classes_till_now.append(animal_class)
            temp_df = annot_df.loc[annot_df["class"] == animal_class]

            images_list = np.array(temp_df["filename"])
            np.random.shuffle(images_list)

            train_images_list = images_list[: int(0.9 * len(images_list))]
            buffer.add_data(train_images_list)
            val_images_list = images_list[int(0.9 * len(images_list)) :]

            train_tempset = AnimalDatasetCombined(
                cfg.DATASET.IMAGES,
                cfg.DATASET.ANNOT,
                train_images_list,
                input_size=(512, 512),
                output_size=(128, 128),
                transforms=torchvision.transforms.Compose([ToTensor()]),
                train=True,
            )

            val_tempset = AnimalDatasetCombined(
                cfg.DATASET.IMAGES,
                cfg.DATASET.ANNOT,
                val_images_list,
                input_size=(512, 512),
                output_size=(128, 128),
                transforms=torchvision.transforms.Compose([ToTensor()]),
                train=False,
            )

            new_train_datasets.append(train_tempset)
            curr_train_datasets.append(train_tempset)
            non_augmented_datasets.append(train_tempset)
            curr_val_datasets.append(val_tempset)

            filename_list_classes[animal_class] = []
            for j in train_images_list:
                filename_list_classes[animal_class].append(j)

        for val_set in curr_val_datasets:
            val_datasets_incremental.append(val_set)

        incremental_trainset = torch.utils.data.ConcatDataset(curr_train_datasets)
        incremental_train_loader = torch.utils.data.DataLoader(
            incremental_trainset,
            batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE,
            shuffle=True,
        )

        incremental_valset = torch.utils.data.ConcatDataset(val_datasets_incremental)
        incremental_val_loader_overall = torch.utils.data.DataLoader(
            incremental_valset, batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE
        )

        incremental_val_loaders_individual = []
        for val_set in val_datasets_incremental:
            temp_loader = torch.utils.data.DataLoader(
                val_set, batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE
            )
            incremental_val_loaders_individual.append(temp_loader)

        opt.trainIters = 0
        opt.val_iters = 0

        best_acc = 0.0
        best_model_weights = deepcopy(m.state_dict())

        logger.info(
            f"#######################################################################################################################"
        )
        logger.info(
            f"############# Starting Incremental Training step {inc_step} with incremental classes {animal_classes} ########################"
        )

        for ep in range(
            cfg.TRAIN_INCREMENTAL.BEGIN_EPOCH, cfg.TRAIN_INCREMENTAL.END_EPOCH
        ):
            opt.epoch = ep
            current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

            logger.info(
                f"############# Starting Epoch {opt.epoch} | LR: {current_lr} #############"
            )

            # Training

            train_loss, train_acc = train_kd(
                opt,
                old_train_loader,
                incremental_train_loader,
                m,
                m_prev,
                criterion,
                optimizer,
                writer,
                phase="Incremental_Train" + str(inc_step),
            )
            logger.epochInfo(
                "Incremental_Train" + str(inc_step), opt.epoch, train_loss, train_acc
            )

            lr_scheduler.step()

            # Prediction Test
            with torch.no_grad():
                for class_num in range(len(incremental_val_loaders_individual)):
                    val_loss, val_acc = validate(
                        m,
                        incremental_val_loaders_individual[class_num],
                        opt,
                        cfg,
                        writer,
                        criterion,
                        batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE,
                    )
                    logger.info(
                        f"##### Evaluating on class {class_num} Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####"
                    )

                val_loss, val_acc = validate(
                    m,
                    incremental_val_loader_overall,
                    opt,
                    cfg,
                    writer,
                    criterion,
                    batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                )
                logger.info(
                    f"##### Evaluating on all classes Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####"
                )

                if opt.board:
                    board_writing(
                        writer,
                        val_loss,
                        val_acc,
                        opt.val_iters,
                        "Incremental_Val" + str(inc_step),
                    )

                opt.val_iters += 1

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_weights = deepcopy(m.state_dict())

            # Time to add DPG
            if i == cfg.TRAIN.DPG_MILESTONE:
                torch.save(
                    best_model_weights,
                    "./exp/{}-{}/model_{}.pth".format(
                        opt.exp_id, cfg.FILE_NAME, "Incremental" + str(inc_step)
                    ),
                )

        torch.save(
            best_model_weights,
            "./exp/{}-{}/model_{}.pth".format(
                opt.exp_id, cfg.FILE_NAME, "Incremental" + str(inc_step)
            ),
        )
        m.load_state_dict(best_model_weights)
        m = nn.DataParallel(m).cuda()
        m_prev = deepcopy(m)
        m_prev = nn.DataParallel(m_prev).cuda()

    torch.save(
        best_model_weights,
        "./exp/{}-{}/final_weights.pth".format(opt.exp_id, cfg.FILE_NAME),
    )


def preset_model(cfg):
    if cfg.MODEL.TYPE == "custom":
        model = DeepLabCut()
    else:
        model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    logger.info("Create new model")
    logger.info("=> init weights")
    if cfg.MODEL.TYPE != "custom":
        model._initialize()

    return model


if __name__ == "__main__":
    main()
