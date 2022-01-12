"""Script for training for incremental learing."""
import time
from itertools import zip_longest
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel

from dppy.finite_dpps import FiniteDPP

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

from utils import *


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


def train_balanced(
    opt,
    old_train_loader,
    new_train_loader,
    m,
    m_prev,
    criterion,
    optimizer,
    writer,
    phase="Balanced_Finetuning",
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

        # print(labels_old.shape, label_masks_old.shape, inps_old.shape)
        # print(labels_new.shape, label_masks_new.shape, inps_new.shape)

        output_old = m(inps_old)

        output_new = m(inps_new)

        output_teacher = m_prev(inps_new)

        # print(output_old.shape, output_new.shape, output_teacher.shape)

        # print(labels_new.mul(label_masks_new))

        loss_orig_old = 0.5 * criterion(
            output_old.mul(label_masks_old), labels_old.mul(label_masks_old)
        )
        loss_orig_new = 0.5 * criterion(
            output_new.mul(label_masks_new), labels_new.mul(label_masks_new)
        )

        loss_kd = 0.5 * criterion(
            output_new.mul(label_masks_new), output_teacher.mul(label_masks_new)
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


def train_kd(opt, train_loader, m, m_prev, criterion, optimizer, writer, phase="Train"):
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

        output_teacher = m_prev(inps)

        loss_orig = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))

        loss_kd = 0.5 * criterion(
            output.mul(label_masks), output_teacher.mul(label_masks)
        )

        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        loss = loss_orig + loss_kd

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


def train_icarl(
    opt,
    new_train_loader,
    old_train_loader,
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

    train_loader = tqdm(
        zip_longest(new_train_loader, old_train_loader, fillvalue=None),
        total=max(len(new_train_loader), len(old_train_loader)),
        dynamic_ncols=True,
    )

    for i, (data_new, data_old) in enumerate(train_loader):

        if data_new:
            inps_new, labels_new, label_masks_new, _ = data_new
            if isinstance(inps_new, list):
                inps_new = [inp_new.cuda().requires_grad_() for inp_new in inps_new]
            else:
                inps_new = inps_new.cuda().requires_grad_()
            labels_new = labels_new.cuda()
            label_masks_new = label_masks_new.cuda()

            output_new = m(inps_new)

        if data_old:
            inps_old, labels_old, label_masks_old, _ = data_old
            if data_old:
                if isinstance(inps_old, list):
                    inps_old = [inp_old.cuda().requires_grad_() for inp_old in inps_old]
                else:
                    inps_old = inps_old.cuda().requires_grad_()
                labels_old = labels_old.cuda()
                label_masks_old = label_masks_old.cuda()

                output_old = m(inps_old)
                output_teacher = m_prev(inps_old)

        if data_new:
            loss_new = 0.5 * criterion(
                output_new.mul(label_masks_new), labels_new.mul(label_masks_new)
            )

            acc_new = 0.5 * calc_accuracy(
                output_new.mul(label_masks_new), labels_new.mul(label_masks_new)
            )

        if data_old:
            loss_kd = criterion(
                output_old.mul(label_masks_old), output_teacher.mul(label_masks_old)
            )

            acc_kd = calc_accuracy(
                output_old.mul(label_masks_old), labels_old.mul(label_masks_old)
            )

        if isinstance(inps_old, list):
            batch_size = inps_old[0].size(0)
        else:
            batch_size = inps_old.size(0)

        if data_new and data_old:
            loss = loss_new + loss_kd
            acc = (acc_new + acc_kd) / 2

            loss_logger.update(loss.item(), 2 * batch_size)
            acc_logger.update(acc, 2 * batch_size)

        elif data_new and not data_old:
            loss = loss_new
            acc = acc_new

            loss_logger.update(loss.item(), batch_size)
            acc_logger.update(acc, batch_size)

        elif not data_new and data_old:
            loss = loss_kd

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
            debug_writing(writer, output_new, labels_new, inps_new, opt.trainIters)

        # TQDM
        train_loader.set_description(
            "loss: {loss:.8f} | acc: {acc:.4f}".format(
                loss=loss_logger.avg, acc=acc_logger.avg
            )
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def train_kd_mixup(
    opt, train_loader, m, m_prev, criterion, optimizer, writer, phase="Train"
):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    m_prev.eval()
    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, _) in enumerate(train_loader):
        inps_flipped = torch.flip(inps, (3,))

        t = np.random.uniform(0, 1, size=(inps.shape[0],))

        inps_mix_up = []

        for j in range(inps.shape[0]):
            inps_mix_up.append(
                t[j] * inps[j].detach().cpu().numpy()
                + (1 - t[j]) * inps_flipped[j].detach().cpu().numpy()
            )

        inps_mix_up = np.array(inps_mix_up)

        inps_mix_up = torch.FloatTensor(inps_mix_up)

        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
            inps_mix_up = [inp.cuda().requires_grad_() for inp in inps_mix_up]
        else:
            inps = inps.cuda().requires_grad_()
            inps_mix_up = inps_mix_up.cuda().requires_grad_()

        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        loss_gt = criterion(output.mul(label_masks), labels.mul(label_masks))

        output_teacher = m_prev(inps_mix_up)
        output = m(inps_mix_up)

        loss_kd = criterion(output.mul(label_masks), output_teacher.mul(label_masks))

        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        loss = 0.25 * loss_kd + 0.5 * loss_gt

        if isinstance(inps, list):
            batch_size = 2 * inps[0].size(0)
        else:
            batch_size = 2 * inps.size(0)

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


def cluster_dpp(keypoints_list, keypoints_to_fname, samples_per_class, animal_class):
    n_clusters = int(np.ceil(samples_per_class / 51))

    if animal_class == "horse":
        n_clusters += 1

    k_orig = int(samples_per_class / n_clusters)

    print(f"Samples expected: {samples_per_class}")

    km = KMeans(n_clusters=n_clusters)
    km.fit(keypoints_list)

    keypoint_list_clusters = []

    for clus in range(n_clusters):
        temp1 = keypoints_list[ClusterIndicesNumpy(clus, km.labels_)]
        # print(temp1.shape)
        k = min(k_orig, np.linalg.matrix_rank(temp1))

        Phi = temp1.dot(temp1.T)

        DPP = FiniteDPP("likelihood", **{"L": Phi})
        # for _ in range(5):
        DPP.sample_exact_k_dpp(size=k)

        max_det = 0
        index_of_samples = DPP.list_of_samples[0]

        # for j in range(5):
        #     matrix = np.array(Phi)
        #     submatrix = matrix[np.ix_(DPP.list_of_samples[j], DPP.list_of_samples[j])]
        #     try:
        #         det = np.linalg.det(submatrix)
        #         if det > max_det:
        #             max_det = det
        #             index_of_samples = DPP.list_of_samples[j]
        #     except:
        #         continue

        temp = temp1[index_of_samples]

        for j in temp:
            keypoint_list_clusters.append(j)

    images_list = []
    for j in keypoint_list_clusters:
        images_list.append(keypoints_to_fname[str(j)])

    return images_list


def rbf_dpp(keypoints_list, keypoints_to_fname, samples_per_class, gamma=50):
    since = time.time()
    Phi = rbf_kernel(keypoints_list, gamma=gamma)

    k = samples_per_class

    eig_vals, eig_vecs = np.linalg.eigh(Phi)

    # DPP = FiniteDPP("likelihood", **{"L": Phi})
    DPP = FiniteDPP("likelihood", **{"L_eig_dec": (eig_vals, eig_vecs)})
    for _ in range(5):
        DPP.sample_exact_k_dpp(size=k)

    max_det = 0
    index_of_samples = DPP.list_of_samples[0]
    print(f"Time Taken for sampling points: {time.time() - since}")

    for j in range(5):
        matrix = np.array(Phi)
        submatrix = matrix[np.ix_(DPP.list_of_samples[j], DPP.list_of_samples[j])]
        try:
            det = np.linalg.det(submatrix)
            if det > max_det:
                max_det = det
                index_of_samples = DPP.list_of_samples[j]
        except:
            continue

    temp = keypoints_list[index_of_samples]
    images_list = []
    for j in temp:
        images_list.append(keypoints_to_fname[str(j)])
    print(len(images_list))

    return images_list


def herding(keypoints_list, animal_list, samples_per_class):
    animal_avg = np.mean(keypoints_list, axis=0)

    final_animal_vec = calc_dist(animal_avg, animal_list)

    final_animal_vec.sort(key=lambda x: x[1])

    images_list = []

    for vec in final_animal_vec[:samples_per_class]:
        images_list.append(vec[0][0])

    return images_list


def cluster(keypoints_list, keypoints_to_fname, samples_per_class, cfg):
    plotX = pd.DataFrame(np.array(keypoints_list))
    plotX.columns = np.arange(0, np.array(keypoints_list).shape[1])

    if cfg.SAMPLING.N_CLUSTERS == 0:
        n_clusters = int(np.ceil(samples_per_class / 51))

    else:
        n_clusters = cfg.SAMPLING.N_CLUSTERS

    km = KMeans(n_clusters=n_clusters)
    km.fit(keypoints_list)

    pca = PCA(n_components=2)
    PCs_2d = pd.DataFrame(pca.fit_transform(plotX))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    plotX = pd.concat([plotX, PCs_2d], axis=1, join="inner")

    clusters = km.predict(keypoints_list)
    plotX["Cluster"] = clusters

    if cfg.SAMPLING.N_CLUSTERS == 0:
        samples_per_cluster = min(samples_per_class, 51)
    else:
        samples_per_cluster = int(samples_per_class / n_clusters)

    keypoint_list_clusters = []
    clusters_data = {}
    for clus in range(n_clusters):
        if cfg.SAMPLING.CLUSTER_PROPORTION != "same":
            samples_per_cluster = samples_per_class * int(
                len(keypoints_list[ClusterIndicesNumpy(clus, km.labels_)])
                / len(keypoints_list)
                + 1
            )

        if cfg.SAMPLING.CLUSTER_SAMPLING == "random":
            temp = keypoints_list[ClusterIndicesNumpy(clus, km.labels_)][
                :samples_per_cluster
            ]

        elif cfg.SAMPLING.CLUSTER_SAMPLING == "dist":
            d = km.transform(keypoints_list)[:, clus]
            dist_tup = list(enumerate(d))
            l = sorted(dist_tup, key=lambda i: i[1])

            rng = l[-1][1] - l[0][1]

            temp1, temp2, temp3, temp4 = [], [], [], []
            for dist in l:
                if dist[1] < l[0][1] + 0.25 * rng:
                    temp1.append(keypoints_list[dist[0]])
                elif dist[1] >= l[0][1] + 0.25 * rng and dist[1] < l[0][1] + 0.50 * rng:
                    temp2.append(keypoints_list[dist[0]])
                elif dist[1] >= l[0][1] + 0.50 * rng and dist[1] < l[0][1] + 0.75 * rng:
                    temp3.append(keypoints_list[dist[0]])
                else:
                    temp4.append(keypoints_list[dist[0]])
            total_len = len(temp1) + len(temp2) + len(temp3) + len(temp4)
            samples_1 = round(samples_per_cluster * (len(temp1) / total_len))
            samples_2 = round(samples_per_cluster * (len(temp2) / total_len))
            samples_3 = round(samples_per_cluster * (len(temp3) / total_len))
            samples_4 = round(samples_per_cluster * (len(temp4) / total_len))

            temp1 = temp1[:samples_1]
            temp2 = temp2[:samples_2]
            temp3 = temp3[:samples_3]
            temp4 = temp4[:samples_4]

            temp3.extend(temp4)
            temp2.extend(temp3)
            temp1.extend(temp2)
            temp = temp1

        elif cfg.SAMPLING.CLUSTER_SAMPLING == "dpp":
            temp1 = keypoints_list[ClusterIndicesNumpy(clus, km.labels_)]
            Phi = temp1.dot(temp1.T)

            DPP = FiniteDPP("likelihood", **{"L": Phi})
            k = 50
            for _ in range(5):
                DPP.sample_exact_k_dpp(size=k)

            max_det = 0
            index_of_samples = DPP.list_of_samples[0]

            for j in range(5):
                matrix = np.array(Phi)
                submatrix = matrix[
                    np.ix_(DPP.list_of_samples[j], DPP.list_of_samples[j])
                ]
                try:
                    det = np.linalg.det(submatrix)
                    if det > max_det:
                        max_det = det
                        index_of_samples = DPP.list_of_samples[j]
                except:
                    continue

            temp = temp1[index_of_samples]

        else:
            d = km.transform(keypoints_list)[:, clus]
            ind = np.argsort(d)[::][:samples_per_cluster]

            temp = keypoints_list[ind]

        clusters_data[str(clus)] = plotX[plotX["Cluster"] == clus]
        for j in temp:
            keypoint_list_clusters.append(j)

    fig, ax = plt.subplots()
    for key in clusters_data.keys():
        ax.scatter(
            clusters_data[key]["PC1_2d"], clusters_data[key]["PC2_2d"], label=key,
        )
    centroids = km.cluster_centers_
    centroids = pca.transform(np.array(centroids))
    ax.scatter(centroids[:, 0], centroids[:, 1], s=80)

    plotS = pd.DataFrame(np.array(keypoint_list_clusters))
    PCs_2dS = pd.DataFrame(pca.transform(plotS))
    PCs_2dS.columns = ["PC1_2d", "PC2_2d"]
    plotS = pd.concat([plotS, PCs_2dS], axis=1, join="inner")

    ax.legend()
    fig.savefig(
        "./exp/{}-{}/clustering_incremental_step_{}_{}.png".format(
            opt.exp_id, cfg.FILE_NAME, i, animal_class
        )
    )

    ax.scatter(plotS["PC1_2d"], plotS["PC2_2d"], label="sampled", marker="x")
    ax.legend()
    fig.savefig(
        "./exp/{}-{}/clustering_incremental_step_sampled_{}_{}.png".format(
            opt.exp_id, cfg.FILE_NAME, i, animal_class
        )
    )

    images_list = []
    for j in keypoint_list_clusters:
        images_list.append(keypoints_to_fname[str(j)])

    print(len(images_list))

    save_images(
        images_list[:5],
        images_path=cfg.DATASET.IMAGES,
        annot_path=cfg.DATASET.ANNOT,
        save_dir="./exp/{}-{}/images_visualizations/".format(opt.exp_id, cfg.FILE_NAME),
        animal_class=animal_class,
    )

    return images_list


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
    if cfg.MODEL.TYPE == "FastPose":
        m = preset_model(cfg)
        m = nn.DataParallel(m).cuda()
    elif cfg.MODEL.TYPE == "pose_resnet":
        m = get_pose_net(cfg, True, logger)
        m = nn.DataParallel(m).cuda()

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

    for animal_class in cfg.ANIMAL_CLASS_BASE:
        classes_till_now.append(animal_class)
        temp_df = annot_df.loc[annot_df["class"] == animal_class]

        images_list = np.array(temp_df["filename"])
        np.random.seed(121)
        np.random.shuffle(images_list)

        train_images_list = images_list[: int(0.9 * len(images_list))]
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

    memory_dict = {}
    classes_visited = []

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

    ###################################
    # Time to do incremental learning #
    ###################################

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
        old_class_datasets = []
        curr_val_datasets = []
        augmented_datasets = []

        samples_per_class_old = int(cfg.MEMORY / len(classes_till_now))

        overall_images_list_new_classes = []
        overall_images_list_old_classes = []

        for animal_class in classes_till_now:
            if animal_class in classes_visited:
                if cfg.MEMORY_TYPE == "growing":
                    images_list = memory_dict[animal_class]
                else:
                    print("Computing images list for fixed memory")
                    animal_list = []
                    keypoints_list = []
                    keypoints_to_fname = {}
                    fname_list = []

                    for fname in memory_dict[animal_class]:
                        temp = (fname, [])
                        for keypt in keypoint_names:
                            temp[1].append(
                                [
                                    annot_df.loc[annot_df["filename"] == fname][
                                        keypt + "_x"
                                    ].item(),
                                    annot_df.loc[annot_df["filename"] == fname][
                                        keypt + "_y"
                                    ].item(),
                                    annot_df.loc[annot_df["filename"] == fname][
                                        keypt + "_visible"
                                    ].item(),
                                ]
                            )

                        animal_list.append(temp)

                        if (
                            cfg.SAMPLING.STRATERGY == "random"
                            or cfg.SAMPLING.STRATERGY == "feature_dpp"
                            or cfg.SAMPLING.STRATERGY == "dpp"
                            or cfg.SAMPLING.STRATERGY == "herding"
                            or cfg.SAMPLING.STRATERGY == "true_random"
                        ):
                            fname_list.append(temp[0])

                        if cfg.SAMPLING.STRATERGY == "herding":
                            keypoints_list.append(temp[1])

                        if (
                            cfg.SAMPLING.STRATERGY == "cluster"
                            or cfg.SAMPLING.STRATERGY == "dpp"
                            or cfg.SAMPLING.STRATERGY == "rbf-dpp"
                        ):
                            temp_2 = np.array(temp[1])
                            temp_2 = temp_2.flatten()
                            keypoints_list.append(temp_2)
                            keypoints_to_fname[str(temp_2)] = fname

                    keypoints_list = np.array(keypoints_list)

                    if cfg.MEMORY <= 1:
                        samples_per_class = int(cfg.MEMORY * len(animal_list))
                    else:
                        samples_per_class = int(cfg.MEMORY / len(classes_till_now))

                    if cfg.SAMPLING.STRATERGY == "true_random":
                        np.random.seed(121)
                        np.random.shuffle(fname_list)
                        overall_images_list_old_classes += fname_list

                    if cfg.SAMPLING.STRATERGY == "random":
                        np.random.seed(121)
                        np.random.shuffle(fname_list)
                        images_list = fname_list[:samples_per_class]

                    elif cfg.SAMPLING.STRATERGY == "herding":
                        images_list = fname_list[:samples_per_class]

                    elif cfg.SAMPLING.STRATERGY == "dpp":
                        images_list = cluster_dpp(
                            keypoints_list,
                            keypoints_to_fname,
                            samples_per_class,
                            animal_class,
                        )

                    elif cfg.SAMPLING.STRATERGY == "rbf-dpp":
                        images_list = rbf_dpp(
                            keypoints_list,
                            keypoints_to_fname,
                            samples_per_class,
                            cfg.SAMPLING.GAMMA,
                        )

                    elif cfg.SAMPLING.STRATERGY == "cluster":
                        images_list = cluster(
                            keypoints_list, keypoints_to_fname, samples_per_class, cfg
                        )

            else:
                print("Computing images list")
                animal_list = []
                keypoints_list = []
                keypoints_to_fname = {}
                fname_list = []
                for fname in filename_list_classes[animal_class]:
                    temp = (fname, [])
                    for keypt in keypoint_names:
                        temp[1].append(
                            [
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_x"
                                ].item(),
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_y"
                                ].item(),
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_visible"
                                ].item(),
                            ]
                        )

                    animal_list.append(temp)

                    if (
                        cfg.SAMPLING.STRATERGY == "random"
                        or cfg.SAMPLING.STRATERGY == "feature_dpp"
                        or cfg.SAMPLING.STRATERGY == "dpp"
                        or cfg.SAMPLING.STRATERGY == "true_random"
                    ):
                        fname_list.append(temp[0])

                    if cfg.SAMPLING.STRATERGY == "herding":
                        keypoints_list.append(temp[1])

                    if (
                        cfg.SAMPLING.STRATERGY == "cluster"
                        or cfg.SAMPLING.STRATERGY == "dpp"
                        or cfg.SAMPLING.STRATERGY == "rbf-dpp"
                    ):
                        temp_2 = np.array(temp[1])
                        temp_2 = temp_2.flatten()
                        keypoints_list.append(temp_2)
                        keypoints_to_fname[str(temp_2)] = fname

                keypoints_list = np.array(keypoints_list)

                if cfg.MEMORY_TYPE == "fix":
                    if cfg.MEMORY <= 1:
                        samples_per_class = int(cfg.MEMORY * len(animal_list))
                    else:
                        samples_per_class = int(cfg.MEMORY / len(classes_till_now))

                if cfg.MEMORY_TYPE != "fix":
                    if cfg.TRAIN_INCREMENTAL.BASE_DATA_FOR_INCREMENTAL <= 1:
                        samples_per_class = int(
                            cfg.TRAIN_INCREMENTAL.BASE_DATA_FOR_INCREMENTAL
                            * len(animal_list)
                        )

                    else:
                        if cfg.SAMPLING.STRATERGY == "cluster":
                            samples_per_class = (
                                cfg.TRAIN_INCREMENTAL.BASE_DATA_FOR_INCREMENTAL
                                * cfg.SAMPLING.N_CLUSTERS
                            )
                        else:
                            samples_per_class = (
                                cfg.TRAIN_INCREMENTAL.BASE_DATA_FOR_INCREMENTAL
                            )

                if cfg.MEMORY_TYPE == "fix" and cfg.SAMPLING.STRATERGY == "dpp":
                    images_list = cluster_dpp(
                        keypoints_list,
                        keypoints_to_fname,
                        samples_per_class,
                        animal_class,
                    )

                if cfg.SAMPLING.STRATERGY == "rbf-dpp":
                    images_list = rbf_dpp(
                        keypoints_list,
                        keypoints_to_fname,
                        samples_per_class,
                        cfg.SAMPLING.GAMMA,
                    )

                if cfg.SAMPLING.STRATERGY == "true_random":
                    np.random.seed(121)
                    np.random.shuffle(fname_list)
                    overall_images_list_new_classes += fname_list[:samples_per_class]
                    print("Here, ", len(overall_images_list_new_classes))

                if cfg.SAMPLING.STRATERGY == "random":
                    np.random.seed(121)
                    np.random.shuffle(fname_list)
                    images_list = fname_list[:samples_per_class]

                if cfg.SAMPLING.STRATERGY == "cluster":
                    images_list = cluster(
                        keypoints_list, keypoints_to_fname, samples_per_class, cfg
                    )

                if cfg.SAMPLING.STRATERGY == "herding":
                    images_list = herding(
                        keypoints_list, animal_list, samples_per_class
                    )

                if cfg.MEMORY_TYPE != "fix" and cfg.SAMPLING.STRATERGY == "dpp":
                    Phi = np.matmul(keypoints_list, keypoints_list.T)
                    DPP = FiniteDPP("likelihood", **{"L": Phi})

                    k = 50
                    for _ in range(5):
                        DPP.sample_exact_k_dpp(size=k)

                    # print(DPP.list_of_samples)

                    max_det = 0
                    index_of_samples = DPP.list_of_samples[0]

                    for j in range(5):
                        matrix = np.array(Phi)
                        submatrix = matrix[
                            np.ix_(DPP.list_of_samples[j], DPP.list_of_samples[j])
                        ]
                        try:
                            det = np.linalg.det(submatrix)
                            if det > max_det:
                                max_det = det
                                index_of_samples = DPP.list_of_samples[j]
                        except:
                            continue

                    temp = keypoints_list[index_of_samples]

                    images_list = []
                    for j in temp:
                        images_list.append(keypoints_to_fname[str(j)])

                samples_per_class_old = samples_per_class

                memory_dict[animal_class] = images_list
                classes_visited.append(animal_class)

            train_tempset = AnimalDatasetCombined(
                cfg.DATASET.IMAGES,
                cfg.DATASET.ANNOT,
                images_list,
                input_size=(512, 512),
                output_size=(128, 128),
                transforms=torchvision.transforms.Compose([ToTensor()]),
                train=True,
            )

            print(len(train_tempset))
            curr_train_datasets.append(train_tempset)
            non_augmented_datasets.append(train_tempset)
            old_class_datasets.append(train_tempset)

            for aug in cfg.TRAIN_INCREMENTAL.AUGMENTATION:
                if aug == "part":
                    if animal_class == "cat":
                        continue
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.IMAGES,
                        cfg.DATASET.ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                        parts_augmentation=True,
                    )

                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_train_datasets.append(augmented_tempset)

                if aug == "rotation":
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.AUG_IMAGES,
                        cfg.DATASET.AUG_ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                    )
                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_train_datasets.append(augmented_tempset)
                    old_class_datasets.append(augmented_tempset)
                    augmented_datasets.append(augmented_tempset)

                if aug == "vanilla_rotation":
                    rotate_augmentation(
                        images_list,
                        annot_df,
                        src=cfg.DATASET.IMAGES,
                        dest=cfg.DATASET.AUG_IMAGES,
                        final_df_path=cfg.DATASET.AUG_ANNOT,
                    )
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.AUG_IMAGES,
                        cfg.DATASET.AUG_ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                    )
                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_train_datasets.append(augmented_tempset)
                    old_class_datasets.append(augmented_tempset)
                    augmented_datasets.append(augmented_tempset)

                if aug == "vanilla_flipping":
                    flipping_augmentation(
                        images_list,
                        annot_df,
                        src=cfg.DATASET.IMAGES,
                        dest=cfg.DATASET.AUG_IMAGES,
                        final_df_path=cfg.DATASET.AUG_ANNOT,
                    )
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.AUG_IMAGES,
                        cfg.DATASET.AUG_ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                    )
                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_train_datasets.append(augmented_tempset)
                    old_class_datasets.append(augmented_tempset)
                    augmented_datasets.append(augmented_tempset)

                if aug == "vanilla_noise":
                    noise_augmentation(
                        images_list,
                        annot_df,
                        src=cfg.DATASET.IMAGES,
                        dest=cfg.DATASET.AUG_IMAGES,
                        final_df_path=cfg.DATASET.AUG_ANNOT,
                    )
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.AUG_IMAGES,
                        cfg.DATASET.AUG_ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                    )
                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_train_datasets.append(augmented_tempset)
                    old_class_datasets.append(augmented_tempset)
                    augmented_datasets.append(augmented_tempset)

        if cfg.TRAIN_INCREMENTAL.KD_LOSS:
            kd_trainset = torch.utils.data.ConcatDataset(curr_train_datasets)
            kd_train_loader = torch.utils.data.DataLoader(
                kd_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True
            )
            curr_train_datasets = []

        if (
            cfg.TRAIN_INCREMENTAL.BALANCED_FINETUNING
            or cfg.TRAIN_INCREMENTAL.ICARL
            or cfg.TRAIN_INCREMENTAL.EEIL
        ):
            old_train_datasets = curr_train_datasets

        if cfg.SAMPLING.STRATERGY == "true_random":
            overall_images_list = overall_images_list_new_classes
            np.random.seed(121)
            np.random.shuffle(overall_images_list_old_classes)
            overall_images_list += overall_images_list_old_classes[
                : cfg.MEMORY - len(overall_images_list_new_classes)
            ]

            print("Memory size: ", len(overall_images_list))

            curr_train_datasets = [
                AnimalDatasetCombined(
                    cfg.DATASET.IMAGES,
                    cfg.DATASET.ANNOT,
                    overall_images_list,
                    input_size=(512, 512),
                    output_size=(128, 128),
                    transforms=torchvision.transforms.Compose([ToTensor()]),
                    train=True,
                )
            ]

        new_train_datasets = []
        for animal_class in animal_classes:
            classes_till_now.append(animal_class)
            temp_df = annot_df.loc[annot_df["class"] == animal_class]

            images_list = np.array(temp_df["filename"])
            np.random.shuffle(images_list)

            train_images_list = images_list[: int(0.9 * len(images_list))]
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

        if cfg.TRAIN_INCREMENTAL.EEIL:
            (
                curr_balanced_train_datasets,
                non_augmented_balanced_datasets,
                old_class_balanced_datasets,
                augmented_balanced_datasets,
            ) = ([], [], [], [])

            for animal_class in animal_classes:
                print("Computing images list for balanced finetuning")
                animal_list = []
                keypoints_list = []
                keypoints_to_fname = {}
                fname_list = []

                for fname in filename_list_classes[animal_class]:
                    temp = (fname, [])
                    for keypt in keypoint_names:
                        temp[1].append(
                            [
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_x"
                                ].item(),
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_y"
                                ].item(),
                                annot_df.loc[annot_df["filename"] == fname][
                                    keypt + "_visible"
                                ].item(),
                            ]
                        )

                    animal_list.append(temp)

                    if (
                        cfg.SAMPLING.STRATERGY == "random"
                        or cfg.SAMPLING.STRATERGY == "feature_dpp"
                        or cfg.SAMPLING.STRATERGY == "dpp"
                    ):
                        fname_list.append(temp[0])

                    if cfg.SAMPLING.STRATERGY == "herding":
                        keypoints_list.append(temp[1])

                    if (
                        cfg.SAMPLING.STRATERGY == "cluster"
                        or cfg.SAMPLING.STRATERGY == "dpp"
                        or cfg.SAMPLING.STRATERGY == "rbf-dpp"
                    ):
                        temp_2 = np.array(temp[1])
                        temp_2 = temp_2.flatten()
                        keypoints_list.append(temp_2)
                        keypoints_to_fname[str(temp_2)] = fname

                keypoints_list = np.array(keypoints_list)

                samples_per_class = samples_per_class_old

                if cfg.MEMORY_TYPE == "fix" and cfg.SAMPLING.STRATERGY == "dpp":
                    images_list = cluster_dpp(
                        keypoints_list,
                        keypoints_to_fname,
                        samples_per_class,
                        animal_class,
                    )

                if cfg.SAMPLING.STRATERGY == "rbf-dpp":
                    images_list = rbf_dpp(
                        keypoints_list,
                        keypoints_to_fname,
                        samples_per_class,
                        cfg.SAMPLING.GAMMA,
                    )

                if cfg.SAMPLING.STRATERGY == "random":
                    np.random.seed(121)
                    np.random.shuffle(fname_list)
                    images_list = fname_list[:samples_per_class]

                if cfg.SAMPLING.STRATERGY == "cluster":
                    images_list = cluster(
                        keypoints_list, keypoints_to_fname, samples_per_class, cfg
                    )

                if cfg.SAMPLING.STRATERGY == "herding":
                    images_list = herding(
                        keypoints_list, animal_list, samples_per_class
                    )

                if cfg.MEMORY_TYPE != "fix" and cfg.SAMPLING.STRATERGY == "dpp":
                    Phi = np.matmul(keypoints_list, keypoints_list.T)
                    DPP = FiniteDPP("likelihood", **{"L": Phi})

                    k = 50
                    for _ in range(5):
                        DPP.sample_exact_k_dpp(size=k)

                    # print(DPP.list_of_samples)

                    max_det = 0
                    index_of_samples = DPP.list_of_samples[0]

                    for j in range(5):
                        matrix = np.array(Phi)
                        submatrix = matrix[
                            np.ix_(DPP.list_of_samples[j], DPP.list_of_samples[j])
                        ]
                        try:
                            det = np.linalg.det(submatrix)
                            if det > max_det:
                                max_det = det
                                index_of_samples = DPP.list_of_samples[j]
                        except:
                            continue

                    temp = keypoints_list[index_of_samples]

                    images_list = []
                    for j in temp:
                        images_list.append(keypoints_to_fname[str(j)])

                memory_dict[animal_class] = images_list
                classes_visited.append(animal_class)

                train_tempset = AnimalDatasetCombined(
                    cfg.DATASET.IMAGES,
                    cfg.DATASET.ANNOT,
                    images_list,
                    input_size=(512, 512),
                    output_size=(128, 128),
                    transforms=torchvision.transforms.Compose([ToTensor()]),
                    train=True,
                )

                print(len(train_tempset))
                curr_balanced_train_datasets.append(train_tempset)
                non_augmented_balanced_datasets.append(train_tempset)
                old_class_balanced_datasets.append(train_tempset)

                if cfg.TRAIN_INCREMENTAL.AUGMENTATION == "part":
                    if animal_class == "cat":
                        continue
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.IMAGES,
                        cfg.DATASET.ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                        parts_augmentation=True,
                    )

                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_balanced_train_datasets.append(augmented_tempset)

                if cfg.TRAIN_INCREMENTAL.AUGMENTATION == "rotation":
                    augmented_tempset = AnimalDatasetCombined(
                        cfg.DATASET.AUG_IMAGES,
                        cfg.DATASET.AUG_ANNOT,
                        images_list,
                        input_size=(512, 512),
                        output_size=(128, 128),
                        transforms=torchvision.transforms.Compose([ToTensor()]),
                        train=True,
                    )
                    print("Length of Augmented Set: ", len(augmented_tempset))
                    curr_balanced_train_datasets.append(augmented_tempset)
                    old_class_balanced_datasets.append(augmented_tempset)
                    augmented_balanced_datasets.append(augmented_tempset)

            # for i in curr_train_datasets:
            #     curr_balanced_train_datasets.append(i)

            balanced_incremental_trainset = torch.utils.data.ConcatDataset(
                curr_balanced_train_datasets
            )
            balanced_incremental_train_loader = torch.utils.data.DataLoader(
                balanced_incremental_trainset,
                batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE,
                shuffle=True,
            )

        if (
            cfg.TRAIN_INCREMENTAL.BALANCED_FINETUNING
            or cfg.TRAIN_INCREMENTAL.ICARL
            or cfg.TRAIN_INCREMENTAL.EEIL
        ):
            old_trainset = torch.utils.data.ConcatDataset(old_train_datasets)
            old_train_loader = torch.utils.data.DataLoader(
                old_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True
            )

            new_trainset = torch.utils.data.ConcatDataset(new_train_datasets)
            new_train_loader = torch.utils.data.DataLoader(
                new_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True
            )

        if (
            cfg.TRAIN_INCREMENTAL.FINETUNING_NON_AUGMENTED
            or cfg.TRAIN_INCREMENTAL.FINETUNING_AUGMENTED
        ):
            non_augmented_trainset = torch.utils.data.ConcatDataset(
                non_augmented_datasets
            )
            non_augmented_trainloader = torch.utils.data.DataLoader(
                non_augmented_trainset,
                batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE,
                shuffle=True,
            )

            augmented_trainset = torch.utils.data.ConcatDataset(augmented_datasets)
            augmented_train_loader = torch.utils.data.DataLoader(
                augmented_trainset,
                batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE,
                shuffle=True,
            )

        if cfg.TRAIN_INCREMENTAL.FINETUNING_OLD_DATA:
            old_trainset = torch.utils.data.ConcatDataset(old_class_datasets)
            old_train_loader = torch.utils.data.DataLoader(
                old_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True,
            )

            new_trainset = torch.utils.data.ConcatDataset(new_train_datasets)
            new_train_loader = torch.utils.data.DataLoader(
                new_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True,
            )

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

            # Training according to various scenarios
            if cfg.TRAIN_INCREMENTAL.KD_LOSS:
                train_loss, train_acc = train(
                    opt,
                    incremental_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Incremental_Train" + str(inc_step),
                )

                if cfg.TRAIN_INCREMENTAL.AUGMENTATION == "mixup":
                    train_loss_kd, train_acc_kd = train_kd_mixup(
                        opt,
                        kd_train_loader,
                        m,
                        m_prev,
                        criterion,
                        optimizer,
                        writer,
                        phase="Incremental_Train" + str(inc_step),
                    )

                else:
                    train_loss_kd, train_acc_kd = train_kd(
                        opt,
                        kd_train_loader,
                        m,
                        m_prev,
                        criterion,
                        optimizer,
                        writer,
                        phase="Incremental_Train" + str(inc_step),
                    )

                logger.epochInfo(
                    "Incremental_Train" + str(inc_step) + "_KD",
                    opt.epoch,
                    train_loss_kd,
                    train_acc_kd,
                )

            elif cfg.TRAIN_INCREMENTAL.BALANCED_FINETUNING:
                train_loss, train_acc = train(
                    opt,
                    incremental_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Incremental_Train" + str(inc_step),
                )
                logger.epochInfo(
                    "Normal Incremental_Train" + str(inc_step),
                    opt.epoch,
                    train_loss,
                    train_acc,
                )
                train_loss, train_acc = train_balanced(
                    opt,
                    old_train_loader,
                    new_train_loader,
                    m,
                    m_prev,
                    criterion,
                    optimizer,
                    writer,
                    phase="Balanced_Finetuing" + str(inc_step),
                )

            elif cfg.TRAIN_INCREMENTAL.FINETUNING_NON_AUGMENTED:
                train_loss, train_acc = train(
                    opt,
                    augmented_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Augmented_Incremental_Train" + str(inc_step),
                )
                logger.epochInfo(
                    "Augmented_Incremental_Train" + str(inc_step),
                    opt.epoch,
                    train_loss,
                    train_acc,
                )

                train_loss, train_acc = train(
                    opt,
                    non_augmented_trainloader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Non_Augmented_Incremental_Train" + str(inc_step),
                )

            elif cfg.TRAIN_INCREMENTAL.FINETUNING_OLD_DATA:
                train_loss, train_acc = train(
                    opt,
                    new_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="New_Incremental_Train" + str(inc_step),
                )
                logger.epochInfo(
                    "New_Incremental_Train" + str(inc_step),
                    opt.epoch,
                    train_loss,
                    train_acc,
                )
                train_loss, train_acc = train(
                    opt,
                    old_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Finetuning" + str(inc_step),
                )

            elif cfg.TRAIN_INCREMENTAL.FINETUNING_AUGMENTED:
                train_loss, train_acc = train(
                    opt,
                    non_augmented_trainloader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Non_Augmented_Incremental_Train" + str(inc_step),
                )
                logger.epochInfo(
                    "Non_Augmented_Incremental_Train" + str(inc_step),
                    opt.epoch,
                    train_loss,
                    train_acc,
                )

                train_loss, train_acc = train(
                    opt,
                    augmented_train_loader,
                    m,
                    criterion,
                    optimizer,
                    writer,
                    phase="Augmented_Incremental_Train" + str(inc_step),
                )

            elif cfg.TRAIN_INCREMENTAL.ICARL or cfg.TRAIN_INCREMENTAL.EEIL:
                train_loss, train_acc = train_icarl(
                    opt,
                    new_train_loader,
                    old_train_loader,
                    m,
                    m_prev,
                    criterion,
                    optimizer,
                    writer,
                    phase="Incremental_Train" + str(inc_step),
                )

            else:
                train_loss, train_acc = train(
                    opt,
                    incremental_train_loader,
                    m,
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

        #######################################
        # Special Case for EEIL type training #
        #######################################

        if cfg.TRAIN_INCREMENTAL.EEIL:
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

            opt.trainIters = 0
            opt.val_iters = 0

            best_acc = 0.0
            best_model_weights = deepcopy(m.state_dict())

            logger.info(
                f"#######################################################################################################################"
            )
            logger.info(
                f"############################ Starting Balanced Finetuning Training step {inc_step}  ##########################################"
            )

            for ep in range(0, 5):
                opt.epoch = ep
                current_lr = (
                    optimizer.state_dict()["param_groups"][0]["lr"] * 0.1
                )  # Reduce the Learning rate by 0.1 for the balanced finetuning stage

                logger.info(
                    f"############# Starting Epoch {opt.epoch} | LR: {current_lr} #############"
                )

                train_loss, train_acc = train_icarl(
                    opt,
                    old_train_loader,
                    balanced_incremental_train_loader,
                    m,
                    m_prev,
                    criterion,
                    optimizer,
                    writer,
                    phase="Balanced_Finetuning" + str(inc_step),
                )

                logger.epochInfo(
                    "Balanced_Finetuning" + str(inc_step),
                    opt.epoch,
                    train_loss,
                    train_acc,
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
                            "Balanced_val" + str(inc_step),
                        )

                    opt.val_iters += 1

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_weights = deepcopy(m.state_dict())

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
