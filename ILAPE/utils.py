import os
import cv2
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import torch
import torchvision

from animal_data_loader import AnimalDatasetCombined, ToTensor


import thinplate as tps


def L2_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.abs(np.sum((v2 - v1) ** 2)))


def calc_dist(avg_vector, animal_list):
    result = []
    for i in animal_list:
        fname = i[0]
        keypoints = i[1]
        dist = L2_dist(avg_vector, keypoints)
        temp = ((fname, keypoints), dist)
        result.append(temp)
    return result


def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


def save_images(
    fname_list,
    images_path="../data/cropped_images/",
    annot_path="../data/updated_df.csv",
    save_dir="./keypoints/",
    animal_class=None,
    train=True,
):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    trainset = AnimalDatasetCombined(
        images_path,
        annot_path,
        fname_list,
        input_size=(512, 512),
        output_size=(128, 128),
        transforms=torchvision.transforms.Compose([ToTensor()]),
        train=train,
    )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    for i, (inps, labels, label_masks, _) in enumerate(train_loader):
        for j in range(labels.shape[0]):
            final_img = np.zeros((128, 128))
            input_img = inps[j].cpu().numpy().transpose((1, 2, 0))

            input_img = Image.fromarray(input_img.astype(np.uint8))
            input_img.save(
                os.path.join(
                    save_dir, "image" + str(animal_class) + "_{}_{}.png".format(i, j)
                )
            )

            for k in range(labels.shape[1]):
                images = labels.cpu().numpy()[j][k]
                final_img += images
                # plt.imsave(
                #     os.path.join(
                #         save_dir,
                #         "temp" + str(animal_class) + "_{}_{}_{}.png".format(i, j, k),
                #     ),
                #     images,
                #     cmap="gray",
                # )
            plt.imsave(
                os.path.join(
                    save_dir, "temp" + str(animal_class) + "_{}_{}.png".format(i, j)
                ),
                final_img,
                cmap="gray",
            )


def get_keypoints(fname, csv_file="../data/updated_df.csv"):
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

    keypoints = []
    vis = []

    annot_df = pd.read_csv(csv_file)

    temp = annot_df.loc[annot_df["filename"] == fname]
    for keypt in keypoint_names:
        keypoints.append((temp[keypt + "_x"].item(), temp[keypt + "_y"].item()))
        vis.append(temp[keypt + "_visible"].item())

    keypoints_dict = {}
    keypoints_dict["L_Eye"] = keypoints[0]
    keypoints_dict["R_Eye"] = keypoints[1]
    keypoints_dict["Nose"] = keypoints[2]
    keypoints_dict["L_Ear"] = keypoints[3]
    keypoints_dict["R_Ear"] = keypoints[4]
    keypoints_dict["LF_Elbow"] = keypoints[5]
    keypoints_dict["LF_Paw"] = keypoints[6]
    keypoints_dict["RF_Paw"] = keypoints[7]
    keypoints_dict["RF_Elbow"] = keypoints[8]
    keypoints_dict["LB_Paw"] = keypoints[9]
    keypoints_dict["RB_Paw"] = keypoints[10]
    keypoints_dict["LB_Elbow"] = keypoints[11]
    keypoints_dict["RB_Elbow"] = keypoints[12]
    keypoints_dict["LF_Knee"] = keypoints[13]
    keypoints_dict["RF_Knee"] = keypoints[14]
    keypoints_dict["LB_Knee"] = keypoints[15]
    keypoints_dict["RB_Knee"] = keypoints[16]

    return keypoints, keypoints_dict, vis


def get_xs_ys(annot_list):
    x, y = [], []
    for i in range(len(annot_list)):
        if i % 2 == 0:
            x.append(annot_list[i])
        else:
            y.append(annot_list[i])

    x.append(x[0])
    y.append(y[0])
    return x, y


def rotate_about_pt(x, y, origin_x, origin_y, angle):
    x_ = x - origin_x
    y_ = y - origin_y
    c = np.cos(angle)
    s = np.sin(angle)
    t_x = x_ * c - y_ * s
    t_y = x_ * s + y_ * c
    x = t_x + origin_x
    y = t_y + origin_y
    return x, y


def f(index):
    return index


def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)


def rotate_elbow_knee_limbs(
    input_dir="../data/cropped_images/",
    output_dir="../data/rotated_images/",
    input_csv_file="../data/updated_df.csv",
    output_csv_file="../data/updated_df_rotated.csv",
    animal_class=None,
):
    if not os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    indices_to_be_rotated = [
        (5, 13, 6),
        # (5, 6),
        (8, 14, 7),
        # (8, 7),
        (11, 15, 9),
        # (11, 9),
        (12, 16, 10),
        # (12, 10),
    ]
    indices_not_to_be_rotated = [0, 1, 2, 3, 4]

    keypoints_dict = {}
    for i in range(17):
        keypoints_dict[i] = []

    df = pd.read_csv(input_csv_file)

    if animal_class:
        df = df.loc[df["class"] == animal_class]

    classes = list(df["class"])
    fname_list = list(df["filename"])

    for fname in fname_list:
        keypoints, _, vis = get_keypoints(fname, csv_file=input_csv_file)

        c_src = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

        c_dst = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

        indices_visited = []

        for ind in indices_to_be_rotated:
            if vis[ind[0]] == 1 and vis[ind[1]] == 1 and vis[ind[2]] == 1:
                angle = random.uniform(0.25, 0.6)
                rot_pt_x, rot_pt_y = rotate_about_pt(
                    keypoints[ind[1]][0],
                    keypoints[ind[1]][1],
                    keypoints[ind[0]][0],
                    keypoints[ind[0]][1],
                    angle,
                )

                rot_pt_x_paw, rot_pt_y_paw = rotate_about_pt(
                    keypoints[ind[2]][0],
                    keypoints[ind[2]][1],
                    keypoints[ind[0]][0],
                    keypoints[ind[0]][1],
                    angle,
                )

                if rot_pt_x < 0 or rot_pt_y < 0 or rot_pt_x_paw < 0 or rot_pt_y_paw < 0:
                    rot_pt_x, rot_pt_y = keypoints[ind[1]][0], keypoints[ind[1]][1]
                    rot_pt_x_paw, rot_pt_y_paw = (
                        keypoints[ind[2]][0],
                        keypoints[ind[2]][1],
                    )

                    # move on
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], 1]
                    )
                    indices_visited.append(ind[0])

                    keypoints_dict[ind[1]].append([rot_pt_x, rot_pt_y, 1])
                    indices_visited.append(ind[1])

                    keypoints_dict[ind[2]].append([rot_pt_x_paw, rot_pt_y_paw, 1])
                    indices_visited.append(ind[2])
                    continue

                ## Original Keypoints
                c_src.append(
                    [keypoints[ind[0]][0] / 512.0, keypoints[ind[0]][1] / 512.0]
                )

                c_src.append(
                    [keypoints[ind[1]][0] / 512.0, keypoints[ind[1]][1] / 512.0]
                )

                c_src.append(
                    [keypoints[ind[2]][0] / 512.0, keypoints[ind[2]][1] / 512.0]
                )

                ## Keypoints after rotating them by a random angle
                c_dst.append(
                    [keypoints[ind[0]][0] / 512.0, keypoints[ind[0]][1] / 512.0]
                )
                c_dst.append([rot_pt_x / 512.0, rot_pt_y / 512.0])
                c_dst.append([rot_pt_x_paw / 512.0, rot_pt_y_paw / 512.0])

                if not ind[0] in indices_visited:
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], 1]
                    )
                    indices_visited.append(ind[0])

                if not ind[1] in indices_visited:
                    keypoints_dict[ind[1]].append([rot_pt_x, rot_pt_y, 1])
                    indices_visited.append(ind[1])

                if not ind[2] in indices_visited:
                    keypoints_dict[ind[2]].append([rot_pt_x_paw, rot_pt_y_paw, 1])
                    indices_visited.append(ind[2])

            else:
                if not ind[0] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[0]].append(
                        [keypoints[ind[0]][0], keypoints[ind[0]][1], vis[ind[0]]]
                    )
                    indices_visited.append(ind[0])

                if not ind[1] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[1]].append(
                        [keypoints[ind[1]][0], keypoints[ind[1]][1], vis[ind[1]]]
                    )
                    indices_visited.append(ind[1])

                if not ind[2] in indices_visited:
                    # keypoints_dict[ind[0]].append([0, 0, 0])
                    keypoints_dict[ind[2]].append(
                        [keypoints[ind[2]][0], keypoints[ind[2]][1], vis[ind[2]]]
                    )
                    indices_visited.append(ind[2])

        for ind in indices_not_to_be_rotated:
            if vis[ind] == 1:
                c_src.append([keypoints[ind][0] / 512.0, keypoints[ind][1] / 512.0])
                c_dst.append([keypoints[ind][0] / 512.0, keypoints[ind][1] / 512.0])
                keypoints_dict[ind].append([keypoints[ind][0], keypoints[ind][1], 1])

            else:
                keypoints_dict[ind].append([0, 0, 0])

        c_src = np.array(c_src)
        c_dst = np.array(c_dst)

        img = cv2.imread(os.path.join(input_dir, fname[:-4] + ".jpg"))

        warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))

        img2gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        dst = cv2.inpaint(warped, mask_inv, 3, cv2.INPAINT_TELEA)

        cv2.imwrite(os.path.join(output_dir, fname[:-4] + ".jpg"), dst)

    indices_to_keypoints_dict = {
        0: "L_Eye",
        1: "R_Eye",
        2: "Nose",
        3: "L_EarBase",
        4: "R_EarBase",
        5: "L_F_Elbow",
        6: "L_F_Paw",
        7: "R_F_Paw",
        8: "R_F_Elbow",
        9: "L_B_Paw",
        10: "R_B_Paw",
        11: "L_B_Elbow",
        12: "R_B_Elbow",
        13: "L_F_Knee",
        14: "R_F_Knee",
        15: "L_B_Knee",
        16: "R_B_Knee",
    }

    data_dict = {}

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

    data_dict["filename"] = fname_list
    data_dict["class"] = classes

    for keypt in keypoint_names:
        data_dict[keypt + "_x"] = []
        data_dict[keypt + "_y"] = []
        data_dict[keypt + "_visible"] = []

    for i in keypoints_dict.keys():
        keypt = indices_to_keypoints_dict[i]
        for d in keypoints_dict[i]:
            data_dict[keypt + "_x"].append(d[0])
            data_dict[keypt + "_y"].append(d[1])
            data_dict[keypt + "_visible"].append(d[2])

    for k in data_dict.keys():
        print(f"Length of {k} keypoints list is : {len(data_dict[k])}")

    final_df = pd.DataFrame(data_dict)
    final_df.to_csv(output_csv_file, index=None)


def rotate_augmentation(
    images_list,
    annot_df,
    src="../data/cropped_images/",
    dest="../data/aug_cropped_images/",
    final_df_path="../data/aug_cropped_images.csv",
):
    if not os.path.exists(dest):
        os.mkdir(dest)

    updated_data = {}

    updated_data["filename"] = []
    updated_data["class"] = []

    keypoint_names = [
        "L_Eye",
        "R_Eye",
        "Nose",
        "L_EarBase",
        "R_EarBase",
        "Throat",
        "Withers",
        "TailBase",
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

    for i in keypoint_names:
        updated_data[i + "_x"] = []
        updated_data[i + "_y"] = []
        updated_data[i + "_visible"] = []

    for fname in images_list:
        image = imageio.imread(os.path.join(src, fname[:-4] + ".jpg"))

        data = annot_df.loc[annot_df["filename"] == fname]

        updated_data["filename"].append(fname)
        updated_data["class"].append(data["class"].item())

        kps_list = []
        for keypt in keypoint_names:
            # if data[keypt + "_visible"] == 0:
            kps_list.append(
                Keypoint(x=data[keypt + "_x"].item(), y=data[keypt + "_y"].item())
            )

        kps = KeypointsOnImage(kps_list, shape=image.shape)
        rot = np.random.rand(1) * 10
        seq = iaa.Sequential(
            [
                iaa.Affine(
                    rotate=10, scale=(0.8, 0.99)
                ),  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
            ]
        )

        image_aug, kps_aug = seq(image=image, keypoints=kps)

        idx = 0
        for keypt in keypoint_names:
            if data[keypt + "_visible"].item() == 0:
                updated_data[keypt + "_x"].append(0.0)
                updated_data[keypt + "_y"].append(0.0)
            else:
                updated_data[keypt + "_x"].append(float(int(kps_aug.keypoints[idx].x)))
                updated_data[keypt + "_y"].append(float(int(kps_aug.keypoints[idx].y)))
            updated_data[keypt + "_visible"].append(data[keypt + "_visible"].item())
            idx += 1

        imageio.imwrite(os.path.join(dest, fname[:-4] + ".jpg"), image_aug)

    df = pd.DataFrame(updated_data)
    df.to_csv(final_df_path, index=False)


def flipping_augmentation(
    images_list,
    annot_df,
    src="../data/cropped_images/",
    dest="../data/aug_cropped_images/",
    final_df_path="../data/aug_cropped_images.csv",
):
    if not os.path.exists(dest):
        os.mkdir(dest)

    updated_data = {}

    updated_data["filename"] = []
    updated_data["class"] = []

    keypoint_names = [
        "L_Eye",
        "R_Eye",
        "Nose",
        "L_EarBase",
        "R_EarBase",
        "Throat",
        "Withers",
        "TailBase",
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

    for i in keypoint_names:
        updated_data[i + "_x"] = []
        updated_data[i + "_y"] = []
        updated_data[i + "_visible"] = []

    for fname in images_list:
        image = imageio.imread(os.path.join(src, fname[:-4] + ".jpg"))

        data = annot_df.loc[annot_df["filename"] == fname]

        updated_data["filename"].append(fname)
        updated_data["class"].append(data["class"].item())

        kps_list = []
        for keypt in keypoint_names:
            # if data[keypt + "_visible"] == 0:
            kps_list.append(
                Keypoint(x=data[keypt + "_x"].item(), y=data[keypt + "_y"].item())
            )

        kps = KeypointsOnImage(kps_list, shape=image.shape)
        rot = np.random.rand(1) * 10
        seq = iaa.Sequential([iaa.Fliplr(1.0),])

        image_aug, kps_aug = seq(image=image, keypoints=kps)

        idx = 0
        for keypt in keypoint_names:
            if data[keypt + "_visible"].item() == 0:
                updated_data[keypt + "_x"].append(0.0)
                updated_data[keypt + "_y"].append(0.0)
            else:
                updated_data[keypt + "_x"].append(float(int(kps_aug.keypoints[idx].x)))
                updated_data[keypt + "_y"].append(float(int(kps_aug.keypoints[idx].y)))
            updated_data[keypt + "_visible"].append(data[keypt + "_visible"].item())
            idx += 1

        imageio.imwrite(os.path.join(dest, fname[:-4] + ".jpg"), image_aug)

    df = pd.DataFrame(updated_data)
    df.to_csv(final_df_path, index=False)


def noise_augmentation(
    images_list,
    annot_df,
    src="../data/cropped_images/",
    dest="../data/aug_cropped_images/",
    final_df_path="../data/aug_cropped_images.csv",
):
    if not os.path.exists(dest):
        os.mkdir(dest)

    updated_data = {}

    updated_data["filename"] = []
    updated_data["class"] = []

    keypoint_names = [
        "L_Eye",
        "R_Eye",
        "Nose",
        "L_EarBase",
        "R_EarBase",
        "Throat",
        "Withers",
        "TailBase",
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

    for i in keypoint_names:
        updated_data[i + "_x"] = []
        updated_data[i + "_y"] = []
        updated_data[i + "_visible"] = []

    for fname in images_list:
        image = imageio.imread(os.path.join(src, fname[:-4] + ".jpg"))

        data = annot_df.loc[annot_df["filename"] == fname]

        updated_data["filename"].append(fname)
        updated_data["class"].append(data["class"].item())

        kps_list = []
        for keypt in keypoint_names:
            # if data[keypt + "_visible"] == 0:
            kps_list.append(
                Keypoint(x=data[keypt + "_x"].item(), y=data[keypt + "_y"].item())
            )

        kps = KeypointsOnImage(kps_list, shape=image.shape)
        rot = np.random.rand(1) * 10
        seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),])

        image_aug, kps_aug = seq(image=image, keypoints=kps)

        idx = 0
        for keypt in keypoint_names:
            if data[keypt + "_visible"].item() == 0:
                updated_data[keypt + "_x"].append(0.0)
                updated_data[keypt + "_y"].append(0.0)
            else:
                updated_data[keypt + "_x"].append(float(int(kps_aug.keypoints[idx].x)))
                updated_data[keypt + "_y"].append(float(int(kps_aug.keypoints[idx].y)))
            updated_data[keypt + "_visible"].append(data[keypt + "_visible"].item())
            idx += 1

        imageio.imwrite(os.path.join(dest, fname[:-4] + ".jpg"), image_aug)

    df = pd.DataFrame(updated_data)
    df.to_csv(final_df_path, index=False)
