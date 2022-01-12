import os
import cv2
import glob
import shutil
import scipy.misc
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xml.etree.ElementTree as ET

from ray.util.multiprocessing import Pool


def f(index):
    return index


def get_transform(center, scale, res, pad_scale=2, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, pad_scale=2, rot=0):
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return (
                torch.zeros(res[0], res[1], img.shape[2])
                if len(img.shape) > 2
                else torch.zeros(res[0], res[1])
            )
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / pad_scale - float(br[1] - ul[1]) / pad_scale)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    # print(new_img.shape)

    # new_img = np.resize(new_img, (res[0],res[1],3) )
    new_img = scipy.misc.imresize(new_img, res)
    return new_img


def move_files(PATH="./data/PASCAL_VOC_2012/", dest="./data/images"):
    classes = os.listdir(PATH)
    for c in classes:
        src = PATH + c + "/"
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)


def xml_to_df(PATH="./data/annot"):
    files = glob.glob(PATH + "/**/*.xml", recursive=True)
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
    data = {}
    data["filename"] = []
    data["class"] = []
    data["xmin_bbox"] = []
    data["ymin_bbox"] = []
    data["width_bbox"] = []
    data["height_bbox"] = []
    for i in keypoint_names:
        data[i + "_x"] = []
        data[i + "_y"] = []
        data[i + "_visible"] = []
    for f in files:
        root = ET.parse(f).getroot()

        ##### MUST CHECK THIS based on input directories ######
        data["filename"].append(f[42:])

        for i in root.findall("category"):
            cls = i.text

        for i in root.findall("visible_bounds"):
            xmin = float(i.get("xmin"))
            ymin = float(i.get("ymin"))
            width = float(i.get("width"))
            height = float(i.get("height"))

        data["class"].append(cls)
        data["xmin_bbox"].append(xmin)
        data["ymin_bbox"].append(ymin)
        data["width_bbox"].append(width)
        data["height_bbox"].append(height)
        for i in root.findall("keypoints/keypoint"):
            x = float(i.get("x"))
            y = float(i.get("y"))
            visible = float(i.get("visible"))
            name = i.get("name")
            data[name + "_x"].append(x)
            data[name + "_y"].append(y)
            data[name + "_visible"].append(visible)
    df = pd.DataFrame.from_dict(data)
    return df


def generate_cropped_images_and_keypoints_df(
    annot_df, src="./data/images/", dest="./data/cropped_images/", final_img_size=512
):
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

    for i in range(len(annot_df)):
        updated_data["filename"].append(annot_df["filename"][i])
        updated_data["class"].append(annot_df["class"][i])

        xmin_bbox = int(annot_df["xmin_bbox"][i])
        ymin_bbox = int(annot_df["ymin_bbox"][i])
        height_bbox = int(annot_df["height_bbox"][i])
        width_bbox = int(annot_df["width_bbox"][i])

        try:
            img = cv2.imread(src + annot_df["filename"][i][:11] + ".jpg")

            ht, wd, cc = img.shape

            # cv2.imwrite("temp.png", img)
            # create new image of desired size for padding.
            ww = 1000
            hh = 1000
            color = (0, 0, 0)
            result = np.full((hh, ww, cc), color, dtype=np.uint8)

            # compute center offset
            xx = (ww - wd) // 2
            yy = (hh - ht) // 2

            # copy img image into center of result image
            result[yy : yy + ht, xx : xx + wd] = img

            finalXs = []
            finalYs = []
            finalVis = []
            if width_bbox > height_bbox:
                dir1 = (width_bbox - height_bbox) // 2

                cropped = result[
                    ymin_bbox + yy - dir1 : ymin_bbox + height_bbox + yy + dir1,
                    xmin_bbox + xx : xmin_bbox + width_bbox + xx,
                ]
                cropped = cv2.resize(cropped, (512, 512))
                for keypt in keypoint_names:
                    newX = int(
                        (512 / width_bbox)
                        * (int(annot_df[keypt + "_x"][i]) - xmin_bbox)
                    )
                    newY = int(
                        (512 / width_bbox)
                        * (int(annot_df[keypt + "_y"][i]) - ymin_bbox + dir1)
                    )

                    if (
                        int(annot_df[keypt + "_x"][i]) == 0
                        and int(annot_df[keypt + "_y"][i]) == 0
                    ):
                        finalXs.append(0)
                        finalYs.append(0)
                        finalVis.append(int(annot_df[keypt + "_visible"][i]))
                        continue
                    finalXs.append(max(0, newX))
                    finalYs.append(max(0, newY))
                    finalVis.append(int(annot_df[keypt + "_visible"][i]))

            else:
                dir1 = (height_bbox - width_bbox) // 2

                cropped = result[
                    ymin_bbox + yy : ymin_bbox + height_bbox + yy,
                    xmin_bbox + xx - dir1 : xmin_bbox + width_bbox + xx + dir1,
                ]
                cropped = cv2.resize(cropped, (512, 512))
                for keypt in keypoint_names:
                    newX = int(
                        (512 / height_bbox)
                        * (int(annot_df[keypt + "_x"][i]) - xmin_bbox + dir1)
                    )
                    newY = int(
                        (512 / height_bbox)
                        * (int(annot_df[keypt + "_y"][i]) - ymin_bbox)
                    )
                    if (
                        int(annot_df[keypt + "_x"][i]) == 0
                        and int(annot_df[keypt + "_y"][i]) == 0
                    ):
                        finalXs.append(0)
                        finalYs.append(0)
                        finalVis.append(int(annot_df[keypt + "_visible"][i]))
                        continue

                    finalXs.append(max(0, newX))
                    finalYs.append(max(0, newY))
                    finalVis.append(int(annot_df[keypt + "_visible"][i]))
            # view result
            # cv2_imshow(cropped)

            cv2.imwrite((dest + annot_df["filename"][i][:-4] + ".jpg"), cropped)

            cnt = 0
            for keypt in keypoint_names:
                updated_data[keypt + "_x"].append(finalXs[cnt])
                updated_data[keypt + "_y"].append(finalYs[cnt])
                updated_data[keypt + "_visible"].append(finalVis[cnt])
                cnt += 1

            # flip image

            flipped_img = cv2.flip(cropped, 1)
            updated_data["filename"].append(annot_df["filename"][i][:-4] + "_f.xml")
            updated_data["class"].append(annot_df["class"][i])

            flipped_Xs = []
            flipped_Ys = []
            flipped_vis = []

            for j in range(len(finalXs)):
                flipped_Xs.append(final_img_size - finalXs[j])
                flipped_Ys.append(finalYs[j])
                flipped_vis.append(finalVis[j])

            cnt = 0
            for keypt in keypoint_names:
                updated_data[keypt + "_x"].append(flipped_Xs[cnt])
                updated_data[keypt + "_y"].append(flipped_Ys[cnt])
                updated_data[keypt + "_visible"].append(flipped_vis[cnt])
                cnt += 1

            cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_f.jpg"), flipped_img)

            # add noise

            noise = np.random.randint(0, 10, (final_img_size, final_img_size, 3))
            noisy_img = noise + cropped

            updated_data["filename"].append(annot_df["filename"][i][:-4] + "_n.xml")
            updated_data["class"].append(annot_df["class"][i])

            cnt = 0
            for keypt in keypoint_names:
                updated_data[keypt + "_x"].append(finalXs[cnt])
                updated_data[keypt + "_y"].append(finalYs[cnt])
                updated_data[keypt + "_visible"].append(finalVis[cnt])
                cnt += 1

            cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_n.jpg"), noisy_img)

            # flipped_noisy_img

            flipped_noisy_img = cv2.flip(noisy_img, 1)
            updated_data["filename"].append(annot_df["filename"][i][:-4] + "_fn.xml")
            updated_data["class"].append(annot_df["class"][i])

            flipped_Xs = []
            flipped_Ys = []
            flipped_vis = []

            for j in range(len(finalXs)):
                flipped_Xs.append(final_img_size - finalXs[j])
                flipped_Ys.append(finalYs[j])
                flipped_vis.append(finalVis[j])

            cnt = 0
            for keypt in keypoint_names:
                updated_data[keypt + "_x"].append(flipped_Xs[cnt])
                updated_data[keypt + "_y"].append(flipped_Ys[cnt])
                updated_data[keypt + "_visible"].append(flipped_vis[cnt])
                cnt += 1

            cv2.imwrite(
                (dest + annot_df["filename"][i][:-4] + "_fn.jpg"), flipped_noisy_img
            )

            # rotate
            updated_data["filename"].append(annot_df["filename"][i][:-4] + "_r.xml")
            updated_data["class"].append(annot_df["class"][i])

            c = (final_img_size / 2, final_img_size / 2)

            s = final_img_size / 200.0 * 1.5
            r = 0
            sf = 0.2
            rf = 2
            s = s * (np.random.randn(1).dot(sf) + 1).clip(1 - sf, 1 + sf)[0]
            r = np.random.randn(1).dot(rf).clip(-2 * rf, 2 * rf)[0]

            rot_img = crop(cropped, c, s, [final_img_size, final_img_size], rot=r)

            cnt = 0
            for keypt in keypoint_names:
                if finalVis[cnt] == 0:
                    temp = [0.0, 0.0, 0.0]
                    updated_data[keypt + "_visible"].append(0)
                else:
                    temp = transform(
                        [finalXs[cnt], finalYs[cnt]],
                        c,
                        s,
                        [final_img_size, final_img_size],
                        rot=r,
                    )
                    updated_data[keypt + "_visible"].append(1)

                cnt += 1
                updated_data[keypt + "_x"].append(temp[0])
                updated_data[keypt + "_y"].append(temp[1])

            cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_r.jpg"), rot_img)

        except:
            print(annot_df["filename"][i])

    updated_df = pd.DataFrame.from_dict(updated_data)
    return updated_df


def generate_cropped_images_and_keypoints_df_without_zero_padding(
    annot_df, src="./data/images/", dest="./data/cropped_images/", final_img_size=512
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

    # pool = Pool()

    # for i in pool.map(f, range(len(annot_df))):
    for i in range(len(annot_df)):
        updated_data["filename"].append(annot_df["filename"][i])
        updated_data["class"].append(annot_df["class"][i])

        xmin_bbox = int(annot_df["xmin_bbox"][i])
        ymin_bbox = int(annot_df["ymin_bbox"][i])
        height_bbox = int(annot_df["height_bbox"][i])
        width_bbox = int(annot_df["width_bbox"][i])

        print(src + annot_df["filename"][i][:11] + ".jpg")
        img = cv2.imread(src + annot_df["filename"][i][:11] + ".jpg")

        ht, wd, cc = img.shape
        # create new image of desired size for padding.
        ww = 1000
        hh = 1000
        color = (0, 0, 0)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy : yy + ht, xx : xx + wd] = img

        finalXs = []
        finalYs = []
        finalVis = []
        dir1 = (width_bbox - height_bbox) // 2

        cropped = result[
            ymin_bbox + yy : ymin_bbox + height_bbox + yy,
            xmin_bbox + xx : xmin_bbox + width_bbox + xx,
        ]
        cropped = cv2.resize(cropped, (512, 512))
        for keypt in keypoint_names:
            newX = int(
                (512 / width_bbox) * (int(annot_df[keypt + "_x"][i]) - xmin_bbox)
            )
            newY = int(
                (512 / height_bbox) * (int(annot_df[keypt + "_y"][i]) - ymin_bbox)
            )

            if (
                int(annot_df[keypt + "_x"][i]) == 0
                and int(annot_df[keypt + "_y"][i]) == 0
            ):
                finalXs.append(0)
                finalYs.append(0)
                finalVis.append(int(annot_df[keypt + "_visible"][i]))
                continue
            finalXs.append(max(0, newX))
            finalYs.append(max(0, newY))
            finalVis.append(int(annot_df[keypt + "_visible"][i]))

        cv2.imwrite((dest + annot_df["filename"][i][:-4] + ".jpg"), cropped)

        cnt = 0
        for keypt in keypoint_names:
            updated_data[keypt + "_x"].append(finalXs[cnt])
            updated_data[keypt + "_y"].append(finalYs[cnt])
            updated_data[keypt + "_visible"].append(finalVis[cnt])
            cnt += 1

        # flip image

        flipped_img = cv2.flip(cropped, 1)
        updated_data["filename"].append(annot_df["filename"][i][:-4] + "_f.xml")
        updated_data["class"].append(annot_df["class"][i])

        flipped_Xs = []
        flipped_Ys = []
        flipped_vis = []

        for j in range(len(finalXs)):
            flipped_Xs.append(final_img_size - finalXs[j])
            flipped_Ys.append(finalYs[j])
            flipped_vis.append(finalVis[j])

        cnt = 0
        for keypt in keypoint_names:
            updated_data[keypt + "_x"].append(flipped_Xs[cnt])
            updated_data[keypt + "_y"].append(flipped_Ys[cnt])
            updated_data[keypt + "_visible"].append(flipped_vis[cnt])
            cnt += 1

        cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_f.jpg"), flipped_img)

        # add noise

        noise = np.random.randint(0, 10, (final_img_size, final_img_size, 3))
        noisy_img = noise + cropped

        updated_data["filename"].append(annot_df["filename"][i][:-4] + "_n.xml")
        updated_data["class"].append(annot_df["class"][i])

        cnt = 0
        for keypt in keypoint_names:
            updated_data[keypt + "_x"].append(finalXs[cnt])
            updated_data[keypt + "_y"].append(finalYs[cnt])
            updated_data[keypt + "_visible"].append(finalVis[cnt])
            cnt += 1

        cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_n.jpg"), noisy_img)

        # flipped_noisy_img

        flipped_noisy_img = cv2.flip(noisy_img, 1)
        updated_data["filename"].append(annot_df["filename"][i][:-4] + "_fn.xml")
        updated_data["class"].append(annot_df["class"][i])

        flipped_Xs = []
        flipped_Ys = []
        flipped_vis = []

        for j in range(len(finalXs)):
            flipped_Xs.append(final_img_size - finalXs[j])
            flipped_Ys.append(finalYs[j])
            flipped_vis.append(finalVis[j])

        cnt = 0
        for keypt in keypoint_names:
            updated_data[keypt + "_x"].append(flipped_Xs[cnt])
            updated_data[keypt + "_y"].append(flipped_Ys[cnt])
            updated_data[keypt + "_visible"].append(flipped_vis[cnt])
            cnt += 1

        cv2.imwrite(
            (dest + annot_df["filename"][i][:-4] + "_fn.jpg"), flipped_noisy_img
        )

        # rotate
        updated_data["filename"].append(annot_df["filename"][i][:-4] + "_r.xml")
        updated_data["class"].append(annot_df["class"][i])

        c = (final_img_size / 2, final_img_size / 2)

        s = final_img_size / 200.0 * 1.5
        r = 0
        sf = 0.2
        rf = 2
        s = s * (np.random.randn(1).dot(sf) + 1).clip(1 - sf, 1 + sf)[0]
        r = np.random.randn(1).dot(rf).clip(-2 * rf, 2 * rf)[0]

        rot_img = crop(
            cropped, c, s, [final_img_size, final_img_size], pad_scale=5, rot=r
        )

        cnt = 0
        for keypt in keypoint_names:
            if finalVis[cnt] == 0:
                temp = [0.0, 0.0, 0.0]
                updated_data[keypt + "_visible"].append(0)
            else:
                temp = transform(
                    [finalXs[cnt], finalYs[cnt]],
                    c,
                    s,
                    [final_img_size, final_img_size],
                    rot=r,
                )
                updated_data[keypt + "_visible"].append(1)

            cnt += 1
            updated_data[keypt + "_x"].append(temp[0])
            updated_data[keypt + "_y"].append(temp[1])

        # image inpainting to fill in 0 pixels
        img2gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        dst = cv2.inpaint(rot_img, mask_inv, 3, cv2.INPAINT_TELEA)

        cv2.imwrite((dest + annot_df["filename"][i][:-4] + "_r.jpg"), dst)

        # except:
        #   print(annot_df['filename'][i])

    updated_df = pd.DataFrame.from_dict(updated_data)
    return updated_df


if __name__ == "__main__":
    # move_files('/media/gaurav/Incremental_pose/data/PASCAL_VOC_2012/', dest='/media/gaurav/Incremental_pose/data/images/')
    # move_files('/media/gaurav/Incremental_pose/data/PASCAL2011_animal_annotation/', dest='/media/gaurav/Incremental_pose/data/annot/')
    annot_df = xml_to_df("/media/gaurav/Incremental_pose/data/annot/")
    # updated_df = generate_cropped_images_and_keypoints_df(annot_df)
    # updated_df.to_csv("./data/updated_df.csv", index=False)
    updated_df = generate_cropped_images_and_keypoints_df_without_zero_padding(
        annot_df,
        src="/media/gaurav/Incremental_pose/data/images/",
        dest="/media/gaurav/Incremental_pose/data/cropped_images_no_zero_padding/",
        final_img_size=512,
    )
    updated_df.to_csv(
        "/media/gaurav/Incremental_pose/data/updated_df_no_zero_padding.csv",
        index=False,
    )

