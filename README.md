# ILAPE-RBF-k-DPP

To run the code you will have to download the AnimalPose Dataset (https://sites.google.com/view/animal-pose/). Extract the keypoints and store them in csv files using the utils.py files. You can further generate classwise csv files if you wish to using generate_classwise_df.py

After generating the data you can run the incremental learning setup using the following command. [Please refer to the config file stored in ILAPE/configs]

```
python train_incremental.py --exp-id 10 --cfg ./configs/rbf_dpp_50_aug.yaml 
```

Website link for the project - https://sites.google.com/view/ilape-rbf-kdpp/home

## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={Incremental Learning for Animal Pose Estimation using {RBF} k-DPP},
      author    = {Gaurav Kumar Nayak and
                Het Shah and
                Anirban Chakraborty},
      booktitle={BMVC},
      year={2021}
    }

## References: 
[1] AlphaPose - https://github.com/MVIG-SJTU/AlphaPose 

[2] DPPy - https://github.com/guilgautier/DPPy

[3] Python Thin Plate Splines - https://github.com/cheind/py-thin-plate-spline

[4] Pose Resnet - https://github.com/microsoft/human-pose-estimation.pytorch

[5] DET - https://github.com/aimagelab/mammoth