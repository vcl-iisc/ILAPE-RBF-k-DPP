# script that simply calls rotate function to augment the dataset 
from utils import *

rotate_elbow_knee_limbs(
    input_dir="../data/cropped_images_no_zero_padding",
    output_dir="../data/rotated_images_no_zero_padding/",
    input_csv_file="../data/updated_df_no_zero_padding.csv",
    output_csv_file="../data/updated_df_rotated_random_all_three_cat.csv",
    animal_class="cat", # can be changed for any animal class
)