import numpy as np
import pandas as pd

def L2_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.abs(np.sum((v2 - v1)**2)))

def calc_dist(avg_vector, animal_list):
    result = []
    for i in animal_list:
        fname = i[0]
        keypoints = i[1]
        dist = L2_dist(avg_vector, keypoints)
        temp = ((fname, keypoints), dist)
        result.append(temp)
    return result

def generate_csv(animal_class="cat", annot_df_path="/media2/het/data/updated_df.csv"):
    keypoint_names = ['L_Eye', 'R_Eye', 'Nose', 'L_EarBase', 'R_EarBase',
                    'L_F_Elbow', 'L_F_Paw', 'R_F_Paw', 'R_F_Elbow',
                    'L_B_Paw', 'R_B_Paw', 'L_B_Elbow', 'R_B_Elbow',
                    'L_F_Knee', 'R_F_Knee', 'L_B_Knee', 'R_B_Knee']

    annot_df = pd.read_csv(annot_df_path)

    animal_df = annot_df.loc[annot_df['class'] == animal_class]

    animal_list = []
    keypoints_list = []
    for i in animal_df.index:
        temp = (animal_df['filename'][i], [])
        for keypt in keypoint_names:
            temp[1].append([animal_df[keypt+'_x'][i], animal_df[keypt+'_y'][i], animal_df[keypt+'_visible'][i]])
        
        animal_list.append(temp)
        keypoints_list.append(temp[1])

    keypoints_list = np.array(keypoints_list)

    animal_avg = np.mean(keypoints_list, axis=0)

    final_animal_vec = calc_dist(animal_avg, animal_list)

    final_animal_vec.sort(key=lambda x: x[1])

    final_animal_df_dict = {}

    final_animal_df_dict['filename'] = []
    final_animal_df_dict['class'] = []
    for i in range(len(keypoint_names)):
        final_animal_df_dict[keypoint_names[i]+'_x'] = []
        final_animal_df_dict[keypoint_names[i]+'_y'] = []
        final_animal_df_dict[keypoint_names[i]+'_visible'] = []

    for i in final_animal_vec:
        final_animal_df_dict['filename'].append(i[0][0])
        final_animal_df_dict['class'].append(animal_class)
        for j in range(len(keypoint_names)):
            final_animal_df_dict[keypoint_names[j]+'_x'].append(i[0][1][j][0])
            final_animal_df_dict[keypoint_names[j]+'_y'].append(i[0][1][j][1])
            final_animal_df_dict[keypoint_names[j]+'_visible'].append(i[0][1][j][2])

    final_animal_df = pd.DataFrame(final_animal_df_dict)
    final_animal_df.to_csv("/media2/het/data/"+animal_class+".csv", index=False)

generate_csv("cat")
generate_csv("cow")
generate_csv("dog")
generate_csv("horse")
generate_csv("sheep")