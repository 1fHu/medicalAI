import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
import importlib
sys.path.append(r'E:/NYU/pencreatic cancer AI/0610')

import NewCombined
from NewCombined import process_one_folder
importlib.reload(NewCombined)

def collect_file_paths(data_root, subfolders_to_use):
    result = []

    for name in subfolders_to_use:
        root_path = Path(data_root) / name
        if not root_path.exists():
            print(f"Warning: folder {root_path} not found.")
            continue

        for xy_folder in root_path.glob("XY*"):
            if xy_folder.is_dir():
                track_files = list(xy_folder.glob("*_tracks.csv"))
                allspots_files = list(xy_folder.glob("*_allspots.csv"))

                if track_files and allspots_files:
                    result.append((xy_folder, track_files[0], allspots_files[0]))
                else:
                    print(f"Warning: Missing CSV files in {xy_folder}")

    return result

def make_folder_trait_dict(file_info):
    alltrait_one_folder = {}
    counter = 1
    for xy_path, track_csv, allspots_csv in tqdm(file_info, desc="Processing one folder"):
    # for xy_path, track_csv, allspots_csv in (file_info):   
        try: 
            one_group_trait = process_one_folder(allspots_csv, track_csv, xy_path)
            for each_cell in one_group_trait:
                alltrait_one_folder[counter] = (each_cell)
                counter += 1
        except Exception as e:
            print(f"Error processing {xy_path}:\n{e}\n")
            continue
    return alltrait_one_folder


def extract_dyn(alltrait_one_folder):
    data = []
    for num in alltrait_one_folder.keys():
        flat = alltrait_one_folder[num][1].flatten()
        if flat.shape[0] != 15:
            raise ValueError("输入 array 必须能展平成 15 个元素")
        track_info = flat.tolist() # [1,2,3, ...]
        track_info.insert(0, num)
        data.append(track_info)
    return data
    # print(data)


def proc_dyn_data(data):
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(16)])
    # 按 feature_9 降序排序
    sorted_df = df.sort_values(by='feature_9', ascending=False)

    # 筛选 feature_9 > 8 的行 track mean quality
    filtered_df = sorted_df[sorted_df['feature_9'] > 8].copy()

    # 计算 feature_1 + feature_4 track distance + mean speed
    filtered_df['sum_1_4'] = filtered_df['feature_1'] + filtered_df['feature_4']

    # 选出 sum_1_4 最大的前10行
    top10_df = filtered_df.nlargest(7, 'sum_1_4').drop(columns=['sum_1_4'])

    # 从原始 df 中移除这10行
    remaining_df = df[~df['feature_0'].isin(top10_df['feature_0'])]

    # 计算剩余行的均值
    mean_row = remaining_df.mean(numeric_only=True)
    mean_row['feature_0'] = 'mean'  # 添加标识

    # 合并 top10 和均值
    final_df = pd.concat([top10_df, pd.DataFrame([mean_row])], ignore_index=True)
    return final_df

def extract_10(final_df, alltrait_one_folder,final_trait_one_folder):
    # final_trait_one_folder = []
    for index, row in final_df.iterrows():          
        if isinstance(row['feature_0'], int): 

            fea_ind = row['feature_0']
            if fea_ind in alltrait_one_folder:
                final_trait_one_folder.append(alltrait_one_folder[fea_ind])
                del alltrait_one_folder[fea_ind]
            else:
                print(f"feature {fea_ind} does not exist")
    # return final_trait_one_folder

def calc_mean(alltrait_one_folder):
    arr_6x5_list = []
    arr_1x15_list = []

    # 遍历所有字典项
    for val in alltrait_one_folder.values():
        arr_6x5, arr_1x15 = val
        arr_6x5_list.append(arr_6x5)
        arr_1x15_list.append(arr_1x15)

    # 逐元素求平均
    mean_6x5 = np.mean(arr_6x5_list, axis=0)
    mean_1x15 = np.mean(arr_1x15_list, axis=0)

    # 最终保留为一个list形式，包含两个array
    final_result = [mean_6x5, mean_1x15]
    return final_result


def dataset_process(folders_to_include):
    
    data_root = r"F:/fullDataset"
    final_all_trait = []
    for group in tqdm(folders_to_include,desc="Process each group"):
        file_info = collect_file_paths(data_root, [group])
        trait_dic = make_folder_trait_dict(file_info)
        top10_mean_dyn = extract_dyn(trait_dic)
        df_top10 = proc_dyn_data(top10_mean_dyn)
        final_fea = []
        extract_10(df_top10, trait_dic, final_fea)
        one_mean = calc_mean(trait_dic)
        final_fea.append(one_mean)
        # print(len(final_fea))
        final_all_trait.append(final_fea)
    return final_all_trait

def main():
    folders_to_include = ["NCI2", "NCI6", "NCI6meso", "NCI8", "NCI8meso", "NCI9"]
    result = dataset_process(folders_to_include)
    print(len(result))
    print(len(result[3]))
    np.save("dataset_first.npy", np.array(result, dtype=object))
    
main()
