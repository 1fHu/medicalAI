import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time


def extract_features_with_cord(photo_address):
    """extract all contours in one photo, 
    and store its coordinations, and morphology information 

    param: photo_address (Path): photo_address of one frame
    
    return: a dictionary of all cells found in one photo with its coordinations
            e.g. {(256, 312) ->[234, 412, 422, 222, 345], ()->[...], ...] 
    """
    gray_img = cv2.imread(photo_address, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    one_img_trait = {}
    min_area = 150
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        
    
        matrics = [area, perimeter, circularity, radius, circle_area]
        
        # one_cell = [(x, y), matrics]
   
        one_img_trait[(x, y)] = matrics
    
    return one_img_trait


def extract_features_in_folder(folder):
    """Process the tif files and extract the features
    with coordinates in it

    Args:
        folder (str): string address of a folder

    Returns:
        dictionary: the frame as key pointing to the trait
        2 -> ( (x, y) -> [a, b, e, d, e]), 3 -> ...
        
    """
    frame_counter = 1
    folder = Path(folder)
    result = {}
    for file in folder.iterdir():
        target_str = f"T{frame_counter:03d}"
        if (file.is_file() and file.suffix == '.tif'
            and target_str in file.name):
            one_frame_data = extract_features_with_cord(file)
            result[frame_counter] = one_frame_data
            frame_counter += 1
    
    return result



def num_tracks(file_allspot):
    df_spot = pd.read_csv(file_allspot)
    df_spot = df_spot.iloc[3:].copy()
    
    df_spot.dropna(subset=["TRACK_ID"], inplace = True)
    df_spot["TRACK_ID"] = df_spot["TRACK_ID"].astype(int)
    
    # unique_track_ids = df_spot["TRACK_ID"].unique()
    
    return df_spot


def find_morp_by_cord_frame(all_features, coordinates, frame):
    # e.g. return [645.0,10.56,0.68, 18.38, 1062.18] 
    (tar_x, tar_y) = coordinates
    one_frame_feature = all_features[frame]
    min_dist = float('inf')
    best_key = None
    
    if not one_frame_feature.keys():
        # print(f"frame {frame} has no feature")
        raise Exception(f" {frame} frame has no feature ")
    for (exac_x, exac_y) in one_frame_feature.keys():
        
        dist = (exac_x - tar_x) ** 2 + (exac_y - tar_y) ** 2
        if dist < min_dist:
            min_dist = dist
            best_key = (exac_x, exac_y)
    
    if best_key is not None: 
        # print(best_key)
        return one_frame_feature[best_key]
    else:
        raise Exception(f"No match found within search radius in {frame}")
    
def calc_array_data(array):
    """calculate the mean, std, min, max, median, variance

    Args:
        array (np.array): 2D array

    Returns:
        array(np.array): 6*5
    """
    if len(array) == 0:
        raise ValueError("emtpy array, cannot calculate")
    
    data_2d = np.array([arr.flatten() for arr in array])

    df = pd.DataFrame(data_2d, columns=[f'Value_{i+1}' for i in range(5)])
    agg_functions_list = ['mean', 'std', 'min', 'max', 'median', 'var']

    aggregated_series = df.agg(agg_functions_list)
    combined_one_cell = aggregated_series.values
    return combined_one_cell   


def proc_one_cell_mor(df_allspot, all_feature, one_track_id):
    """ store all coordinates in a dictionary with the key of one frame,
        traverse through the number of frame and use position to find trait
    """
    track_data = df_allspot[df_allspot["TRACK_ID"] == one_track_id].copy()
    
    positionXY = {} # one cell positions in all frame
    for (f, x, y) in zip(track_data["FRAME"], 
                         track_data["POSITION_X"], track_data["POSITION_Y"]):
        f = int(f)
        positionXY[f] = (float(x), float(y))
    
    result = []
    for frame in positionXY.keys():
        try:
            one_result = find_morp_by_cord_frame(all_feature, 
                                                positionXY[frame], frame+1)
        except Exception as e:
            # raise Exception(f"There is exception happen in {one_track_id} track, {e}")
            continue
        # frame in file name begins with 1, in csv begin with 0
        
        result.append(one_result)
    
    return np.array(result)

def read_tracks(csv_track):
    df_track = pd.read_csv(csv_track, encoding='ISO-8859-1')
    df_track = df_track.iloc[3:].copy()
    df_track["NUMBER_SPOTS"] = pd.to_numeric(df_track["NUMBER_SPOTS"], errors='coerce')
    df_track = df_track[df_track["NUMBER_SPOTS"] > 2]
    df_track_filtered = df_track.iloc[:, [2, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]].copy()
    return df_track_filtered


def make_dyn_dict(df_track_filtered):
    # track_id 2 -> [a, b, c, d, ...], 3-> ... 
    
    track_id_dict = {}
    for row in df_track_filtered.itertuples(index=True):
        float_values = []
        for num in row[2:]:
            
            if pd.isna(num):
                num = 0.0
            else: 
                num = float(num)
            float_values.append(num)
            
        track_id_dict[int(row.TRACK_ID)] = np.array(float_values)

    return (track_id_dict)




    
    

def process_one_folder( allspot_csv = Path(r"F:\fullDataset\NCI2\XY1\_allspots.csv"), 
                        tracks_csv = Path(r"F:\fullDataset\NCI2\XY1\_tracks.csv"), 
                        photo_folder = Path(r"F:\fullDataset\NCI2\XY1")):
    """ traverse through track_id to cooresonding the trait from morphology
    to the trait in dynamics for one cell
    returns the whole group trait: num_of_cells * [6*5 + 15]
    """
    all_mor_feature = extract_features_in_folder(photo_folder)
    df_allspot_revised = num_tracks(allspot_csv)
    
    df_tracks_filtered = read_tracks(tracks_csv)
    tracks_id_dict = make_dyn_dict(df_tracks_filtered)
    
    # print(tracks_id_dict.keys())
    # return 
    result = []
    for track_id in tracks_id_dict.keys():
        # all track 0
        one_cell_mor_data = proc_one_cell_mor(df_allspot_revised,
                                              all_mor_feature, track_id)
        if len(one_cell_mor_data) == 0:
            continue
        one_cell_mor_data = calc_array_data(one_cell_mor_data)
        one_cell_dyn_data = tracks_id_dict[track_id]
        # if one_cell_dyn_data is None:
        #     print(f"[Warning] No dynamics data for track_id {track_id}")
        #     continue
        result.append([one_cell_mor_data, one_cell_dyn_data])
    
    # print(result[:5])
    return result


# process_one_folder()