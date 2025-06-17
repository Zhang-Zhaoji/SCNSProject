import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from oasis.functions import deconvolve

from load_data import read_Decoding_csv, load_neural_npz
def build_drift_grating_stim(all_data: pd.DataFrame, time_length=1000000) -> np.ndarray:
    print('timelength = ',time_length)
    drift_headers = ['temporal_frequency', 'orientation', 'blank_sweep']
    non_nan_datas = all_data[~all_data.isnull().any(axis=1)].copy()
    blank_data = all_data[all_data.isnull().any(axis=1)].copy()
    # print(non_nan_datas) # [598 rows x 6 columns]
    # print(blank_data) # [30 rows x 6 columns]
    unique_stimuli = non_nan_datas[drift_headers[:2]].drop_duplicates().values
    stim_to_index = {tuple(row): i for i, row in enumerate(unique_stimuli)}

    # 快速获取每个非空数据对应的刺激索引
    def get_stim_index(row):
        return stim_to_index.get(tuple(row[drift_headers[:2]]), -1)

    non_nan_datas['stim_index'] = non_nan_datas.apply(get_stim_index, axis=1)

    # 初始化输出矩阵
    drift_gate_vector = np.zeros((len(unique_stimuli) + 1, time_length), dtype=np.float32)  # +1 for blank sweeps

    # # 填充非空白刺激
    for _, row in tqdm(non_nan_datas.iterrows()):
        stim_index = int(row['stim_index'])
        start = int(row['start'])
        end = int(row['end'])
        # distance = end - start
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # drift_gate_vector[stim_index, _init_start:end] = Gaussian_kernel
        drift_gate_vector[stim_index, start:end] = 1

    # # 填充空白刺激
    for _, row in blank_data.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        drift_gate_vector[-1, start:end] = 1
        # 
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # drift_gate_vector[-1, _init_start:end] = Gaussian_kernel
    print('Drift grating stimulus_mat.shape = ',drift_gate_vector.shape)
    return drift_gate_vector

def build_static_grating_stim(all_data: pd.DataFrame, time_length=1000000) -> np.ndarray:
    '''Test later'''
    print('timelength = ',time_length)
    drift_headers = ['orientation','spatial_frequency','phase']
    non_nan_datas = all_data[~all_data.isnull().any(axis=1)].copy()
    blank_data = all_data[all_data.isnull().any(axis=1)].copy()
    #print(non_nan_datas) # [598 rows x 6 columns]
    #print(blank_data) # [30 rows x 6 columns]
    unique_stimuli = non_nan_datas[drift_headers[:3]].drop_duplicates().values
    stim_to_index = {tuple(row): i for i, row in enumerate(unique_stimuli)}

    # 快速获取每个非空数据对应的刺激索引
    def get_stim_index(row):
        return stim_to_index.get(tuple(row[drift_headers[:3]]), -1)

    non_nan_datas['stim_index'] = non_nan_datas.apply(get_stim_index, axis=1)

    # 初始化输出矩阵
    static_gate_vector = np.zeros((len(unique_stimuli) + 1, time_length), dtype=np.float32)  # +1 for blank sweeps

    # # 填充非空白刺激
    for _, row in tqdm(non_nan_datas.iterrows()):
        stim_index = int(row['stim_index'])
        start = int(row['start'])
        end = int(row['end'])
        static_gate_vector[stim_index, start:end] = 1
        # 
        # distance = end - start
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[stim_index, _init_start:end] = Gaussian_kernel

    # # 填充空白刺激
    for _, row in blank_data.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        static_gate_vector[-1, start:end] = 1
        # 
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[-1, _init_start:end] = Gaussian_kernel
    print('Static grating stimulus_mat.shape = ',static_gate_vector.shape)
    return static_gate_vector

def build_nature_scenes_grating_stim(all_data: pd.DataFrame, time_length=1000000) -> np.ndarray:
    print('timelength = ',time_length)
    drift_headers = ['frame']
    non_nan_datas = all_data[~all_data.isnull().any(axis=1)].copy()
    blank_data = all_data[all_data.isnull().any(axis=1)].copy()
    # print(non_nan_datas) # [598 rows x 6 columns]
    
    #print(blank_data) # [30 rows x 6 columns]
    unique_stimuli = non_nan_datas[drift_headers[:1]].drop_duplicates().values
    stim_to_index = {tuple(row): i for i, row in enumerate(unique_stimuli)}
    # 快速获取每个非空数据对应的刺激索引
    def get_stim_index(row):
        return stim_to_index.get(tuple(row[drift_headers[:1]]), -1)

    non_nan_datas['stim_index'] = non_nan_datas.apply(get_stim_index, axis=1)

    # 初始化输出矩阵
    Scenes_vector = np.zeros((len(unique_stimuli) + 1, time_length), dtype=np.float32)  # +1 for blank sweeps

    # # 填充非空白刺激
    for _, row in tqdm(non_nan_datas.iterrows()):
        stim_index = int(row['stim_index'])
        start = int(row['start'])
        end = int(row['end'])
        Scenes_vector[stim_index, start:end] = 1
        # end = int(row['end'])
        # distance = end - start
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[stim_index, _init_start:end] = Gaussian_kernel

    # # 填充空白刺激
    for _, row in blank_data.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        Scenes_vector[-1, start:end] = 1
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[-1, _init_start:end] = Gaussian_kernel
    print('Nature Scene stimulus_mat.shape = ',Scenes_vector.shape)
    return Scenes_vector

def build_nature_movie_grating_stim(all_data: pd.DataFrame, time_length=1000000) -> np.ndarray:
    """
    暂时没想好怎么把movie的也转化成非one-hot的向量，等等之后我们再做吧。
    """
    print('timelength = ',time_length)
    drift_headers = ['frame','repeat']
    non_nan_datas = all_data[~all_data.isnull().any(axis=1)].copy()
    blank_data = all_data[all_data.isnull().any(axis=1)].copy()
    # print(non_nan_datas) # [598 rows x 6 columns]
    
    #print(blank_data) # [30 rows x 6 columns]
    unique_stimuli = non_nan_datas[drift_headers[:1]].drop_duplicates().values
    stim_to_index = {tuple(row): i for i, row in enumerate(unique_stimuli)}
    # 快速获取每个非空数据对应的刺激索引
    def get_stim_index(row):
        return stim_to_index.get(tuple(row[drift_headers[:1]]), -1)

    non_nan_datas['stim_index'] = non_nan_datas.apply(get_stim_index, axis=1)

    # 初始化输出矩阵
    movie_vector = np.zeros((len(unique_stimuli) + 1, time_length), dtype=np.float32)  # +1 for blank sweeps

    # # 填充非空白刺激
    for _, row in tqdm(non_nan_datas.iterrows()):
        stim_index = int(row['stim_index'])
        start = int(row['start'])
        end = int(row['end'])
        movie_vector[stim_index, start:end] = 1
        

        # distance = end - start
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[stim_index, _init_start:end] = Gaussian_kernel

    # # 填充空白刺激
    for _, row in blank_data.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        movie_vector[stim_index, start:end] = 1
        # _init_start = max(start - distance//2,0)
        # _init_distance = end - _init_start
        # input_sequence = np.linspace(_init_start, end, _init_distance)
        # Gaussian_kernel = np.exp(-(input_sequence - start) ** 2 / (2 * distance ** 2))
        # static_gate_vector[-1, _init_start:end] = Gaussian_kernel
    print('Movie stimulus_mat.shape = ',movie_vector.shape)
    return movie_vector

def stimulus_mat2vec(stimulus_mat):
    """We want to convert stimulus matrix(n_stim,ts) to vector(ts)"""
    ans = np.sum(stimulus_mat * np.arange(stimulus_mat.shape[0]).reshape(-1,1),axis=0)
    return ans

def get_contained_names(information_dicts):
    names = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie','spontaneous']
    # times_top = ['start','end']
    # drift_headers = ['temporal_frequency', 'orientation', 'blank_sweep']
    # static_headers = ['orientation','spatial_frequency','phase']
    # natural_scenes_headers = ['frame']
    # natural_movie_headers = ['frame','repeat']
    # spontaneous_headers = []
    contained_names = {}
    for name in names:
        for key_ in information_dicts.keys():
            if key_.find(name) != -1:
                contained_names[name] = information_dicts[key_]
                break
    # thus we have write down the chain that we have some names in contained_names, and we have corresponding names in information_dicts using
    # information_dicts[contained_names[name]] for name in contained_names.keys()
    print(contained_names.keys())
    contained_keys = list(contained_names.keys())
    return contained_names, contained_keys

def create_stimulus_mat(contained_names,ts):
    """
    加载几个不同的刺激，不过现在我我们只选了grating是和natural scenes，因为movie帧比较多不好弄，spontaneous没意思
    """
    total_stimulus = []
    useful_stimulus_name = ['drifting_gratings', 'static_gratings', 'natural_scenes']#, 'natural_movie','spontaneous']
    for _name, info_dataframe in contained_names.items():
        if _name == useful_stimulus_name[0]:
            total_stimulus.append(build_drift_grating_stim(info_dataframe,time_length=len(ts)))
        elif _name == useful_stimulus_name[1]:
            total_stimulus.append(build_static_grating_stim(info_dataframe,time_length=len(ts)))
        elif _name == useful_stimulus_name[2]:
            continue
            # we do not want to take natural_scenes into our consideration at this point
            # total_stimulus.append(build_nature_scenes_grating_stim(info_dataframe,time_length=len(ts)))
    total_stimulus = np.concatenate(total_stimulus,axis=0)
    print(total_stimulus.shape)
    return total_stimulus