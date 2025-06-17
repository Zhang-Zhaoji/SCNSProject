import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from oasis.functions import deconvolve
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import f_oneway
from collections import defaultdict

def plot_cell_roi(all_roi_masks:np.ndarray, cids:np.ndarray=None,id:int=None)->None:
    """
    plot cell roi for id.
    if cids is not None, print cell id.
    if id == None, plot all roi.
    """
    plt.figure(figsize=(10,10))
    if id is not None:
        all_roi_mask = all_roi_masks[id,:,:]
    else:
        all_roi_mask = np.max(all_roi_masks,axis=0)
    plt.imshow(all_roi_mask,cmap='gray')
    if cids is not None and id is not None:
        print("cell id:",cids[id])
    plt.show()

def load_neural_npz(npz_path:str)->list[np.ndarray]:
    """
    load neural_data.npz
    """
    data:np.lib.npyio.NpzFile = np.load(npz_path,allow_pickle=True)
    print("check subtitles, should be ['ts', 'dff', 'all_roi_masks', 'cids', 'metadata']")
    print(data.files)
    # should be ['ts', 'dff', 'all_roi_masks', 'cids', 'metadata']
    ts, dff, all_roi_masks, cids, metadata = list(map(lambda x:data[x], data.files))
    return ts, dff, all_roi_masks, cids, metadata

def load_static_gating_csv(csv_path:str)->pd.DataFrame:
    """
    load static gating csv and do some preprocessing
    """
    static_gating = pd.read_csv(csv_path)
    return static_gating

def load_data(datas_path:str,mode:str = 'static_grating'):
    """
        Example data: ./data/VISal_three_session_B_501929146 \n
        ts (Time Seires): 
        (113852,) \n
        dff (dataframe of firing): 
        (172, 113852)  \n
        all_roi_masks (region of interest mask):
        (172, 512, 512) \n
        cids (Cell ids):
        (172,) \n
        metadata: \n
        {'sex': 'male', 'targeted_structure': 'VISal', 'ophys_experiment_id': 501929146, 'experiment_container_id': 511510715, 'excitation_lambda': '910 nanometers', 
        'indicator': 'GCaMP6f', 'fov': '400x400 microns (512 x 512 pixels)', 'genotype': 'Cux2-CreERT2/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/Ai93(TITL-GCaMP6f)', 
        'session_start_time': datetime.datetime(2016, 2, 11, 9, 54, 37), 'session_type': 'three_session_B', 'specimen_name': 'Cux2-CreERT2;Camk2a-tTA;Ai93-222426', 
        'cre_line': 'Cux2-CreERT2/wt', 'imaging_depth_um': 175, 'age_days': 111, 'device': 'Nikon A1R-MP multiphoton microscope', 'device_name': 'CAM2P.2', 'pipeline_version': '3.0'}
    """
    for data_path in os.listdir(datas_path):
        if data_path.endswith('.npz'):
            ts, dff, all_roi_masks, cids, metadata = load_neural_npz(os.path.join(datas_path, data_path))

            print("metadata:",end='')
            print(metadata)
            print("ts shape:",end='')
            print(ts.shape)
            print("dff shape:",end='')
            print(dff.shape)
            print("all_roi_masks shape:",end='')
            print(all_roi_masks.shape)
            print("cids shape:",end='')
            print(cids.shape)
            # plot_cell_roi(all_roi_masks,cids)
        elif data_path.endswith('.csv') and data_path.find(mode) != -1 and mode == 'static_grating':
            tgt_data = load_static_gating_csv(os.path.join(datas_path, data_path))
            print("static_gating:")
            print(tgt_data)
        else:
            pass
    return ts, dff, all_roi_masks, cids, metadata, tgt_data

def get_SG_FR(ts:np.ndarray, dff:np.ndarray,tgt_data:pd.DataFrame):
    """
    get static gating firing rate and static grating information.\n
    each time stamp represent a frame? \n
    113627 for 17.23731 - 3802.9922 in ts, and each frame represents around 0.03325seconds (30.07 fps)
    """
    orientation = tgt_data['orientation'].to_numpy()
    spatial_frequency = tgt_data['spatial_frequency'].to_numpy()
    phase = tgt_data['phase'].to_numpy()
    non_NaN_idx = [idx for idx in range(len(orientation)) if not np.isnan(orientation[idx])]
    orientation = orientation[non_NaN_idx]
    spatial_frequency = spatial_frequency[non_NaN_idx]
    phase = phase[non_NaN_idx]
    avail_times = list(zip(tgt_data['start'].to_list(),tgt_data['end'].to_list()))
    avail_times = [avail_times[idx] for idx in non_NaN_idx]
    firing_data, _ts = process_Firing_Data(ts, dff, avail_times)
    return orientation, spatial_frequency, phase, firing_data, _ts

def process_Firing_Data(ts:np.ndarray, dff:np.ndarray,avail_times:list[tuple[int,int]],mode:str = 'sum')->tuple[np.ndarray,np.ndarray]:
    """
    process Firing Data, consider different processing mode including sum, mean, max, median...
    """
    de_conv_data = np.array([deconvolve(dff[_i,:], penalty=1)[1] for _i in range(dff.shape[0])]) # only s
    plt.plot(ts,dff[0,:],label='dff')
    plt.plot(ts,de_conv_data[0,:],label = 'deconvolution')
    plt.title('Original floroscope signal and processed deconvolution signal')
    plt.legend()
    plt.show()
    firing_data = []
    de_conv_data = dff
    ts_data = [ts[start:end] for start, end in avail_times]
    for avail_time in avail_times:
        start, end = avail_time

        # c : array, shape (T,) The inferred denoised fluorescence signal at each time-bin. 
        # s : array, shape (T,) Discretized deconvolved neural activity (spikes). 
        # b : float Fluorescence baseline value. 
        # g : tuple of float Parameters of the AR(2) process that models the fluorescence impulse response. 
        # lam: float Optimal Lagrange multiplier for noise constraint under L1 penalty
        
        if mode == 'sum':
            firing_data.append(np.sum(de_conv_data[:,start:end],axis=1)/(ts[end]-ts[start])) # count firing spike 
        elif mode == 'mean':
            firing_data.append(np.mean(de_conv_data[:,start:end],axis=1))
        elif mode == 'max':
            firing_data.append(np.max(de_conv_data[:,start:end],axis=1))
        elif mode == 'median':
            firing_data.append(np.median(de_conv_data[:,start:end],axis=1))
        else:
            raise ValueError("mode should be mean, max, median, other mode is not supported till now.")
    return np.array(firing_data), np.array(ts_data)
    
def extract_feature_lists(firing_data:np.ndarray,orientation:np.ndarray,spatial_frequency:np.ndarray,phase:np.ndarray):
    """
    We have firing data, orientation, spatial_frequency, phase. And we want to calculate for each category, do we have some sort of way to find if there are some relationship between them?
    shape example:
    (5805,) (5805, 172) (5805,) (5805,)
    orientation = np.array([0, 30, 60, 90, 120, 150])
    spatial_frequency = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    phase = np.array([0, 0.25, 0.5, 0.75])
    Thus, there are 6*5*4 = 120 categories.
    """
    print(f'orientation.shape={orientation.shape},firing_data.shape={firing_data.shape},spatial_frequency.shape={spatial_frequency.shape},phase.shape={phase.shape}')
    firing_datas = []
    for orientation_ in np.unique(orientation):
        for spatial_frequency_ in np.unique(spatial_frequency):
            for phase_ in np.unique(phase):
                index = np.where((orientation==orientation_)&(spatial_frequency==spatial_frequency_)&(phase==phase_))
                firing_data_ = firing_data[index]
                firing_datas.append([firing_data_,orientation_,spatial_frequency_,phase_])
    return firing_datas

def similarity_metric(firing_datas, N_neurons):
    modulation = {
        'orientation': np.zeros(N_neurons),
        'spatial_frequency': np.zeros(N_neurons),
        'phase': np.zeros(N_neurons)
    }

    print("Grouping data...")
    group_orientation = defaultdict(list)
    group_spatial_frequency = defaultdict(list)
    group_phase = defaultdict(list)

    for j in range(len(firing_datas)):
        o, s, p = firing_datas[j][1], firing_datas[j][2], firing_datas[j][3]
        responses = firing_datas[j][0]
        for i in range(N_neurons):
            group_orientation[(o, i)].extend(responses[:, i])
            group_spatial_frequency[(s, i)].extend(responses[:, i])
            group_phase[(p, i)].extend(responses[:, i])

    print("Computing ANOVA...")
    for i in range(N_neurons):
        # Orientation
        groups = [group_orientation[(k, i)] for k in np.unique(orientation)]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            _, p = f_oneway(*groups)
            modulation['orientation'][i] = -np.log10(p)

        # Spatial Frequency
        groups = [group_spatial_frequency[(k, i)] for k in np.unique(spatial_frequency)]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            _, p = f_oneway(*groups)
            modulation['spatial_frequency'][i] = -np.log10(p)

        # Phase
        groups = [group_phase[(k, i)] for k in np.unique(phase)]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            _, p = f_oneway(*groups)
            modulation['phase'][i] = -np.log10(p)

    return modulation

def plot_modulation_radar_tiled(modulation, neuron_indices,
                                neurons_per_subplot=3,
                                subplots_per_fig=(5, 6),
                                cmap='BrBG_r',
                                title=None,
                                sig_threshold=2):  # -log10(0.01) = 2
    """
    在多个子图中批量展示神经元的调制雷达图，每个子图可以叠加多个神经元
    
    参数:
        modulation: dict，包含 orientation / sf / phase 的调制强度 (-log10(p))
        neuron_indices: list of int，要绘制的神经元编号
        neurons_per_subplot: 每个子图最多显示多少个神经元（默认 3）
        subplots_per_fig: 每张图有多少个子图（行数 x 列数），默认 (5, 6)
        cmap: 颜色映射
        title: 图表标题
        sig_threshold: 显著性阈值，默认 2（即 p < 0.01）
    """
    labels = np.array(['Orientation', 'Spatial Frequency', 'Phase'])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    total_subplots_needed = (len(neuron_indices) + neurons_per_subplot - 1) // neurons_per_subplot
    rows, cols = subplots_per_fig
    subplots_per_figure = rows * cols

    for fig_idx in range(0, total_subplots_needed, subplots_per_figure):
        fig_start = fig_idx
        fig_end = min(fig_idx + subplots_per_figure, total_subplots_needed)
        current_batches = [neuron_indices[i*neurons_per_subplot : (i+1)*neurons_per_subplot]
                           for i in range(fig_start, fig_end)]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), subplot_kw=dict(polar=True))
        if rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        elif cols == 1 and rows > 1:
            axes = axes.reshape(-1, 1)

        norm = Normalize(vmin=0, vmax=5)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for ax_idx, batch in enumerate(current_batches):
            ax = axes[ax_idx // cols, ax_idx % cols]
            title_parts = []

            for idx in batch:
                stats = np.array([
                    modulation['orientation'][idx],
                    modulation['spatial_frequency'][idx],
                    modulation['phase'][idx]
                ])
                stats = np.concatenate((stats, [stats[0]]))  # Close the radar chart

                color = sm.to_rgba(stats[:-1].mean())
                ax.plot(angles, stats, color=color, linewidth=1.5, alpha=0.8, label=f"Neuron {idx}")
                ax.fill(angles, stats, color=color, alpha=0.15)
                title_parts.append(f"{idx}")

                # 添加显著性标记 ★
                for i in range(num_vars):
                    if stats[i] > sig_threshold:
                        ax.text(angles[i], stats[i], '*', fontsize=24, ha='center', va='center')
                        # ax.plot(angles[i], stats[i], marker='*', markersize=10, color='red', linestyle='none')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticklabels([])
            ax.set_title("Neurons: " + ", ".join(title_parts), fontsize=10)

        # 关闭多余的子图
        for ax_idx in range(len(current_batches), rows * cols):
            fig.delaxes(axes.flatten()[ax_idx])

        if title is not None:
            fig.suptitle(title, fontsize=32)
        else:
            fig.suptitle(f"Modulation Profiles - Batch {fig_idx // subplots_per_figure + 1}", fontsize=32)

        fig.tight_layout()
        plt.show()

def read_Decoding_csv(data_path='../data', region_type = 'VISal', session_letter = 'A'):
    '''
    Read Decoding csv file, return a dict containing all the data and session folder name
    '''
    data_dict = {}
    folder_names:list[str] = os.listdir(data_path)
    tgt_name = ''
    for folder_name in folder_names:
        if folder_name.startswith(region_type) and folder_name.split('_')[-2] == session_letter:
            tgt_name = folder_name
            break
    else:
        print('No such folder!')
        raise RuntimeError(f'No such folder with region type {region_type} and session letter {session_letter}! in {data_path} as target!, {os.listdir(data_path)}')
    tgt_path = os.path.join(data_path, tgt_name)
    csv_names = [name for name in os.listdir(tgt_path) if name.endswith('.csv')]
    for csv_name in csv_names:
        csv_path = os.path.join(tgt_path, csv_name)
        data_dict[csv_name] = pd.read_csv(csv_path)
    print('we have keys in data_dict, including:')
    for key in data_dict.keys():
        print(key)
    return data_dict, tgt_name


if __name__ == '__main__':
    # data_path = './data/VISal_three_session_A_501876401'
    # manifest_path = ''
    # stimuli = 'natural_scenes'
    # session_id = 504065
    # load_data(data_path, manifest_path, stimuli, session_id)
    tgt_data_path = './data/VISal_three_session_B_501929146'
    print(os.path.exists(tgt_data_path))
    print(os.getcwd())
    ts, dff, all_roi_masks, cids, metadata, tgt_data = load_data(tgt_data_path)
    position = metadata.tolist()
    print('================================================')
    print('Caution, the position of this file is about the brain part of', position['targeted_structure'])
    print('================================================')

    orientation, spatial_frequency, phase, firing_data, _ts = get_SG_FR(ts,dff,tgt_data)
    read_Decoding_csv()