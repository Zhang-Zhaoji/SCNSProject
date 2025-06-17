import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import circvar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings
from load_data import read_Decoding_csv, load_neural_npz


class NeuronClusterAnalysis:
    def __init__(self, debug=False):
        self.scaler = RobustScaler()  # 使用RobustScaler增强鲁棒性
        self.results = {}
        self.debug = debug
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def calculate_tuning_features(self, ts, dff, tgt_dict, fillna_zero=True):
        """计算每个神经元的调谐特征，增强错误处理和调试"""
        if dff is None or len(dff) == 0:
            raise ValueError("dff 数据为空，无法提取特征")

        n_neurons = dff.shape[0]
        self._debug_print(f"开始为 {n_neurons} 个神经元计算调谐特征...")

        # Initialize feature dictionary with zeros
        tuning_features = {
            'neuron_id': np.arange(n_neurons),
            'dsi': np.zeros(n_neurons),
            'osi': np.zeros(n_neurons),
            'sf_pref': np.zeros(n_neurons),
            'tf_pref': np.zeros(n_neurons),
            'natural_response': np.zeros(n_neurons),
            'movie_response': np.zeros(n_neurons),
            'reliability': np.zeros(n_neurons),
            'lifetime_sparseness': np.zeros(n_neurons)
        }

        # Check if all dff values are zero
        if np.allclose(dff, 0):
            self._debug_print("警告: 所有神经活动值均为零!")
            return pd.DataFrame(tuning_features)

        # 处理每种刺激类型
        for stim_key, stim_df in tgt_dict.items():
            try:
                if 'static_gratings' in stim_key:
                    self._process_static_gratings(stim_df, dff, tuning_features)
                elif 'drifting_gratings' in stim_key:
                    self._process_drifting_gratings(stim_df, dff, tuning_features)
                elif 'natural_scenes' in stim_key:
                    self._process_natural_scenes(stim_df, dff, tuning_features)
                elif 'natural_movie' in stim_key:
                    self._process_natural_movie(stim_df, dff, tuning_features)
                elif 'spontaneous' in stim_key or 'total' in stim_key:
                    continue  # 跳过自发性活动和总活动
                else:
                    self._debug_print(f"未知刺激类型: {stim_key}")
            except Exception as e:
                self._debug_print(f"处理 {stim_key} 时出错: {str(e)}")
                continue

        # 检查所有特征是否维度一致
        feature_lengths = {k: len(v) for k, v in tuning_features.items()}
        if len(set(feature_lengths.values())) != 1:
            raise ValueError(f"特征维度不一致：{feature_lengths}")

        features_df = pd.DataFrame(tuning_features).replace([np.inf, -np.inf], np.nan)
        if fillna_zero:
            features_df = features_df.fillna(0)
            
        return features_df

    def _validate_time_window(self, start, end, dff_shape):
        """验证时间窗口是否有效"""
        if not (0 <= start < end <= dff_shape[1]):
            return False
        return True

    def _process_static_gratings(self, stim_df, dff, features):
        """处理静态光栅数据，根据提供的表头"""
        if stim_df.empty:
            self._debug_print("静态光栅数据为空")
            return

        # 确保列名正确
        required_columns = {'orientation', 'spatial_frequency', 'start', 'end'}
        if not required_columns.issubset(stim_df.columns):
            missing = required_columns - set(stim_df.columns)
            raise ValueError(f"静态光栅数据缺少必要列: {missing}")

        orientations = np.sort(stim_df['orientation'].unique())
        sfs = np.sort(stim_df['spatial_frequency'].unique())
        
        self._debug_print(f"处理静态光栅: {len(orientations)}方向, {len(sfs)}空间频率")

        for neuron in range(dff.shape[0]):
            ori_responses = []
            for ori in orientations:
                trials = stim_df[stim_df['orientation'] == ori]
                trial_responses = []
                for _, row in trials.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    if self._validate_time_window(start, end, dff.shape):
                        response = np.mean(dff[neuron, start:end])
                        if not np.isnan(response):
                            trial_responses.append(response)
                
                if trial_responses:
                    ori_responses.append(np.mean(trial_responses))
                else:
                    ori_responses.append(np.nan)

            # 计算方向选择性指数(OSI)
            valid_ori = np.array([r for r in ori_responses if not np.isnan(r)])
            if len(valid_ori) >= 2:
                pref_ori_idx = np.nanargmax(ori_responses)
                ortho_ori_idx = (pref_ori_idx + len(orientations)//4) % len(orientations)
                osi = (ori_responses[pref_ori_idx] - ori_responses[ortho_ori_idx]) / (
                    ori_responses[pref_ori_idx] + ori_responses[ortho_ori_idx] + 1e-6)
            else:
                osi = np.nan

            # 计算空间频率偏好
            sf_responses = []
            for sf in sfs:
                trials = stim_df[stim_df['spatial_frequency'] == sf]
                trial_responses = []
                for _, row in trials.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    if self._validate_time_window(start, end, dff.shape):
                        response = np.mean(dff[neuron, start:end])
                        if not np.isnan(response):
                            trial_responses.append(response)
                
                if trial_responses:
                    sf_responses.append(np.mean(trial_responses))
                else:
                    sf_responses.append(np.nan)

            valid_sf = np.array([r for r in sf_responses if not np.isnan(r)])
            if len(valid_sf) > 0:
                pref_sf_idx = np.nanargmax(sf_responses)
                sf_pref = sfs[pref_sf_idx]
            else:
                sf_pref = np.nan

            # 计算生命周期稀疏性
            valid_mean = np.mean(valid_ori) if len(valid_ori) > 0 else np.nan
            if len(valid_ori) > 0:
                sparseness = 1 - (valid_mean ** 2) / (np.mean([r ** 2 for r in valid_ori]) + 1e-6)
            else:
                sparseness = np.nan

            features['osi'][neuron] = osi
            features['sf_pref'][neuron] = sf_pref
            features['lifetime_sparseness'][neuron] = sparseness

    def _process_drifting_gratings(self, stim_df, dff, features):
        """处理漂移光栅数据，根据提供的表头"""
        if stim_df.empty:
            self._debug_print("漂移光栅数据为空")
            return

        # 确保列名正确
        required_columns = {'orientation', 'temporal_frequency', 'start', 'end'}
        if not required_columns.issubset(stim_df.columns):
            missing = required_columns - set(stim_df.columns)
            raise ValueError(f"漂移光栅数据缺少必要列: {missing}")

        directions = np.sort(stim_df['orientation'].unique())
        tfs = np.sort(stim_df['temporal_frequency'].unique())
        
        self._debug_print(f"处理漂移光栅: {len(directions)}方向, {len(tfs)}时间频率")

        for neuron in range(dff.shape[0]):
            dir_responses = []
            for dir in directions:
                trials = stim_df[stim_df['orientation'] == dir]
                trial_responses = []
                for _, row in trials.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    if self._validate_time_window(start, end, dff.shape):
                        response = np.mean(dff[neuron, start:end])
                        if not np.isnan(response):
                            trial_responses.append(response)
                
                if trial_responses:
                    dir_responses.append(np.mean(trial_responses))
                else:
                    dir_responses.append(np.nan)

            # 计算方向选择性指数(DSI)
            valid_dir = np.array([r for r in dir_responses if not np.isnan(r)])
            if len(valid_dir) >= 2:
                pref_dir_idx = np.nanargmax(dir_responses)
                null_dir_idx = (pref_dir_idx + len(directions)//2) % len(directions)
                dsi = (dir_responses[pref_dir_idx] - dir_responses[null_dir_idx]) / (
                    dir_responses[pref_dir_idx] + dir_responses[null_dir_idx] + 1e-6)
            else:
                dsi = np.nan

            # 计算时间频率偏好
            tf_responses = []
            for tf in tfs:
                trials = stim_df[stim_df['temporal_frequency'] == tf]
                trial_responses = []
                for _, row in trials.iterrows():
                    start, end = int(row['start']), int(row['end'])
                    if self._validate_time_window(start, end, dff.shape):
                        response = np.mean(dff[neuron, start:end])
                        if not np.isnan(response):
                            trial_responses.append(response)
                
                if trial_responses:
                    tf_responses.append(np.mean(trial_responses))
                else:
                    tf_responses.append(np.nan)

            valid_tf = np.array([r for r in tf_responses if not np.isnan(r)])
            if len(valid_tf) > 0:
                pref_tf_idx = np.nanargmax(tf_responses)
                tf_pref = tfs[pref_tf_idx]
            else:
                tf_pref = np.nan

            features['dsi'][neuron] = dsi
            features['tf_pref'][neuron] = tf_pref

    def _process_natural_scenes(self, stim_df, dff, features):
        """处理自然场景数据，根据提供的表头"""
        if stim_df.empty:
            self._debug_print("自然场景数据为空")
            return

        # 确保列名正确
        required_columns = {'start', 'end'}
        if not required_columns.issubset(stim_df.columns):
            missing = required_columns - set(stim_df.columns)
            raise ValueError(f"自然场景数据缺少必要列: {missing}")

        self._debug_print(f"处理自然场景: {len(stim_df)} trials")

        for neuron in range(dff.shape[0]):
            responses = []
            segments = []
            
            for _, row in stim_df.iterrows():
                start, end = int(row['start']), int(row['end'])
                if self._validate_time_window(start, end, dff.shape):
                    response = np.mean(dff[neuron, start:end])
                    if not np.isnan(response):
                        responses.append(response)
                        segments.append(dff[neuron, start:end])

            features['natural_response'][neuron] = np.mean(responses) if responses else np.nan

            # 计算可靠性
            if len(segments) >= 2:
                try:
                    # 确保所有片段长度相同
                    min_length = min(len(s) for s in segments)
                    trimmed_segments = [s[:min_length] for s in segments]
                    corr_matrix = np.corrcoef(trimmed_segments)
                    reliability = np.mean(corr_matrix[np.triu_indices(len(trimmed_segments), k=1)])
                    features['reliability'][neuron] = reliability
                except:
                    features['reliability'][neuron] = np.nan
            else:
                features['reliability'][neuron] = np.nan

    def _process_natural_movie(self, stim_df, dff, features):
        """处理自然电影数据，根据提供的表头"""
        if stim_df.empty:
            self._debug_print("自然电影数据为空")
            return

        # 确保列名正确
        required_columns = {'start', 'end'}
        if not required_columns.issubset(stim_df.columns):
            missing = required_columns - set(stim_df.columns)
            raise ValueError(f"自然电影数据缺少必要列: {missing}")

        self._debug_print(f"处理自然电影: {len(stim_df)} trials")

        for neuron in range(dff.shape[0]):
            responses = []
            for _, row in stim_df.iterrows():
                start, end = int(row['start']), int(row['end'])
                if self._validate_time_window(start, end, dff.shape):
                    response = np.mean(dff[neuron, start:end])
                    if not np.isnan(response):
                        responses.append(response)
            
            features['movie_response'][neuron] = np.mean(responses) if responses else np.nan

    def cluster_neurons(self, features_df, n_clusters=4):
        feature_cols = ['dsi', 'osi', 'sf_pref', 'tf_pref',
                        'natural_response', 'movie_response',
                        'lifetime_sparseness', 'reliability']

        # 筛选有效特征列
        valid_cols = []
        for col in feature_cols:
            values = features_df[col].values
            if not np.all(np.isnan(values)) and np.var(values[~np.isnan(values)]) > 1e-6:
                valid_cols.append(col)
            else:
                self._debug_print(f"跳过无效特征列: {col} (全为NaN或方差为零)")

        if not valid_cols:
            raise ValueError("所有特征列均无效，无法聚类")

        X = features_df[valid_cols].values
        valid_rows = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_rows]

        if X_valid.shape[0] < n_clusters:
            raise ValueError(f"有效样本数 {X_valid.shape[0]} 少于聚类数 {n_clusters}，无法聚类")

        # 如果只有一个有效特征，跳过标准化和 PCA
        if X_valid.shape[1] == 1:
            self._debug_print("只有一个有效特征，将使用原始数据进行聚类，跳过 PCA")
            X_scaled = X_valid
            pca = None
            X_pca = X_valid
        else:
            X_scaled = self.scaler.fit_transform(X_valid)
            pca_n_components = min(2, X_valid.shape[1])  # 自动适配最大成分
            pca = PCA(n_components=pca_n_components)
            X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(X_scaled)

        original_neuron_ids = features_df['neuron_id'][valid_rows].values

        return {
            'features': features_df,
            'X_scaled': X_scaled,
            'X_pca': X_pca,
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'pca': pca,
            'valid_cols': valid_cols,
            'original_neuron_ids': original_neuron_ids,
            'n_clusters': n_clusters
        }

    def analyze_region(self, region, session_letter, data_root):
        dataset_name = f"{region}_{session_letter}"
        self._debug_print(f"\n{'='*60}")
        self._debug_print(f"开始分析数据集: {dataset_name}")
        self._debug_print(f"数据根目录: {data_root}")

        try:
            # 1. 读取刺激信息
            self._debug_print("\n步骤1: 读取解码CSV文件...")
            tgt_dict, tgt_folder = read_Decoding_csv(
                data_path=data_root,
                region_type=region,
                session_letter=session_letter
            )
            self._debug_print(f"找到目标文件夹: {tgt_folder}")
            self._debug_print(f"刺激类型: {list(tgt_dict.keys())}")

            # 2. 加载神经活动数据
            neural_file_path = os.path.join(data_root, tgt_folder)
            self._debug_print(f"\n步骤2: 在 {neural_file_path} 中查找神经数据...")
            
            npz_files = [f for f in os.listdir(neural_file_path) if f.endswith('.npz')]
            if not npz_files:
                raise FileNotFoundError(f"在 {neural_file_path} 中未找到NPZ文件")
            
            self._debug_print(f"找到NPZ文件: {npz_files}")
            
            # 加载第一个有效的NPZ文件
            for file_name in npz_files:
                try:
                    file_path = os.path.join(neural_file_path, file_name)
                    self._debug_print(f"\n加载文件: {file_path}")
                    
                    ts, dff, all_roi_masks, cids, metadata = load_neural_npz(file_path)
                    
                    if dff is None or len(dff) == 0:
                        self._debug_print(f"文件 {file_name} 中dff数据为空，尝试下一个文件")
                        continue
                        
                    self._debug_print(f"成功加载神经数据: {dff.shape}")
                    break
                except Exception as e:
                    self._debug_print(f"加载文件 {file_name} 失败: {str(e)}")
                    continue
            else:
                raise ValueError("所有NPZ文件都无法加载有效数据")

            # 3. 计算调谐特征
            self._debug_print("\n步骤3: 计算调谐特征...")
            features_df = self.calculate_tuning_features(ts, dff, tgt_dict, fillna_zero=False)
            
            # 检查特征有效性
            zero_ratio = (features_df[features_df.columns[1:]] == 0).mean()
            nan_ratio = features_df[features_df.columns[1:]].isna().mean()
            
            self._debug_print(f"\n【{dataset_name}】各特征列统计:")
            self._debug_print("零值比例:")
            self._debug_print(zero_ratio.to_string())
            self._debug_print("\nNaN值比例:")
            self._debug_print(nan_ratio.to_string())
            
            # 4. 聚类分析
            self._debug_print("\n步骤4: 执行聚类分析...")
            results = self.cluster_neurons(features_df, n_clusters=4)
            
            # 存储结果
            results['region'] = region
            results['session'] = session_letter
            results['n_neurons'] = dff.shape[0]
            self.results[dataset_name] = results

            self._debug_print(f"\n分析完成: {dataset_name}")
            self._debug_print(f"神经元数量: {dff.shape[0]}")
            self._debug_print(f"有效神经元数量: {len(results['original_neuron_ids'])}")
            self._debug_print(f"聚类数量: {results['n_clusters']}")
            
            return results

        except Exception as e:
            error_msg = f"分析 {dataset_name} 失败: {str(e)}"
            self._debug_print(f"\n错误: {error_msg}")
            return {
                'error': error_msg,
                'region': region,
                'session': session_letter,
                'success': False
            }

    def plot_cluster_profiles(self, results, figsize=(12, 8)):
        if 'error' in results:
            print(f"无法绘制图表: {results['error']}")
            return

        if 'original_neuron_ids' not in results or 'clusters' not in results:
            print("缺少必要的聚类结果数据，无法绘制轮廓图")
            return

        features_df = results['features']
        clusters = results['clusters']
        valid_cols = results.get('valid_cols', [])
        original_neuron_ids = results['original_neuron_ids']

        if not valid_cols:
            print("没有有效特征列，跳过雷达图绘制")
            return

        # 确保我们只使用有效神经元的数据
        valid_features_df = features_df.loc[features_df['neuron_id'].isin(original_neuron_ids)]
    
        # 确保聚类标签与数据行数匹配
        if len(clusters) != len(valid_features_df):
            print(f"错误: 聚类标签数量({len(clusters)})与有效神经元数量({len(valid_features_df)})不匹配")
            return

        # 标准化特征以进行可视化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(valid_features_df[valid_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=valid_cols)
    
        # 添加聚类标签到数据框
        scaled_df['cluster'] = clusters

        # 计算每个簇的均值和标准误
        cluster_means = scaled_df.groupby('cluster')[valid_cols].mean()
        cluster_sems = scaled_df.groupby('cluster')[valid_cols].sem()

        angles = np.linspace(0, 2 * np.pi, len(valid_cols), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
        for i in range(len(cluster_means)):
            values = cluster_means.iloc[i].tolist() + [cluster_means.iloc[i][0]]
            errors = cluster_sems.iloc[i].tolist() + [cluster_sems.iloc[i][0]]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
            ax.fill_between(angles, 
                        np.array(values) - np.array(errors), 
                        np.array(values) + np.array(errors), 
                         alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(valid_cols)
        ax.set_title(f"Neuron Cluster Profiles - {results['region']} {results['session']}", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.show()

    def plot_cluster_distributions(self, results, figsize=(15, 10)):
        if 'error' in results:
            print(f"无法绘制图表: {results['error']}")
            return

        features_df = results['features']
        clusters = results['clusters']
        valid_cols = results.get('valid_cols', [])

        if not valid_cols:
            print("没有有效特征列，跳过分布图绘制")
            return

        n_cols = 4
        n_rows = int(np.ceil(len(valid_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel()

        for i, feature in enumerate(valid_cols):
            for cluster in sorted(np.unique(clusters)):
                mask = (clusters == cluster)
                if np.any(mask):
                    data = features_df[feature].iloc[results['original_neuron_ids'][mask]]
                    if len(data) > 1 and np.var(data) > 1e-6:
                        sns.kdeplot(data, ax=axes[i], 
                                   label=f'Cluster {cluster} (n={len(data)})', 
                                   warn_singular=False)
            axes[i].set_title(feature)
            axes[i].legend()

        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Feature Distributions by Cluster - {results['region']} {results['session']}")
        plt.tight_layout()
        plt.show()

    def plot_pca_clusters(self, results, figsize=(8, 6)):
        if 'error' in results:
            print(f"无法绘制图表: {results['error']}")
            return

        X_pca = results['X_pca']
        clusters = results['clusters']
        n_clusters = results['n_clusters']

        plt.figure(figsize=figsize)
        for i in range(n_clusters):
            mask = clusters == i
            n_neurons = np.sum(mask)
            if X_pca.shape[1] > 1:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           label=f'Cluster {i} (n={n_neurons})', 
                           alpha=0.7)
            else:
                plt.scatter(X_pca[mask, 0], np.zeros(n_neurons),
                           label=f'Cluster {i} (n={n_neurons})',
                           alpha=0.7)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2' if X_pca.shape[1] > 1 else '')
        plt.title(f"Neuron Clusters in PCA Space - {results['region']} {results['session']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_cluster_report(self, results):
        if 'error' in results:
            print(f"无法生成报告: {results['error']}")
            return None

        features_df = results['features']
        clusters = results['clusters']
        n_clusters = results['n_clusters']
        valid_cols = results.get('valid_cols', [])
        original_neuron_ids = results.get('original_neuron_ids', [])

        report = []

        for cluster in range(n_clusters):
            cluster_mask = clusters == cluster
            cluster_data = features_df.loc[features_df['neuron_id'].isin(original_neuron_ids[cluster_mask])]
            
            if len(cluster_data) == 0:
                continue
                
            cluster_stats = {
                'Cluster': cluster,
                'N_Neurons': len(cluster_data),
                'Mean_DSI': cluster_data['dsi'].mean(),
                'Std_DSI': cluster_data['dsi'].std(),
                'Mean_OSI': cluster_data['osi'].mean(),
                'Std_OSI': cluster_data['osi'].std(),
                'Mean_SF_Pref': cluster_data['sf_pref'].mean(),
                'Std_SF_Pref': cluster_data['sf_pref'].std(),
                'Mean_TF_Pref': cluster_data['tf_pref'].mean(),
                'Std_TF_Pref': cluster_data['tf_pref'].std(),
                'Mean_Natural_Resp': cluster_data['natural_response'].mean(),
                'Std_Natural_Resp': cluster_data['natural_response'].std(),
                'Mean_Movie_Resp': cluster_data['movie_response'].mean(),
                'Std_Movie_Resp': cluster_data['movie_response'].std(),
                'Mean_Sparseness': cluster_data['lifetime_sparseness'].mean(),
                'Std_Sparseness': cluster_data['lifetime_sparseness'].std(),
                'Mean_Reliability': cluster_data['reliability'].mean(),
                'Std_Reliability': cluster_data['reliability'].std()
            }
            report.append(cluster_stats)

        report_df = pd.DataFrame(report)
        print(f"\n=== Cluster Report for {results['region']} {results['session']} ===")
        print(f"Total Neurons: {results['n_neurons']}")
        print(f"Valid Neurons: {len(original_neuron_ids)}")
        print(f"Valid Features: {valid_cols}")
        print("\nCluster Statistics:")
        print(report_df.to_string(index=False))
        return report_df


if __name__ == "__main__":
    # 配置分析参数
    target_brain_regions = ['VISp', 'VISal', 'VISl']
    session_letters = ['A', 'B']
    data_root = r'e:\Users\asus\Desktop\PBL3\SCNSProject-main\data'
    
    # 创建分析器实例（启用调试模式）
    analyzer = NeuronClusterAnalysis(debug=True)
    
    print("\n开始神经元集群分析...")
    print("=" * 60)

    # 执行分析
    for region in target_brain_regions:
        for session_letter in session_letters:
            dataset_name = f"{region}_{session_letter}"
            print(f"\n分析数据集: {dataset_name}")
            
            # 执行分析
            results = analyzer.analyze_region(region, session_letter, data_root)
            
            # 如果分析成功，生成图表和报告
            if results and 'error' not in results:
                analyzer.plot_cluster_profiles(results)
                analyzer.plot_cluster_distributions(results)
                analyzer.plot_pca_clusters(results)
                analyzer.generate_cluster_report(results)
            else:
                print(f"跳过 {dataset_name} 的可视化 - 分析失败")

    print("\n所有分析完成!")