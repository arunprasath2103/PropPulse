import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DIR = os.path.abspath(os.path.join(UTILS_DIR, '..', 'data', 'raw'))
DEFAULT_PROCESSED_DIR = os.path.abspath(os.path.join(UTILS_DIR, '..', 'data', 'processed'))

class CMAPSSDataPipeline:
    def __init__(self, raw_data_dir=DEFAULT_RAW_DIR, processed_data_dir=DEFAULT_PROCESSED_DIR):
        """
        Initialize the data pipeline for CMAPSS dataset.
        
        Args:
            raw_data_dir (str): Path to raw data directory.
            processed_data_dir (str): Path to processed data directory.
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.subsets = ['FD001', 'FD002', 'FD003', 'FD004']
        self.sensor_channels = ['s2', 's3', 's4', 's7', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20']
        
        self.columns = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
        self.keep_columns = ['unit', 'cycle'] + self.sensor_channels
        
        self.window_size = 30
        self.stride = 5
        self.max_rul = 125
        self.seed = 42
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
    def execute(self):
        """
        Executes the complete data preprocessing pipeline.
        """
        print("Starting data pipeline execution...")
        np.random.seed(self.seed)
        
        train_dfs = []
        test_dfs = []
        
        # Step 1 & 2: Load Raw Data & Compute RUL
        print("Loading raw data and computing RUL...")
        for subset in self.subsets:
            train_df, test_df = self._load_and_compute_rul(subset)
            if not train_df.empty:
                train_df['subset'] = subset
                test_df['subset'] = subset
                train_dfs.append(train_df)
                test_dfs.append(test_df)
                
        if not train_dfs:
            print(f"Warning: No raw data found in {self.raw_data_dir}. Please place the CMAPSS text files there.")
            return
            
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # Step 3: Clip and Normalize RUL
        print("Clipping and normalizing RUL...")
        train_df['RUL'] = np.clip(train_df['RUL'], 0, self.max_rul) / self.max_rul
        test_df['RUL'] = np.clip(test_df['RUL'], 0, self.max_rul) / self.max_rul
        
        # Step 4: Normalize Sensor Channels
        print("Normalizing sensor channels...")
        train_df, test_df = self._normalize_sensors(train_df, test_df)
        
        # Step 5: Early Termination Augmentation
        print("Applying early termination augmentation...")
        train_df = self._early_termination_augmentation(train_df)
        
        # Step 7: Client Partitioning
        print("Partitioning data into clients...")
        clients_data = self._client_partitioning(train_df, test_df)
        
        # Step 8, 6 & 9: Fault Injection, Sliding Window, and Saving Processed Data
        print("Applying fault injection, generating sliding windows, and saving processed data...")
        self._process_and_save(clients_data)
        
        print(f"Pipeline complete. Processed data saved to {self.processed_data_dir}")

    def _load_and_compute_rul(self, subset):
        train_file = os.path.join(self.raw_data_dir, f'train_{subset}.txt')
        test_file = os.path.join(self.raw_data_dir, f'test_{subset}.txt')
        rul_file = os.path.join(self.raw_data_dir, f'RUL_{subset}.txt')
        
        if not os.path.exists(train_file):
            return pd.DataFrame(), pd.DataFrame()
            
        train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=self.columns)[self.keep_columns]
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=self.columns)[self.keep_columns]
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        # Compute Train RUL: max cycle - current cycle
        rul_train = pd.DataFrame(train_df.groupby('unit')['cycle'].max()).reset_index()
        rul_train.columns = ['unit', 'max_cycle']
        train_df = train_df.merge(rul_train, on=['unit'], how='left')
        train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
        train_df.drop(columns=['max_cycle'], inplace=True)
        
        # Compute Test RUL: remaining cycles from RUL text + (max cycle in test sequence - current cycle)
        rul_test = pd.DataFrame(test_df.groupby('unit')['cycle'].max()).reset_index()
        rul_test.columns = ['unit', 'max_cycle']
        rul_test['RUL_actual'] = rul_df['RUL'].values
        test_df = test_df.merge(rul_test, on=['unit'], how='left')
        test_df['RUL'] = test_df['max_cycle'] - test_df['cycle'] + test_df['RUL_actual']
        test_df.drop(columns=['max_cycle', 'RUL_actual'], inplace=True)
        
        return train_df, test_df
        
    def _normalize_sensors(self, train_df, test_df):
        scaler = StandardScaler()
        # Compute global mean and std using only the training set
        train_df[self.sensor_channels] = scaler.fit_transform(train_df[self.sensor_channels])
        test_df[self.sensor_channels] = scaler.transform(test_df[self.sensor_channels])
        
        # Save normalization parameters
        params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'channels': self.sensor_channels
        }
        with open(os.path.join(self.processed_data_dir, 'normalisation_params.json'), 'w') as f:
            json.dump(params, f, indent=4)
            
        return train_df, test_df
        
    def _early_termination_augmentation(self, df):
        np.random.seed(self.seed)
        
        grouped = df.groupby(['subset', 'unit'])
        units = list(grouped.groups.keys())
        
        # Randomly select 20% of training engine trajectories
        n_augment = int(0.2 * len(units))
        augment_indices = np.random.choice(len(units), n_augment, replace=False)
        augment_set = set([units[i] for i in augment_indices])
        
        new_dfs = []
        for name, group in grouped:
            if name in augment_set:
                # Truncate at random point between 60% and 90%
                trunc_pct = np.random.uniform(0.6, 0.9)
                max_cycle = group['cycle'].max()
                trunc_cycle = int(max_cycle * trunc_pct)
                group = group[group['cycle'] <= trunc_cycle]
            new_dfs.append(group)
            
        return pd.concat(new_dfs, ignore_index=True)
        
    def _client_partitioning(self, train_df, test_df):
        np.random.seed(self.seed)
        clients_data = {i: {} for i in range(1, 5)}
        
        for subset in self.subsets:
            train_sub = train_df[train_df['subset'] == subset]
            test_sub = test_df[test_df['subset'] == subset]
            
            units = train_sub['unit'].unique()
            np.random.shuffle(units)
            # Divide into 4 equal groups
            splits = np.array_split(units, 4)
            
            for i, split in enumerate(splits):
                client_id = i + 1
                clients_data[client_id][subset] = {
                    'train': train_sub[train_sub['unit'].isin(split)],
                    'test': test_sub[test_sub['unit'].isin(split)]
                }
                
        return clients_data

    def _process_and_save(self, clients_data):
        np.random.seed(self.seed)
        fault_types = ['Bias', 'Drift', 'Spike', 'Stuck-at', 'Noise']
        
        for client_id, subset_dict in clients_data.items():
            client_fault_windows = []
            client_fault_labels = []
            
            for subset, data_splits in subset_dict.items():
                subset_windows = []
                subset_labels = []
                
                for split_name, df in data_splits.items():
                    if df.empty: continue
                    df = df.copy()
                    
                    units = df['unit'].unique()
                    # Select 30% of engine units per client
                    n_faults = int(0.3 * len(units))
                    fault_units = np.random.choice(units, n_faults, replace=False)
                    
                    df['fault_flag'] = 0
                    df['fault_type'] = 'None'
                    
                    for unit in fault_units:
                        unit_mask = df['unit'] == unit
                        total_cycles = df[unit_mask]['cycle'].max()
                        # Apply fault starting randomly between 20% and 60% of total life
                        fault_start = int(total_cycles * np.random.uniform(0.2, 0.6))
                        
                        channel = np.random.choice(self.sensor_channels)
                        fault_type = np.random.choice(fault_types)
                        
                        fault_mask = unit_mask & (df['cycle'] >= fault_start)
                        df.loc[fault_mask, 'fault_flag'] = 1
                        df.loc[fault_mask, 'fault_type'] = fault_type
                        
                        if fault_type == 'Bias':
                            df.loc[fault_mask, channel] += 2.0
                        elif fault_type == 'Drift':
                            n_cycles = fault_mask.sum()
                            df.loc[fault_mask, channel] += np.linspace(0, 3.0, n_cycles)
                        elif fault_type == 'Spike':
                            spike_mask = fault_mask & (np.random.rand(len(df)) < 0.1)
                            df.loc[spike_mask, channel] *= 3.0
                        elif fault_type == 'Stuck-at':
                            val_idx = df[unit_mask & (df['cycle'] == fault_start)].index
                            if len(val_idx) > 0:
                                val = df.loc[val_idx[0], channel]
                                df.loc[fault_mask, channel] = val
                        elif fault_type == 'Noise':
                            n_cycles = fault_mask.sum()
                            df.loc[fault_mask, channel] += np.random.normal(0, 1.0, n_cycles)
                            
                    for unit in units:
                        u_df = df[df['unit'] == unit].sort_values('cycle')
                        # Discard engines with < 30 cycles
                        if len(u_df) < self.window_size:
                            continue
                            
                        sensor_data = u_df[self.sensor_channels].values
                        rul_data = u_df['RUL'].values
                        ff_data = u_df['fault_flag'].values
                        ft_data = u_df['fault_type'].values
                        
                        # Overlapping windows of 30 consecutive cycles with stride of 5
                        for start_idx in range(0, len(u_df) - self.window_size + 1, self.stride):
                            end_idx = start_idx + self.window_size
                            w = sensor_data[start_idx:end_idx]
                            # Label = RUL at the last cycle of the window
                            l = rul_data[end_idx - 1]
                            
                            subset_windows.append(w)
                            subset_labels.append(l)
                            
                            # Identify if the window contains any fault cycles
                            if np.any(ff_data[start_idx:end_idx] == 1):
                                types_in_window = ft_data[start_idx:end_idx]
                                valid_types = types_in_window[types_in_window != 'None']
                                ft = valid_types[0] if len(valid_types) > 0 else 'None'
                                client_fault_windows.append(w)
                                # Binary fault flag (1) and fault type string
                                client_fault_labels.append([1, ft])
                                
                if subset_windows:
                    np.save(os.path.join(self.processed_data_dir, f'client_{client_id}_subset_{subset}_windows.npy'), np.array(subset_windows))
                    np.save(os.path.join(self.processed_data_dir, f'client_{client_id}_subset_{subset}_labels.npy'), np.array(subset_labels))
                    
            if client_fault_windows:
                np.save(os.path.join(self.processed_data_dir, f'client_{client_id}_fault_windows.npy'), np.array(client_fault_windows))
                np.save(os.path.join(self.processed_data_dir, f'client_{client_id}_fault_labels.npy'), np.array(client_fault_labels, dtype=object))

if __name__ == "__main__":
    pipeline = CMAPSSDataPipeline()
    pipeline.execute()
