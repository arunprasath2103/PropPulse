import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from numpy.testing import assert_array_almost_equal
from backend.utils.data_pipeline import CMAPSSDataPipeline

@pytest.fixture
def pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    return CMAPSSDataPipeline(raw_data_dir=str(raw_dir), processed_data_dir=str(processed_dir))

def test_rul_computation_and_clipping(pipeline):
    subset = 'FD001'
    train_data = []
    # 1 unit, 150 cycles
    for cycle in range(1, 151):
        train_data.append([1, cycle, 0, 0, 0] + [0.5]*21)
    
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(os.path.join(pipeline.raw_data_dir, f'train_{subset}.txt'), sep=' ', index=False, header=False)
    pd.DataFrame().to_csv(os.path.join(pipeline.raw_data_dir, f'test_{subset}.txt'), sep=' ', index=False, header=False)
    pd.DataFrame([0]).to_csv(os.path.join(pipeline.raw_data_dir, f'RUL_{subset}.txt'), sep=' ', index=False, header=False)
    
    train_df_loaded, _ = pipeline._load_and_compute_rul(subset)
    
    # Standard Behavior: Reverse countdown
    assert train_df_loaded.loc[train_df_loaded['cycle'] == 1, 'RUL'].values[0] == 149
    assert train_df_loaded.loc[train_df_loaded['cycle'] == 150, 'RUL'].values[0] == 0
    
    # Edge Case - Clipping
    train_df_loaded['RUL'] = np.clip(train_df_loaded['RUL'], 0, pipeline.max_rul) / pipeline.max_rul
    
    # >= 125 clips to exactly 1.0 (since 149 > 125)
    assert train_df_loaded.loc[train_df_loaded['cycle'] == 1, 'RUL'].values[0] == 1.0
    # 150 - 25 = 125 -> exactly 1.0
    assert train_df_loaded.loc[train_df_loaded['cycle'] == 25, 'RUL'].values[0] == 1.0
    # 0 normalizes to 0.0
    assert train_df_loaded.loc[train_df_loaded['cycle'] == 150, 'RUL'].values[0] == 0.0

def test_normalization_and_leakage(pipeline):
    train_df = pd.DataFrame({
        'unit': [1, 1], 'cycle': [1, 2],
        's2': [10.0, 20.0]
    })
    test_df = pd.DataFrame({
        'unit': [2, 2], 'cycle': [1, 2],
        's2': [100.0, 200.0]
    })
    
    for ch in pipeline.sensor_channels:
        if ch != 's2':
            train_df[ch] = 0.0
            test_df[ch] = 0.0
            
    # Edge Case - Zero Variance (s3 is all 5.0)
    train_df['s3'] = 5.0
    test_df['s3'] = 5.0
    
    train_norm, test_norm = pipeline._normalize_sensors(train_df, test_df)
    
    # Standard Behavior: z-score normalization
    # mean=15.0, std=5.0 for s2 in train
    assert_array_almost_equal(train_norm['s2'].values, [-1.0, 1.0])
    
    # Edge Case - Strict Isolation: test set uses train mean/std
    assert_array_almost_equal(test_norm['s2'].values, [17.0, 37.0]) # (100-15)/5, (200-15)/5
    
    # Edge Case - Zero Variance safety
    assert_array_almost_equal(train_norm['s3'].values, [0.0, 0.0])
    assert_array_almost_equal(test_norm['s3'].values, [0.0, 0.0])

def test_sliding_window_generation(pipeline):
    df_data = []
    # Standard: 50 cycles (unit 1)
    for c in range(1, 51):
        row = {'unit': 1, 'cycle': c, 'RUL': 50 - c}
        for ch in pipeline.sensor_channels:
            row[ch] = 0.0
        df_data.append(row)
        
    # Edge Case - Short Trajectories: 29 cycles (unit 2)
    for c in range(1, 30):
        row = {'unit': 2, 'cycle': c, 'RUL': 29 - c}
        for ch in pipeline.sensor_channels:
            row[ch] = 0.0
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    
    clients_data = {
        1: {
            'FD001': {
                'train': df,
                'test': pd.DataFrame()
            }
        }
    }
    
    # Only 2 units, 30% of 2 is 0 -> no fault injection natively
    pipeline._process_and_save(clients_data)
        
    windows = np.load(os.path.join(pipeline.processed_data_dir, 'client_1_subset_FD001_windows.npy'))
    labels = np.load(os.path.join(pipeline.processed_data_dir, 'client_1_subset_FD001_labels.npy'))
    
    # Shape verification and discarded short trajectory check
    assert windows.shape == (5, 30, 12)
    assert len(labels) == 5
    
    # Edge Case - Label Alignment
    # Window 0: cycles 1-30. Last cycle is 30. RUL = 50 - 30 = 20.
    assert labels[0] == 20
    # Window 1: cycles 6-35. Last cycle is 35. RUL = 50 - 35 = 15.
    assert labels[1] == 15

def test_early_termination_augmentation(pipeline):
    df_data = []
    for u in range(1, 101):
        for c in range(1, 41):
            df_data.append({'subset': 'FD001', 'unit': u, 'cycle': c})
            
    df = pd.DataFrame(df_data)
    
    aug_df = pipeline._early_termination_augmentation(df)
    unit_counts = aug_df.groupby('unit').size()
    
    truncated_units = unit_counts[unit_counts < 40]
    
    # Standard Behavior: 20% truncated
    assert len(truncated_units) == 20
    
    # Edge Case - Minimum Length Safety: between 60% and 90%
    assert truncated_units.min() >= 24
    assert truncated_units.max() <= 36

def test_client_partitioning(pipeline):
    df_data = []
    # Edge Case - Indivisible Groups: 101 engines
    for u in range(1, 102):
        df_data.append({'subset': 'FD001', 'unit': u, 'cycle': 1})
        
    train_df = pd.DataFrame(df_data)
    test_df = pd.DataFrame(columns=['subset', 'unit', 'cycle'])
    
    clients_data = pipeline._client_partitioning(train_df, test_df)
    
    sizes = []
    for i in range(1, 5):
        client_train = clients_data[i]['FD001']['train']
        sizes.append(len(client_train['unit'].unique()))
        
    assert sum(sizes) == 101
    assert set(sizes) == {26, 25}

@patch('numpy.random.choice')
def test_fault_injection_logic(mock_choice, pipeline):
    def mock_choice_side_effect(a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a_arr = np.arange(a)
        else:
            a_arr = np.array(a)
            
        if size is not None:
            return a_arr[:size]
            
        a_list = list(a_arr)
        if 'Stuck-at' in a_list: return 'Stuck-at'
        if 's2' in a_list: return 's2'
        return a_list[0]
        
    mock_choice.side_effect = mock_choice_side_effect
    
    df_data = []
    # Ensure enough units so 30% is at least 1 unit. 4 units -> int(1.2) = 1
    for u in range(1, 5):
        for c in range(1, 51):
            row = {'unit': u, 'cycle': c, 'RUL': 50 - c}
            for ch in pipeline.sensor_channels:
                row[ch] = float(c) # Mock changing signal
            df_data.append(row)
        
    df = pd.DataFrame(df_data)
    
    clients_data = {
        1: {
            'FD001': {
                'train': df,
                'test': pd.DataFrame()
            }
        }
    }
    
    # Fault start at cycle 29 (50 * 0.59 = 29.5 -> int is 29)
    with patch('numpy.random.uniform', return_value=0.59):
        pipeline._process_and_save(clients_data)
        
    fault_windows = np.load(os.path.join(pipeline.processed_data_dir, 'client_1_fault_windows.npy'), allow_pickle=True)
    fault_labels = np.load(os.path.join(pipeline.processed_data_dir, 'client_1_fault_labels.npy'), allow_pickle=True)
    
    # Edge Case - Mixed Window Labeling
    assert len(fault_windows) == 5
    assert list(fault_labels[0]) == [1, 'Stuck-at']
    
    # Edge Case - Exact Boundary Faults
    s2_idx = pipeline.sensor_channels.index('s2')
    
    w0_s2 = fault_windows[0][:, s2_idx]
    assert w0_s2[27] == 28.0 # Cycle 28 (Healthy)
    assert w0_s2[28] == 29.0 # Cycle 29 (Fault Started)
    assert w0_s2[29] == 29.0 # Cycle 30 (Stuck)
    
    w4_s2 = fault_windows[-1][:, s2_idx] # Cycles 21-50
    assert w4_s2[-1] == 29.0 # Cycle 50 (Still stuck)
