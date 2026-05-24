import math
import warnings
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from utils.configs import configs

warnings.filterwarnings('ignore')

class Data(Dataset):
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        '''
        wrapper for Sensor dataset

        - number of variables: 9 (Value_0 ~ Value_8)
        - number of groups: 141
        - time unit: milliseconds
        - irregular time series
        '''
        logger.debug(f"getting {flag} set of Sensor dataset")
        self.configs = configs
        assert flag in ['train', 'test', 'val', 'test_all']
        self.flag = flag

        self.seq_len = configs.seq_len  # in milliseconds for Sensor dataset
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  # in milliseconds for Sensor dataset

        self.dataset_root_path = configs.dataset_root_path

        # Absolute time window parameters (in milliseconds)
        self.seq_len_ms = getattr(configs, 'seq_len_ms', 15000)  # 15 seconds
        self.pred_len_ms = getattr(configs, 'pred_len_ms', 1500)  # 1.5 seconds
        self.sliding_stride_ms = getattr(configs, 'sliding_stride_ms', 3000)  # 3 seconds
        self.min_data_points = getattr(configs, 'min_data_points', 3)  # minimum data points per window
        self.min_group_duration_ms = getattr(configs, 'min_group_duration_ms', 16500)  # seq_len_ms + pred_len_ms

        self.preprocess()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def preprocess(self):
        # Load CSV file based on flag
        if self.flag == 'train':
            csv_file = f"{self.dataset_root_path}/train_norm.csv"
        elif self.flag == 'val':
            csv_file = f"{self.dataset_root_path}/val_norm.csv"
        elif self.flag == 'test':
            csv_file = f"{self.dataset_root_path}/test_norm.csv"
        elif self.flag == 'test_all':
            # For test_all, we'll combine all datasets
            train_data = self._load_and_process_csv(f"{self.dataset_root_path}/train_norm.csv")
            val_data = self._load_and_process_csv(f"{self.dataset_root_path}/val_norm.csv")
            test_data = self._load_and_process_csv(f"{self.dataset_root_path}/test_norm.csv")
            self.data = train_data + val_data + test_data
            return
        else:
            raise ValueError(f"Invalid flag: {self.flag}")

        self.data = self._load_and_process_csv(csv_file)

        # For validation and test sets, we don't need to compute max lengths
        # They should be inherited from training set
        if self.flag == "train":
            # Determine max sequence lengths for irregular time series
            # This is needed for padding in collate functions
            test_all_data = self.data  # For simplicity, use current data
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            self.patch_len_max_irr = 0

            SEQ_LEN = self.configs.seq_len
            PRED_LEN = self.configs.pred_len

            for sample in test_all_data:
                if sample["x"].shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = sample["x"].shape[0]
                if sample["y"].shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = sample["y"].shape[0]

                if self.configs.collate_fn == "collate_fn_patch":
                    PATCH_LEN = self.configs.patch_len
                    assert SEQ_LEN % PATCH_LEN == 0, f"seq_len {SEQ_LEN} should be divisible by patch_len {PATCH_LEN}"
                    n_patch: int = SEQ_LEN // PATCH_LEN
                    n_patch_y: int = math.ceil(self.configs.pred_len / PATCH_LEN)

                    patch_i_end_previous = 0
                    for i in range(n_patch):
                        observations = sample["x_mark"] < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                        patch_i_end = observations.sum()
                        sample_mask = slice(patch_i_end_previous, patch_i_end)
                        x_patch_i = sample["x"][sample_mask]
                        if len(x_patch_i) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(x_patch_i)

                        patch_i_end_previous = patch_i_end

                    patch_j_end_previous = 0
                    for j in range(n_patch_y):
                        observations = sample["y_mark"] < (((n_patch + j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
                        patch_j_end = observations.sum()
                        sample_mask = slice(patch_j_end_previous, patch_j_end)
                        y_patch_j = sample["y"][sample_mask]
                        if len(y_patch_j) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(y_patch_j)

                        patch_j_end_previous = patch_j_end

            if self.configs.collate_fn == "collate_fn_patch":
                n_patch: int = SEQ_LEN // PATCH_LEN
                n_patch_y: int = math.ceil(self.configs.pred_len / PATCH_LEN)
                self.seq_len_max_irr = max(self.seq_len_max_irr, self.patch_len_max_irr * n_patch)
                self.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)

            # Create new fields in global configs to pass information to models
            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                self.configs.patch_len_max_irr = self.patch_len_max_irr
                logger.debug(f"{self.configs.patch_len_max_irr=}")
            logger.debug(f"{self.configs.seq_len_max_irr=}")
            logger.debug(f"{self.configs.pred_len_max_irr=}")
        else:
            # For validation and test sets, check if max lengths are already set
            # If not, we need to compute them (this should happen when validation/test is loaded before training)
            if not hasattr(self.configs, 'seq_len_max_irr') or self.configs.seq_len_max_irr is None:
                logger.warning(f"seq_len_max_irr not set for {self.flag} set. Computing from {self.flag} data...")
                self.seq_len_max_irr = 0
                self.pred_len_max_irr = 0
                self.patch_len_max_irr = 0
                
                SEQ_LEN = self.configs.seq_len
                PRED_LEN = self.configs.pred_len
                
                for sample in self.data:
                    if sample["x"].shape[0] > self.seq_len_max_irr:
                        self.seq_len_max_irr = sample["x"].shape[0]
                    if sample["y"].shape[0] > self.pred_len_max_irr:
                        self.pred_len_max_irr = sample["y"].shape[0]
                    
                    if self.configs.collate_fn == "collate_fn_patch":
                        PATCH_LEN = getattr(self.configs, 'patch_len', None)
                        if PATCH_LEN is None:
                            continue
                        if SEQ_LEN % PATCH_LEN != 0:
                            continue
                        n_patch: int = SEQ_LEN // PATCH_LEN
                        n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)
                        
                        patch_i_end_previous = 0
                        for i in range(n_patch):
                            observations = sample["x_mark"] < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                            patch_i_end = observations.sum()
                            sample_mask = slice(patch_i_end_previous, patch_i_end)
                            x_patch_i = sample["x"][sample_mask]
                            if len(x_patch_i) > self.patch_len_max_irr:
                                self.patch_len_max_irr = len(x_patch_i)
                            patch_i_end_previous = patch_i_end
                        
                        patch_j_end_previous = 0
                        for j in range(n_patch_y):
                            observations = sample["y_mark"] < (((n_patch + j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
                            patch_j_end = observations.sum()
                            sample_mask = slice(patch_j_end_previous, patch_j_end)
                            y_patch_j = sample["y"][sample_mask]
                            if len(y_patch_j) > self.patch_len_max_irr:
                                self.patch_len_max_irr = len(y_patch_j)
                            patch_j_end_previous = patch_j_end
                
                self.configs.seq_len_max_irr = self.seq_len_max_irr
                self.configs.pred_len_max_irr = self.pred_len_max_irr
                if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                    self.configs.patch_len_max_irr = self.patch_len_max_irr
                    logger.debug(f"{self.configs.patch_len_max_irr=}")
                logger.debug(f"{self.configs.seq_len_max_irr=}")
                logger.debug(f"{self.configs.pred_len_max_irr=}")

    def _load_and_process_csv(self, csv_file):
        """Load CSV file and generate sliding windows"""
        logger.debug(f"Loading Sensor data from {csv_file}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Extract value columns and mask columns
        value_cols = [f'Value_{i}' for i in range(9)]
        mask_cols = [f'Mask_{i}' for i in range(9)]
        
        # Group by ID
        grouped = df.groupby('ID')
        
        all_windows = []
        sample_ID = 0
        
        for group_id, group_df in grouped:
            # Sort by Time
            group_df = group_df.sort_values('Time')
            
            # Get time values and data
            times = group_df['Time'].values.astype(np.float32)
            values = group_df[value_cols].values.astype(np.float32)
            masks = group_df[mask_cols].values.astype(np.float32)
            
            # Convert to torch tensors
            times_tensor = torch.from_numpy(times)
            values_tensor = torch.from_numpy(values)
            masks_tensor = torch.from_numpy(masks)
            
            # Generate sliding windows
            group_windows = self._generate_sliding_windows(
                group_id, times_tensor, values_tensor, masks_tensor, sample_ID
            )
            
            all_windows.extend(group_windows)
            sample_ID += len(group_windows)
        
        logger.debug(f"Generated {len(all_windows)} windows from {csv_file}")
        return all_windows

    def _generate_sliding_windows(self, group_id, times, values, masks, start_sample_id):
        """Generate sliding windows for a single group"""
        windows = []
        
        if len(times) == 0:
            return windows
        
        # Get time range for this group
        t_min = times.min().item()
        t_max = times.max().item()
        
        # Check if group duration is sufficient
        if t_max - t_min < self.min_group_duration_ms:
            logger.debug(f"Group {group_id} duration {t_max - t_min}ms < {self.min_group_duration_ms}ms, skipping")
            return windows
        
        # Generate sliding windows
        window_start = t_min
        sample_id = start_sample_id
        
        while window_start + self.seq_len_ms + self.pred_len_ms <= t_max:
            # Define window boundaries
            history_end = window_start + self.seq_len_ms
            pred_end = history_end + self.pred_len_ms
            
            # Find indices for history window
            hist_mask = (times >= window_start) & (times < history_end)
            hist_indices = torch.where(hist_mask)[0]
            
            # Find indices for prediction window
            pred_mask = (times >= history_end) & (times < pred_end)
            pred_indices = torch.where(pred_mask)[0]
            
            # Apply minimum data points requirement (different for train/val)
            min_points = self.min_data_points if self.flag == 'train' else 1
            
            if len(hist_indices) >= min_points:
                # Extract data for this window
                hist_times = times[hist_indices]
                hist_values = values[hist_indices]
                hist_masks = masks[hist_indices]
                
                pred_times = times[pred_indices] if len(pred_indices) > 0 else torch.empty(0)
                pred_values = values[pred_indices] if len(pred_indices) > 0 else torch.empty(0, values.shape[1])
                pred_masks = masks[pred_indices] if len(pred_indices) > 0 else torch.empty(0, masks.shape[1])
                
                # Normalize time marks to [0, 1] range
                window_duration = self.seq_len_ms + self.pred_len_ms
                
                if len(hist_times) > 0:
                    hist_time_marks = (hist_times - window_start) / window_duration
                else:
                    hist_time_marks = torch.empty(0)
                
                if len(pred_times) > 0:
                    pred_time_marks = (pred_times - window_start) / window_duration
                else:
                    pred_time_marks = torch.empty(0)
                
                # Create sample dictionary with original time information
                sample = {
                    "sample_ID": sample_id,
                    "x_mark": hist_time_marks,
                    "y_mark": pred_time_marks,
                    "x": hist_values,
                    "y": pred_values,
                    "x_mask": hist_masks,
                    "y_mask": pred_masks,
                    "window_start_ms": window_start,          # Original window start time in milliseconds
                    "window_duration_ms": window_duration,    # Total window duration in milliseconds
                }
                
                windows.append(sample)
                sample_id += 1
            
            # Slide window
            window_start += self.sliding_stride_ms
        
        return windows

def fix_nan_x_mark(x_mark, seq_len):
    # For Sensor dataset, we use actual time marks, so no need to fix NaN
    # Just ensure no NaN values
    nan_mask = torch.isnan(x_mark)
    if nan_mask.any():
        # Replace NaN with appropriate values (e.g., linear interpolation of time marks)
        BATCH_SIZE, SEQ_LEN_MAX_IRR, _ = x_mark.shape
        indices = torch.linspace(start=0, end=seq_len/(seq_len + configs.pred_len), steps=SEQ_LEN_MAX_IRR).to(x_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)
        x_mark[nan_mask] = indices[nan_mask]
    
    return x_mark

def fix_nan_y_mark(y_mark):
    # For Sensor dataset, we use actual time marks
    nan_mask = torch.isnan(y_mark)
    if nan_mask.any():
        BATCH_SIZE, PRED_LEN, _ = y_mark.shape
        indices = torch.linspace(start=configs.seq_len/(configs.seq_len + configs.pred_len), end=1, steps=PRED_LEN).to(y_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)
        y_mark[nan_mask] = indices[nan_mask]
    
    return y_mark

def collate_fn(
    batch: list[dict[str,Tensor]],
) -> dict[str,Tensor]:
    '''
    time-aligned padding for Sensor dataset
    '''
    global configs
    seq_len_max_irr: int = configs.seq_len_max_irr
    pred_len_max_irr: int = configs.pred_len_max_irr

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []
    window_starts: list[float] = []
    window_durations: list[float] = []

    for sample in batch:
        x_mark = sample["x_mark"]
        x = sample["x"]
        y_mark = sample["y_mark"]
        y = sample["y"]

        x_mask = sample["x_mask"]
        y_mask = sample["y_mask"]
        sample_ID = sample["sample_ID"]
        window_start = sample.get("window_start_ms", 0.0)
        window_duration = sample.get("window_duration_ms", 16500.0)  # default: seq_len_ms + pred_len_ms

        xs.append(x)
        x_marks.append(x_mark)
        x_masks.append(x_mask)

        ys.append(y)
        y_marks.append(y_mark)
        y_masks.append(y_mask)

        sample_IDs.append(sample_ID)
        window_starts.append(window_start)
        window_durations.append(window_duration)

    ENC_IN = xs[0].shape[-1]

    # To ensure padding to n_observations_max, we manually append a sample with desired shape then removed.
    xs.append(torch.zeros(seq_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(seq_len_max_irr))
    x_masks.append(torch.zeros(seq_len_max_irr, ENC_IN))
    ys.append(torch.zeros(pred_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(pred_len_max_irr))
    y_masks.append(torch.zeros(pred_len_max_irr, ENC_IN))

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True, padding_value=float("nan"))
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True, padding_value=float("nan"))
    y_masks=pad_sequence(y_masks, batch_first=True)

    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    ys = ys[:-1]
    y_marks = y_marks[:-1]
    y_masks = y_masks[:-1]

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # Manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

    # Convert window starts and durations to tensors
    window_starts_tensor = torch.tensor(window_starts).float()
    window_durations_tensor = torch.tensor(window_durations).float()
    
    return {
        "x": torch.nan_to_num(xs),
        "x_mark": fix_nan_x_mark(x_marks.unsqueeze(-1), seq_len=configs.seq_len).float(),
        "x_mask": x_masks.float(),
        "y": torch.nan_to_num(ys),
        "y_mark": fix_nan_y_mark(y_marks.unsqueeze(-1)).float(),
        "y_mask": y_masks.float(),
        "sample_ID": sample_IDs,
        "window_start_ms": window_starts_tensor,
        "window_duration_ms": window_durations_tensor
    }

def collate_fn_patch(
    batch: list[dict[str,Tensor]],
) -> dict[str,Tensor]:
    '''
    Patch version for Sensor dataset
    '''
    global configs
    seq_len_max_irr: int = configs.seq_len_max_irr
    pred_len_max_irr: int = max(configs.pred_len_max_irr, configs.patch_len_max_irr)
    # actual patch length can be smaller or even greater than configs.patch_len, depending on the actual sampling rate of the irregular time series
    # because configs.patch_len is describing number of time units (e.g., milliseconds), but patch_len_max_irr is describing number of actual observations
    patch_len_max_irr: int = configs.patch_len_max_irr

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    PATCH_LEN = configs.patch_len
    SEQ_LEN = configs.seq_len
    PRED_LEN = configs.pred_len
    assert SEQ_LEN % PATCH_LEN == 0, f"seq_len {SEQ_LEN} should be divisible by patch_len {PATCH_LEN}"
    n_patch: int = SEQ_LEN // PATCH_LEN
    n_patch_y: int = math.ceil(configs.pred_len / PATCH_LEN)

    for sample in batch:
        x_mark = sample["x_mark"]
        x = sample["x"]
        y_mark = sample["y_mark"]
        y = sample["y"]

        x_mask = sample["x_mask"]
        y_mask = sample["y_mask"]
        sample_ID = sample["sample_ID"]

        patch_i_end_previous = 0

        for i in range(n_patch):
            observations = x_mark < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
            patch_i_end = observations.sum()
            sample_mask = slice(patch_i_end_previous, patch_i_end)
            x_patch_i = x[sample_mask]
            if len(x_patch_i) == 0:
                xs.append(torch.full((1, x.shape[-1]), fill_value=float("nan"), device=x.device))
                x_marks.append(torch.zeros((1), device=x.device))
                x_masks.append(torch.zeros((1, x.shape[-1]), device=x.device))
            else:
                xs.append(x_patch_i)
                x_marks.append(x_mark[sample_mask])
                x_masks.append(x_mask[sample_mask])

            patch_i_end_previous = patch_i_end

        patch_j_end_previous = 0

        for j in range(n_patch_y):
            observations = y_mark < (((n_patch + j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
            patch_j_end = observations.sum()
            sample_mask = slice(patch_j_end_previous, patch_j_end)
            y_patch_j = y[sample_mask]
            if len(y_patch_j) == 0:
                ys.append(torch.full((1, y.shape[-1]), fill_value=float("nan"), device=y.device))
                y_marks.append(torch.zeros((1), device=y.device))
                y_masks.append(torch.zeros((1, y.shape[-1]), device=y.device))
            else:
                ys.append(y_patch_j)
                y_marks.append(y_mark[sample_mask])
                y_masks.append(y_mask[sample_mask])

            patch_j_end_previous = patch_j_end

        sample_IDs.append(sample_ID)

    ENC_IN = xs[0].shape[-1]

    # manually append a sample with desired shape then removed.
    xs.append(torch.zeros(patch_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(patch_len_max_irr))
    x_masks.append(torch.zeros(patch_len_max_irr, ENC_IN))
    ys.append(torch.zeros(patch_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(patch_len_max_irr))
    y_masks.append(torch.zeros(patch_len_max_irr, ENC_IN))

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    x_marks=pad_sequence(x_marks, batch_first=True)
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    y_marks=pad_sequence(y_marks, batch_first=True)
    y_masks=pad_sequence(y_masks, batch_first=True)

    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    ys = ys[:-1]
    y_marks = y_marks[:-1]
    y_masks = y_masks[:-1]

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

    # note that patch_len_max_irr * n_patch does not necessarily equal to configs.seq_len. see patch_len_max_irr definition for explanation
    return {
        "x": torch.nan_to_num(xs.view(-1, patch_len_max_irr * n_patch, ENC_IN)),
        "x_mark": x_marks.view(-1, patch_len_max_irr * n_patch).unsqueeze(-1).float(),
        "x_mask": x_masks.view(-1, patch_len_max_irr * n_patch, ENC_IN).float(),
        "y": torch.nan_to_num(ys.view(-1, patch_len_max_irr * n_patch_y, ENC_IN)),
        "y_mark": y_marks.view(-1, patch_len_max_irr * n_patch_y).unsqueeze(-1).float(),
        "y_mask": y_masks.view(-1, patch_len_max_irr * n_patch_y, ENC_IN).float(),
        "sample_ID": sample_IDs
    }

def collate_fn_tpatch(
    batch: list[dict[str,Tensor]],
) -> dict[str,Tensor]:
    '''
    tPatchGNN version for Sensor dataset
    rewrite the collate_fn to return dictionary of Tensors, aligning with api
    '''
    global configs

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    sample_IDs: list[int] = []

    PATCH_LEN = configs.patch_len
    SEQ_LEN = configs.seq_len
    assert SEQ_LEN % PATCH_LEN == 0, f"seq_len {SEQ_LEN} should be divisible by patch_len {PATCH_LEN}"
    n_patch: int = SEQ_LEN // PATCH_LEN
    n_patch_y: int = math.ceil(configs.pred_len / PATCH_LEN)

    for sample in batch:
        x = sample["x"]
        y = sample["y"]
        x_mark = sample["x_mark"]
        y_mark = sample["y_mark"]
        x_mask = sample["x_mask"]
        y_mask = sample["y_mask"]
        sample_ID = sample["sample_ID"]

        patch_i_end_previous = 0
        for i in range(n_patch):
            observations = x_mark < ((i + 1) * PATCH_LEN / (SEQ_LEN + configs.pred_len))
            patch_i_end = observations.sum()
            sample_mask = slice(patch_i_end_previous, patch_i_end)
            x_patch_i = x[sample_mask]
            x_mask_patch_i = x_mask[sample_mask]
            for variable in range(x_patch_i.shape[-1]):
                x_patch_i_variable = x_patch_i[:, variable]
                x_mask_patch_i_variable = x_mask_patch_i[:, variable]
                non_zero_mask = x_mask_patch_i_variable > 0
                x_patch_i_non_zero = x_patch_i_variable[non_zero_mask]
                x_mask_patch_i_non_zero = x_mask_patch_i_variable[non_zero_mask]
                if len(x_patch_i_variable) == 0:
                    xs.append(torch.full((1,), fill_value=float("nan"), device=x.device))
                    x_marks.append(torch.zeros((1), device=x.device))
                    x_masks.append(torch.zeros((1), device=x.device))
                else:
                    xs.append(x_patch_i_non_zero)
                    x_marks.append(x_mark[sample_mask][non_zero_mask])
                    x_masks.append(x_mask_patch_i_non_zero)

            patch_i_end_previous = patch_i_end

        patch_j_end_previous = 0

        for j in range(n_patch_y):
            observations = y_mark < (((n_patch + j + 1) * PATCH_LEN) / (SEQ_LEN + configs.pred_len))
            patch_j_end = observations.sum()
            sample_mask = slice(patch_j_end_previous, patch_j_end)
            y_patch_j = y[sample_mask]
            y_mask_patch_j = y_mask[sample_mask]
            for variable in range(y_patch_j.shape[-1]):
                y_patch_j_variable = y_patch_j[:, variable]
                y_mask_patch_j_variable = y_mask_patch_j[:, variable]
                non_zero_mask = y_mask_patch_j_variable > 0
                y_patch_j_non_zero = y_patch_j_variable[non_zero_mask]
                y_mask_patch_j_non_zero = y_mask_patch_j_variable[non_zero_mask]
                if len(y_patch_j_variable) == 0:
                    ys.append(torch.full((1,), fill_value=float("nan"), device=y.device))
                    y_marks.append(torch.zeros((1), device=y.device))
                    y_masks.append(torch.zeros((1), device=y.device))
                else:
                    ys.append(y_patch_j_non_zero)
                    y_marks.append(y_mark[sample_mask][non_zero_mask])
                    y_masks.append(y_mask_patch_j_non_zero)
            
            patch_j_end_previous = patch_j_end

        sample_IDs.append(sample_ID)

    ENC_IN = xs[0].shape[-1]

    xs=pad_sequence(xs, batch_first=True, padding_value=float("nan"))
    # x_marks=pad_sequence(x_marks, batch_first=True)
    x_masks=pad_sequence(x_masks, batch_first=True)
    ys=pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    # y_marks=pad_sequence(y_marks, batch_first=True)
    y_masks=pad_sequence(y_masks, batch_first=True)

    sample_IDs = torch.tensor(sample_IDs).float()

    if configs.missing_rate > 0:
        # manually mask out some observations in input
        # Flatten the mask and data tensor
        flat_mask = x_masks.view(-1)
        flat_x = xs.view(-1)

        # Find indices of available data (where mask is 1)
        available_flat_indices = torch.where(flat_mask == 1)[0]
        num_available = available_flat_indices.size(0)
        num_to_mask = int(configs.missing_rate * num_available)

        if num_to_mask > 0:
            # Generate random permutation on the same device
            perm = torch.randperm(num_available, device=available_flat_indices.device)
            selected_flat = available_flat_indices[perm[:num_to_mask]]
            
            # Apply masking to x and x_mask. In-place operation
            flat_x[selected_flat] = torch.nan
            flat_mask[selected_flat] = 0
        else:
            logger.warning(f"Number of observations {num_available} * missing rate {configs.missing_rate} = {num_to_mask} observations to be masked. Tips: either observations are too sparse, or --missing_rate is too small. Consider increase --missing_rate.")

    return {
        "x": torch.nan_to_num(xs),
        # "x_mark": x_marks.unsqueeze(-1).float(),
        "x_mask": x_masks.float(),
        "y": torch.nan_to_num(ys),
        # "y_mark": y_marks.unsqueeze(-1).float(),
        "y_mask": y_masks.float(),
        "sample_ID": sample_IDs
    }
   