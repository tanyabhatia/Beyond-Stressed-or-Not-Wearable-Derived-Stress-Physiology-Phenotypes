# -*- coding: utf-8 -*-
"""LifeSnaps feature extraction and aggregation"""

import os
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# Configuration
PRE_DIR = Path('./data/lifesnaps/processed/PreProcess')
POST_DIR = Path('./data/lifesnaps/processed/PostProcess/PostDataset')
POST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SLEEP PROCESSING (Part 1 & 2)
# ============================================================================

# Sleep type categorization
SLEEP_NAMES = ['Asleep', 'light', 'rem', 'deep', 'asleep', 'Deep', 'Rem', 
               'Core', 'Light sleep', 'Deep sleep', 'REM sleep', 'Sleep']
NON_SLEEP_NAMES = ['InBed', 'wake', 'awake', 'restless', 'Awake (during sleep)']
LIGHT_SLEEP_NAMES = ['light', 'Light sleep', 'Core']
REM_SLEEP_NAMES = ['rem', 'REM sleep', 'Rem']
DEEP_SLEEP_NAMES = ['deep', 'Deep sleep', 'Deep']

NIGHT_SLEEP_START_HOUR = 18
NIGHT_SLEEP_END_HOUR = 6
NIGHT_MAX_GAP_SECONDS = 10800
NAP_MAX_GAP_SECONDS = 5400
DATE_HOUR_MIN = 0
DATE_HOUR_MAX = 5
DAY_SLEEP_THRESHOLD = 5 * 3600


def calculate_fall_asleep_param(group):
    """Time to fall asleep and time to reach deep sleep for a sleep session."""
    global SLEEP_NAMES, NON_SLEEP_NAMES

    time_to_fall_asleep = 0
    time_asleep = 0
    asleep = False
    bedtime_start = None

    for _, row in group.iterrows():
        if row['Type'] in NON_SLEEP_NAMES:
            time_to_fall_asleep += row['Duration_seconds'] + time_asleep
            time_asleep = 0
            asleep = False
        elif row['Type'] in SLEEP_NAMES:
            if not asleep:
                time_asleep = row['Duration_seconds']
                bedtime_start = row['Bedtime_start']
                asleep = True
            else:
                time_asleep += row['Duration_seconds']
            if time_asleep > 300:
                break

    time_to_deep_sleep = 0
    in_deep = False

    for _, row in group.iterrows():
        if bedtime_start and row['Bedtime_start'] >= bedtime_start:
            if row['Type'] in ['deep', 'Deep', 'Deep sleep']:
                in_deep = True
                break
            else:
                time_to_deep_sleep += row['Duration_seconds']

    if not in_deep:
        time_to_deep_sleep = None

    return bedtime_start, time_to_fall_asleep, time_to_deep_sleep


def process_sleep_first_order(sleep_data, foldername, verbose=False):
    """
    Extract sleep sessions, bedtimes, and stage durations from raw Fitbit events.
    Handles session detection (night sleep vs naps) and edge cases like device resets.
    """
    print(f'Processing {foldername}')

    # Remove duplicates and sort
    sleep_data = sleep_data.drop_duplicates(subset=['datetime']).reset_index(drop=True)
    sleep_data['datetime'] = pd.to_datetime(sleep_data['datetime'])
    sleep_data = sleep_data.sort_values('datetime').reset_index(drop=True)

    sleep_data['Value'] = sleep_data['Value'] * 10
    sleep_data['Duration_seconds'] = sleep_data['Value']

    # Identify sleep sessions (transitions from non-sleep to sleep state)
    sleep_data['Type_shifted'] = sleep_data['Type'].shift(1)
    sleep_data['Sleep_Record_Number'] = 0
    current_session = 0

    for i in range(1, len(sleep_data)):
        prev_type = sleep_data.loc[i - 1, 'Type']
        curr_type = sleep_data.loc[i, 'Type']

        if prev_type in NON_SLEEP_NAMES and curr_type in SLEEP_NAMES:
            current_session += 1

        sleep_data.loc[i, 'Sleep_Record_Number'] = current_session

    sleep_data.loc[0, 'Sleep_Record_Number'] = 0

    # Compute bedtime start/end per session
    bedtime_info = sleep_data.groupby('Sleep_Record_Number').apply(
        lambda g: pd.Series({
            'Bedtime_start': g['datetime'].min(),
            'Bedtime_end': g['datetime'].max(),
        })
    )

    sleep_data = sleep_data.merge(bedtime_info, left_on='Sleep_Record_Number', right_index=True)

    # Aggregate to session level
    aggregated = sleep_data.groupby('Sleep_Record_Number').agg({
        'Bedtime_start': 'first',
        'Bedtime_end': 'first',
        'Duration_seconds': 'sum',
    }).reset_index()

    aggregated['Date'] = pd.to_datetime(aggregated['Bedtime_start']).dt.date

    # Classify sessions (night sleep vs nap)
    aggregated['Session_type'] = 'Nap'
    for idx, row in aggregated.iterrows():
        hour = row['Bedtime_start'].hour
        duration = row['Duration_seconds']

        if (NIGHT_SLEEP_START_HOUR <= hour or hour < NIGHT_SLEEP_END_HOUR) and duration > 1800:
            aggregated.loc[idx, 'Session_type'] = 'Night_Sleep'
        elif 10 <= hour <= 17 and duration > DAY_SLEEP_THRESHOLD:
            aggregated.loc[idx, 'Session_type'] = 'Night_Shift'
        elif duration > 3600:
            aggregated.loc[idx, 'Session_type'] = 'Long_Nap'

    # Break down by stage type
    stage_mapping = {'Light': LIGHT_SLEEP_NAMES, 'REM': REM_SLEEP_NAMES, 'Deep': DEEP_SLEEP_NAMES}
    for stage_col, stage_list in stage_mapping.items():
        aggregated[f'{stage_col.lower()}_sleep'] = sleep_data[
            (sleep_data['Sleep_Record_Number'] == sleep_data['Sleep_Record_Number']) &
            (sleep_data['Type'].isin(stage_list))
        ].groupby('Sleep_Record_Number')['Duration_seconds'].sum().reindex(aggregated['Sleep_Record_Number'], fill_value=0).values

    # Time awake and fall asleep metrics
    aggregated['Time_in_bed'] = aggregated['Duration_seconds']
    aggregated['Time_awake'] = sleep_data[
        sleep_data['Type'].isin(NON_SLEEP_NAMES)
    ].groupby('Sleep_Record_Number')['Duration_seconds'].sum().reindex(aggregated['Sleep_Record_Number'], fill_value=0).values

    aggregated['Time_asleep'] = aggregated['Time_in_bed'] - aggregated['Time_awake']
    aggregated['Awake_count'] = sleep_data[
        sleep_data['Type'].isin(NON_SLEEP_NAMES)
    ].groupby('Sleep_Record_Number').size().reindex(aggregated['Sleep_Record_Number'], fill_value=0).values

    # Fall asleep times
    fall_asleep_results = sleep_data.groupby('Sleep_Record_Number').apply(calculate_fall_asleep_param)
    aggregated['Time_to_fall_asleep'] = fall_asleep_results.apply(lambda x: x[1] if pd.notna(x[0]) else 0)
    aggregated['Time_to_deep_sleep'] = fall_asleep_results.apply(lambda x: x[2])

    # Filter to night sleep only
    main_sleep_data = aggregated[aggregated['Session_type'] == 'Night_Sleep'].copy()
    naps_data = aggregated[aggregated['Session_type'] != 'Night_Sleep'].copy()

    data_to_save = {
        'sleep_data_full': sleep_data,
        'aggregated_data': aggregated,
        'main_sleep_data': main_sleep_data,
        'naps_data': naps_data,
        'foldername': foldername
    }

    return sleep_data, aggregated, main_sleep_data, naps_data


def aggregate_sleep_features():
    """Combine individual sleep feature files into one cohort-level dataset."""
    sleep_dir = PRE_DIR / 'Sleep'
    patient_list = [d.name for d in sleep_dir.iterdir() if d.is_dir()]

    all_sleep_dfs = []
    folder_count = 0

    for foldername in patient_list:
        folder_path = sleep_dir / foldername
        pkl_path = folder_path / 'sleep_data_w_stages.pkl'

        if pkl_path.exists():
            folder_count += 1
            print(f'Processing folder {folder_count} | {foldername}')
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                df_sleep = data['main_sleep_data']

            if isinstance(df_sleep, list):
                print(f'No data for {foldername}')
                continue

            df_sleep = df_sleep[[
                'Date', 'Bedtime_start', 'Bedtime_end',
                'Time_in_bed', 'Time_to_fall_asleep', 'Time_asleep',
                'light_sleep', 'deep_sleep', 'rem_sleep',
                'Time_to_deep_sleep', 'Time_awake', 'Awake_count'
            ]].copy()

            df_sleep['Global_Deidentified'] = foldername
            df_sleep = df_sleep[['Global_Deidentified'] + [c for c in df_sleep.columns if c != 'Global_Deidentified']]
            all_sleep_dfs.append(df_sleep)

    combined_df = pd.concat(all_sleep_dfs, ignore_index=True)
    print(f"Processed {folder_count} folders. Combined shape: {combined_df.shape}")

    # Convert sleep durations from seconds to hours
    time_columns = ['Time_in_bed', 'Time_to_fall_asleep', 'Time_asleep',
                    'light_sleep', 'deep_sleep', 'rem_sleep',
                    'Time_to_deep_sleep', 'Time_awake']
    for col in time_columns:
        combined_df[col] = (combined_df[col] / 3600).round(2)

    combined_df.to_parquet(POST_DIR / 'Sleep_features.parquet', engine='pyarrow', index=False)
    return combined_df


# ============================================================================
# HEART RATE PROCESSING
# ============================================================================

def process_heart_rate(hr_file_path, sleep_pkl_path, foldername, verbose=False):
    """Extract HR statistics during each sleep stage."""
    if verbose:
        print(f"Processing HR for {foldername}...")

    # Load data
    try:
        with open(sleep_pkl_path, 'rb') as f:
            data = pickle.load(f)
        df_sleep = data['sleep_data_full']
    except Exception as e:
        print(f"Error loading sleep data: {e}")
        return pd.DataFrame()

    try:
        df_hr = pd.read_parquet(hr_file_path).drop_duplicates()
    except Exception as e:
        print(f"Error loading HR data: {e}")
        return pd.DataFrame()

    # Ensure datetime types
    df_hr['datetime'] = pd.to_datetime(df_hr['datetime'])
    df_sleep['Bedtime_start'] = pd.to_datetime(df_sleep['Bedtime_start'])
    df_sleep['Bedtime_end'] = pd.to_datetime(df_sleep['Bedtime_end'])

    # Normalize sleep stage names
    mapping = {
        'light': 'light', 'Light sleep': 'light', 'Core': 'light',
        'rem': 'rem', 'REM sleep': 'rem', 'Rem': 'rem',
        'deep': 'deep', 'Deep sleep': 'deep', 'Deep': 'deep',
    }
    df_sleep['Type'] = df_sleep['Type'].replace(mapping)
    valid_types = ['light', 'rem', 'deep']
    df_sleep = df_sleep[df_sleep['Type'].isin(valid_types)]

    if df_sleep.empty:
        print(f"No valid sleep stages for {foldername}")
        return pd.DataFrame()

    # Process each sleep session
    nightly_stats = {}
    for sleep_record, group in df_sleep.groupby('Sleep_Record_Number'):
        stage_hr = {stage: [] for stage in valid_types}

        for _, row in group.iterrows():
            stage = row['Type']
            hr_mask = (df_hr['datetime'] > row['Bedtime_start']) & (df_hr['datetime'] <= row['Bedtime_end'])
            stage_hr[stage].extend(df_hr[hr_mask]['Value'].tolist())

        nightly_stats[sleep_record] = {}
        for stage, values in stage_hr.items():
            if values:
                nightly_stats[sleep_record][f'{stage}_mean_hr'] = np.mean(values)
                nightly_stats[sleep_record][f'{stage}_median_hr'] = np.median(values)
                nightly_stats[sleep_record][f'{stage}_std_hr'] = np.std(values)
            else:
                nightly_stats[sleep_record][f'{stage}_mean_hr'] = None
                nightly_stats[sleep_record][f'{stage}_median_hr'] = None
                nightly_stats[sleep_record][f'{stage}_std_hr'] = None

    # Convert to dataframe and merge with sleep dates
    result_df = pd.DataFrame([
        {'Sleep_Record_Number': k, **v} for k, v in nightly_stats.items()
    ])

    result_df = result_df.merge(
        data['main_sleep_data'][['Sleep_Record_Number', 'Date']],
        on='Sleep_Record_Number', how='left'
    )

    result_df['Global_Deidentified'] = foldername
    return result_df


def aggregate_heart_rate():
    """Combine individual HR feature files into cohort dataset."""
    hr_dir = PRE_DIR / 'HR'
    sleep_dir = PRE_DIR / 'Sleep'

    all_hr_dfs = []
    for foldername in [d.name for d in hr_dir.iterdir() if d.is_dir()]:
        sleep_pkl = sleep_dir / foldername / 'sleep_data_w_stages.pkl'
        hr_file = hr_dir / foldername / 'HR.parquet'

        if sleep_pkl.exists() and hr_file.exists():
            df_hr = process_heart_rate(hr_file, sleep_pkl, foldername)
            if not df_hr.empty:
                all_hr_dfs.append(df_hr)

    if all_hr_dfs:
        combined_df = pd.concat(all_hr_dfs, ignore_index=True)
        combined_df.to_parquet(POST_DIR / 'HR_features.parquet', engine='pyarrow', index=False)
        print(f"Saved HR features: {combined_df.shape}")
        return combined_df


# ============================================================================
# HRV PROCESSING
# ============================================================================

def process_hrv(hrv_file_path, sleep_pkl_path, foldername, verbose=False):
    """Extract HRV statistics (RMSSD, LF, HF, LF/HF) during each sleep stage."""
    if verbose:
        print(f"Processing HRV for {foldername}...")

    # Load HRV data
    df_hrv = pd.read_parquet(hrv_file_path).drop_duplicates(subset=['datetime', 'Type'])
    df_hrv = df_hrv.sort_values('datetime').reset_index(drop=True)

    try:
        df_hrv = df_hrv.pivot(index='datetime', columns='Type', values='Value').reset_index()
        df_hrv.columns.name = None
        df_hrv.rename(columns={'lf': 'LF', 'hf': 'HF', 'rmssd': 'RMSSD'}, inplace=True)
    except Exception as e:
        print(f"Error pivoting HRV: {e}")
        return pd.DataFrame()

    required = ['LF', 'HF', 'RMSSD']
    if not all(c in df_hrv.columns for c in required):
        print(f"Missing HRV columns for {foldername}")
        return pd.DataFrame()

    df_hrv['LF_HF_Ratio'] = df_hrv['LF'] / df_hrv['HF']
    df_hrv = df_hrv[['datetime', 'LF', 'HF', 'LF_HF_Ratio', 'RMSSD']].dropna()

    if df_hrv.empty:
        return pd.DataFrame()

    # Load sleep data
    try:
        with open(sleep_pkl_path, 'rb') as f:
            data = pickle.load(f)
        df_sleep = data['sleep_data_full']
    except Exception as e:
        print(f"Error loading sleep data: {e}")
        return pd.DataFrame()

    # Normalize and filter sleep stages
    mapping = {
        'light': 'light', 'Light sleep': 'light', 'Core': 'light',
        'rem': 'rem', 'REM sleep': 'rem', 'Rem': 'rem',
        'deep': 'deep', 'Deep sleep': 'deep', 'Deep': 'deep',
    }
    df_sleep['Type'] = df_sleep['Type'].replace(mapping)
    valid_types = ['light', 'rem', 'deep']
    df_sleep = df_sleep[df_sleep['Type'].isin(valid_types)]

    df_sleep['Bedtime_start'] = pd.to_datetime(df_sleep['Bedtime_start'])
    df_sleep['Bedtime_end'] = pd.to_datetime(df_sleep['Bedtime_end'])
    df_hrv['datetime'] = pd.to_datetime(df_hrv['datetime'])

    # Process each sleep session
    nightly_stats = {}
    for sleep_record, group in df_sleep.groupby('Sleep_Record_Number'):
        stage_data = {stage: {col: [] for col in ['LF', 'HF', 'LF_HF_Ratio', 'RMSSD']} 
                      for stage in valid_types}

        for _, row in group.iterrows():
            stage = row['Type']
            hrv_mask = (df_hrv['datetime'] > row['Bedtime_start']) & (df_hrv['datetime'] <= row['Bedtime_end'])
            for col in ['LF', 'HF', 'LF_HF_Ratio', 'RMSSD']:
                stage_data[stage][col].extend(df_hrv[hrv_mask][col].tolist())

        nightly_stats[sleep_record] = {}
        for stage, metrics in stage_data.items():
            for col, values in metrics.items():
                if values:
                    nightly_stats[sleep_record][f'deep_{col}_mean'] = np.mean(values)
                    nightly_stats[sleep_record][f'deep_{col}_median'] = np.median(values)
                    nightly_stats[sleep_record][f'deep_{col}_std'] = np.std(values)
                else:
                    nightly_stats[sleep_record][f'deep_{col}_mean'] = None
                    nightly_stats[sleep_record][f'deep_{col}_median'] = None
                    nightly_stats[sleep_record][f'deep_{col}_std'] = None

    result_df = pd.DataFrame([
        {'Sleep_Record_Number': k, **v} for k, v in nightly_stats.items()
    ])

    result_df = result_df.merge(
        data['main_sleep_data'][['Sleep_Record_Number', 'Date']],
        on='Sleep_Record_Number', how='left'
    )

    result_df['Global_Deidentified'] = foldername
    return result_df


def aggregate_hrv():
    """Combine individual HRV feature files into cohort dataset."""
    hrv_dir = PRE_DIR / 'HRV'
    sleep_dir = PRE_DIR / 'Sleep'

    all_hrv_dfs = []
    for foldername in [d.name for d in hrv_dir.iterdir() if d.is_dir()]:
        sleep_pkl = sleep_dir / foldername / 'sleep_data_w_stages.pkl'
        hrv_file = hrv_dir / foldername / 'HRV.parquet'

        if sleep_pkl.exists() and hrv_file.exists():
            df_hrv = process_hrv(hrv_file, sleep_pkl, foldername)
            if not df_hrv.empty:
                all_hrv_dfs.append(df_hrv)

    if all_hrv_dfs:
        combined_df = pd.concat(all_hrv_dfs, ignore_index=True)
        combined_df.to_parquet(POST_DIR / 'HRV_features.parquet', engine='pyarrow', index=False)
        print(f"Saved HRV features: {combined_df.shape}")
        return combined_df


# ============================================================================
# SLEEP REGULARITY INDEX (SRI)
# ============================================================================

def preprocess_sleep_data_for_sri(df_input):
    """Split sleep sessions that cross midnight into separate rows."""
    sleep_data = df_input.copy()
    sleep_data['Bedtime_start'] = pd.to_datetime(sleep_data['Bedtime_start'])
    sleep_data['Bedtime_end'] = pd.to_datetime(sleep_data['Bedtime_end'])

    processed_rows = []
    for _, row in sleep_data.iterrows():
        start = row['Bedtime_start']
        end = row['Bedtime_end']

        if start.date() != end.date():
            # Split across midnight
            end_of_day = start.replace(hour=23, minute=59, second=59)
            start_of_next = end.replace(hour=0, minute=0, second=0)

            first = row.copy()
            first['Bedtime_end'] = end_of_day
            processed_rows.append(first)

            second = row.copy()
            second['Bedtime_start'] = start_of_next
            second['Date'] = end.date()
            processed_rows.append(second)
        else:
            processed_rows.append(row)

    processed = pd.DataFrame(processed_rows)
    processed['Date'] = processed['Bedtime_start'].dt.date
    return processed


def create_interval_data(processed_sleep):
    """Create 30-second epoch intervals marking sleep (0) vs wake (1) for each day."""
    non_sleep = ['InBed', 'wake', 'awake', 'restless', 'Awake (during sleep)', 'Unused', 'Out']
    intervals = pd.date_range('00:00', '23:59:30', freq='30S').time
    interval_cols = [str(t) for t in intervals]

    full_data = pd.DataFrame(columns=['Date'] + interval_cols)

    for date in processed_sleep['Date'].unique():
        row_data = [[date] + [1] * len(intervals)]
        date_rows = processed_sleep[processed_sleep['Date'] == date]

        for _, row in date_rows.iterrows():
            start_t = row['Bedtime_start'].time()
            end_t = row['Bedtime_end'].time()
            sleep_intervals = [str(t) for t in intervals if start_t <= t <= end_t]
            value = 0 if row['Type'] not in non_sleep else 1

            for interval in sleep_intervals:
                row_data[0][interval_cols.index(interval) + 1] = value

        date_df = pd.DataFrame(row_data, columns=['Date'] + interval_cols)
        full_data = pd.concat([full_data, date_df], ignore_index=True)

    full_data['Date'] = pd.to_datetime(full_data['Date'])
    return full_data.sort_values('Date').reset_index(drop=True)


def calculate_sri(interval_data):
    """
    Sleep Regularity Index: measures night-to-night concordance of sleep patterns.
    Range 0-100, higher is more regular.
    """
    arr = interval_data.drop(columns='Date').values
    N = arr.shape[0]
    M = arr.shape[1]

    if M == 0 or N == 0 or N == 1:
        return np.nan, 0

    delta_sum = 0
    for j in range(M):
        for i in range(N - 1):
            if arr[i, j] == arr[i + 1, j]:
                delta_sum += 1

    sri = -100 + (200 / (M * (N - 1))) * delta_sum
    return sri, N


def compute_sri():
    """Compute SRI for all participants."""
    sleep_dir = PRE_DIR / 'Sleep'
    patient_list = [d.name for d in sleep_dir.iterdir() if d.is_dir()]

    sri_scores = []
    sri_ids = []
    sri_nights = []
    start = time.time()
    folder_count = 0

    for foldername in patient_list:
        pkl_path = sleep_dir / foldername / 'sleep_data_w_stages.pkl'

        if pkl_path.exists():
            folder_count += 1
            print(f'Processing folder {folder_count}: {foldername}')

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                sleep_full = data['sleep_data_full']

            processed = preprocess_sleep_data_for_sri(sleep_full)
            intervals = create_interval_data(processed)
            sri, n = calculate_sri(intervals)

            if not np.isnan(sri):
                sri_scores.append(sri)
                sri_ids.append(foldername)
                sri_nights.append(n)

    df_sri = pd.DataFrame({
        'Global_Deidentified': sri_ids,
        'SRI': sri_scores,
        'NumNights': sri_nights
    })

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"Processed {folder_count} folders in {elapsed}")

    df_sri.to_csv(POST_DIR / 'SRI.csv', index=False)
    return df_sri


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("LifeSnaps feature extraction pipeline\n")

    print("1. Extracting sleep features...")
    df_sleep = aggregate_sleep_features()

    print("\n2. Extracting heart rate features...")
    df_hr = aggregate_heart_rate()

    print("\n3. Extracting HRV features...")
    df_hrv = aggregate_hrv()

    print("\n4. Computing Sleep Regularity Index...")
    df_sri = compute_sri()

    print("\nAll features complete!")


if __name__ == '__main__':
    main()
