"""
=============================================================
  SLEEP ANALYSIS PIPELINE  –  Raspberry Pi
  Senzori: MAX30102, MPU6050, MAX4466, Flex+ADS1015, BMP280
  Fereastra de analiza: 30 secunde
  Output: features CSV, clasificare K-Means/KNN, scor somn, grafice
=============================================================
"""

import sqlite3
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import signal
from scipy.stats import iqr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")                                             
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os, sys, json
from datetime import datetime

                                               
               
                                               
DB_PATH = "/home/pi/raw_sensor_data.db"                                    
OUTPUT_DIR    = "sleep_output"                                           
WINDOW_SEC    = 30                                                 
RAW_TABLE     = "raw_data"                           
MIC_TABLE     = "mic_data"                                    
FLEX_TABLE    = "flex_data"                                        
FS_IMU        = 10                                              
FS_PPG        = 12                                                                                           
FS_MIC        = 200                                        
FS_FLEX       = 5                                             
N_CLUSTERS    = 4                                                     

                                                                    
                                                                 
                                                       
SPO2_OFFSET   = 4.0                                                      

os.makedirs(OUTPUT_DIR, exist_ok=True)

                                               
                               
                                               

def load_raw_data(db_path: str, session_id: str = None) -> pd.DataFrame:
    """Incarca datele brute din tabelul raw_data."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {RAW_TABLE}"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    query += " ORDER BY pc_timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

                         
    df["pc_timestamp"] = pd.to_datetime(df["pc_timestamp"])
    df = df.sort_values("pc_timestamp").reset_index(drop=True)
    print(f"[RAW]  {len(df)} randuri incarcate | sesiune: {session_id or 'toate'}")
    return df


def load_mic_data(db_path: str, session_id: str = None) -> pd.DataFrame:
    """Incarca datele microfon din tabelul mic_data."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {MIC_TABLE}"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    query += " ORDER BY pc_timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["pc_timestamp"] = pd.to_datetime(df["pc_timestamp"])
    print(f"[MIC]  {len(df)} blocuri microfon incarcate")
    return df


PPG_TABLE = "ppg_data"


def load_ppg_data(db_path: str, session_id: str = None) -> pd.DataFrame:
    """Incarca datele PPG din tabelul ppg_data."""
    conn = sqlite3.connect(db_path)
                                  
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ppg_data'", conn
    )
    if tables.empty:
        print("[PPG]  Tabel ppg_data inexistent - folosesc raw_data pentru PPG")
        conn.close()
        return pd.DataFrame()
    query = f"SELECT * FROM {PPG_TABLE}"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    query += " ORDER BY pc_timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        print("[PPG]  Nicio date PPG burst gasite")
        return df
    df["pc_timestamp"] = pd.to_datetime(df["pc_timestamp"])
    df = df.sort_values("pc_timestamp").reset_index(drop=True)
    print(f"[PPG]  {len(df)} blocuri PPG burst incarcate | sesiune: {session_id or 'toate'}")
    return df


def load_flex_data(db_path: str, session_id: str = None) -> pd.DataFrame:
    """Incarca datele flex buffered din tabelul flex_data (4Hz)."""
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='flex_data'", conn
    )
    if tables.empty:
        conn.close()
        return pd.DataFrame()
    query = f"SELECT * FROM {FLEX_TABLE}"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    query += " ORDER BY pc_timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        return df
    df["pc_timestamp"] = pd.to_datetime(df["pc_timestamp"])
    df = df.sort_values("pc_timestamp").reset_index(drop=True)
    print(f"[FLEX] {len(df)} blocuri flex incarcate | sesiune: {session_id or 'toate'}")
    return df


def expand_flex_blocks(flex_df: pd.DataFrame) -> pd.DataFrame:
    """Expandeaza blocurile flex intr-un DataFrame cu timestamp per sample.
    Similar cu expand_ppg_blocks — interpoleaza timestamps uniform pe bloc."""
    if flex_df.empty:
        return pd.DataFrame(columns=["pc_timestamp", "flex_raw", "flex_delta"])

    all_rows = []
    for _, row in flex_df.iterrows():
        fr_str = str(row.get("fr_samples", ""))
        fd_str = str(row.get("fd_samples", ""))
        if not fr_str or fr_str == "nan":
            continue
        try:
            fr_vals = [int(x) for x in fr_str.split(",") if x.strip()]
            fd_vals = [int(x) for x in fd_str.split(",") if x.strip()]
        except ValueError:
            continue
        n = min(len(fr_vals), len(fd_vals))
        if n == 0:
            continue
        ts = row["pc_timestamp"]
        for i in range(n):
            all_rows.append({
                "pc_timestamp": ts + pd.Timedelta(milliseconds=i * 250),               
                "flex_raw":     fr_vals[i],
                "flex_delta":   fd_vals[i],
            })

    if not all_rows:
        return pd.DataFrame(columns=["pc_timestamp", "flex_raw", "flex_delta"])

    result = pd.DataFrame(all_rows)
    result = result.sort_values("pc_timestamp").reset_index(drop=True)

                    
    if len(result) >= 3:
        diffs = result["pc_timestamp"].diff().dt.total_seconds().dropna()
        median_int = float(diffs.median())
        block_sizes = flex_df["fr_samples"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)
        bs = int(block_sizes.median())
        print(f"[FLEX] {len(result)} samples expandate la ~{1/median_int:.0f}Hz "
              f"(block_size={bs}, interval={median_int*1000:.0f}ms)")
    return result


def expand_ppg_blocks(ppg_df: pd.DataFrame) -> pd.DataFrame:
    """Expandeaza blocurile PPG intr-un DataFrame cu timestamp per sample.
    Detecteaza automat sample rate din intervalul intre blocuri."""
    global FS_PPG
    if ppg_df.empty:
        return pd.DataFrame()

                                                 
    first_ir = str(ppg_df.iloc[0]["ir_samples"]).split(",")
    block_size = len([x for x in first_ir if x.strip()])

                                                                               
    diffs = pd.Series(dtype=float)
    if len(ppg_df) >= 3:
        block_times = ppg_df["pc_timestamp"].values
        diffs = pd.Series(block_times).diff().dt.total_seconds().dropna()
        median_interval = float(diffs.median())
        if median_interval > 0:
            FS_PPG = max(int(round(block_size / median_interval)), 1)
        else:
            FS_PPG = block_size
    else:
        FS_PPG = block_size

    block_rate_hz = (1.0 / float(diffs.median())) if len(diffs) > 0 and float(diffs.median()) > 0 else 0.0
    sample_interval_ms = 1000.0 / FS_PPG
    print(f"[PPG]  FS real={FS_PPG}Hz (block_size={block_size}, "
          f"block_rate={block_rate_hz:.1f}Hz, sample_interval={sample_interval_ms:.1f}ms)")

    rows = []
    for _, row in ppg_df.iterrows():
        try:
            ir_vals  = [int(x) for x in str(row["ir_samples"]).split(",") if x.strip()]
            red_vals = [int(x) for x in str(row["red_samples"]).split(",") if x.strip()]
        except Exception:
            continue
        n = min(len(ir_vals), len(red_vals))
        t_base = row["pc_timestamp"]
        for i in range(n):
            rows.append({
                "pc_timestamp": t_base + pd.Timedelta(milliseconds=i * sample_interval_ms),
                "ir":  ir_vals[i],
                "red": red_vals[i],
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("pc_timestamp").reset_index(drop=True)
    print(f"[PPG]  {len(df)} sample-uri PPG expandate la {FS_PPG}Hz")
    return df


def get_sessions(db_path: str):
    """Returneaza lista de sesiuni disponibile."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT DISTINCT session_id FROM {RAW_TABLE}", conn)
    conn.close()
    return df["session_id"].dropna().tolist()

                                               
                           
                                               

def split_windows(df: pd.DataFrame, window_sec: int = WINDOW_SEC):
    """Imparte dataframe-ul in ferestre de window_sec secunde."""
    if df.empty:
        return []
    t_start = df["pc_timestamp"].iloc[0]
    t_end   = df["pc_timestamp"].iloc[-1]
    windows = []
    t = t_start
    while t < t_end:
        t_next = t + pd.Timedelta(seconds=window_sec)
        chunk = df[(df["pc_timestamp"] >= t) & (df["pc_timestamp"] < t_next)]
        if len(chunk) >= 5:                                              
            windows.append((t, chunk))
        t = t_next
    print(f"[WIN]  {len(windows)} ferestre de {window_sec}s")
    return windows

                                               
                        
                                               

                                              

def bandpass_filter(sig, lowcut=0.5, highcut=3.5, fs=FS_PPG, order=3):
    """Filtru bandpass Butterworth pentru semnal PPG.
    Ajusteaza automat la Nyquist daca fs e mic (ex. 5Hz → highcut devine 2.4Hz)."""
    nyq = fs / 2.0
    highcut_safe = min(highcut, nyq * 0.95)                       
    lowcut_safe  = max(lowcut, 0.1)
    if lowcut_safe >= highcut_safe:
        return sig                                         
    lo = lowcut_safe  / nyq
    hi = highcut_safe / nyq
    lo = max(lo, 1e-4)
    hi = min(hi, 0.999)
    b, a = signal.butter(order, [lo, hi], btype="band")
    return signal.filtfilt(b, a, sig)


                                                                
_ppg_50hz_df: pd.DataFrame = pd.DataFrame()

                                                                    
_flex_4hz_df: pd.DataFrame = pd.DataFrame()


def compute_ppg_features(chunk: pd.DataFrame, t_start=None, t_end=None) -> dict:
    """Extrage features PPG dintr-o fereastra.
    Daca exista date PPG burst in _ppg_50hz_df le foloseste preferential.
    Altfel foloseste datele la 1Hz din raw_data.
    """
    feats = {}
    nan_result = {k: np.nan for k in [
        "bpm_mean","bpm_std","bpm_min","bpm_max",
        "contact_valid_ratio","hrv_simple","spo2_estimated"]}
    nan_result["ppg_valid"] = 0.0

                                                            
    global _ppg_50hz_df
    use_50hz = False
    if not _ppg_50hz_df.empty and t_start is not None and t_end is not None:
        ppg_chunk = _ppg_50hz_df[
            (_ppg_50hz_df["pc_timestamp"] >= t_start) &
            (_ppg_50hz_df["pc_timestamp"] <  t_end)
        ]
        if len(ppg_chunk) >= max(int(FS_PPG * 0.5), 10):                           
            ir  = ppg_chunk["ir"].values.astype(float)
            red = ppg_chunk["red"].values.astype(float)
            fs  = FS_PPG
            use_50hz = True

                                                           
    if not use_50hz:
        ir  = chunk["ir"].dropna().values.astype(float)
        red = chunk["red"].dropna().values.astype(float)
        fs  = FS_IMU          
        if len(ir) < 10:
            return nan_result

                                                            
    valid_mask = (ir > 50000) & (ir < 250000)
    contact_ratio = float(np.mean(valid_mask))
    feats["contact_valid_ratio"] = contact_ratio
                                                                        
    feats["ppg_valid"] = 1.0 if contact_ratio >= 0.5 else 0.0
    if contact_ratio < 0.5:
        nan_result["contact_valid_ratio"] = contact_ratio
        nan_result["ppg_valid"] = 0.0
        return nan_result

    ir  = ir[valid_mask]
    red_valid = red[valid_mask] if len(red) == len(valid_mask) else red

                                                            
    dc      = np.mean(ir)
    ir_norm = (ir - dc) / dc * 100

    try:
        ir_filt = bandpass_filter(ir_norm, lowcut=0.5, highcut=3.5, fs=fs)
    except Exception:
        ir_filt = ir_norm

                                                                 
                                              
    min_dist = max(int(fs * 0.5), 1)
    prom_min = max(np.std(ir_filt) * 0.6, 0.1)                        
    peaks, _ = signal.find_peaks(
        ir_filt,
        distance=min_dist,
        prominence=prom_min,
        height=0
    )

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs                 
        bpm_arr = 60.0 / rr_intervals
        bpm_arr = bpm_arr[(bpm_arr >= 40) & (bpm_arr <= 120)]
    else:
        bpm_arr = np.array([])

    feats["bpm_mean"] = float(np.mean(bpm_arr))   if len(bpm_arr) > 0 else np.nan
    feats["bpm_std"]  = float(np.std(bpm_arr))    if len(bpm_arr) > 1 else np.nan
    feats["bpm_min"]  = float(np.min(bpm_arr))    if len(bpm_arr) > 0 else np.nan
    feats["bpm_max"]  = float(np.max(bpm_arr))    if len(bpm_arr) > 0 else np.nan

                                                                         
     
                                                                   
                                              
                                                                      
                                                                       
                                                                         
                                                                     
    if fs < 10:
                                          
        if len(bpm_arr) >= 4:
            bpm_std_val = float(np.std(bpm_arr))
            if bpm_std_val > 0:
                hrv_est = float(np.clip(bpm_std_val * 2.5, 5.0, 120.0))
                feats["hrv_simple"] = hrv_est
            else:
                feats["hrv_simple"] = np.nan
        else:
            feats["hrv_simple"] = np.nan
    elif len(peaks) >= 3 and len(bpm_arr) >= 2:
                                                                               
                                                                    
        refined_peaks = []
        for p in peaks:
            if 1 <= p <= len(ir_filt) - 2:
                y0, y1, y2 = ir_filt[p-1], ir_filt[p], ir_filt[p+1]
                denom = 2.0 * (2*y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    delta = (y0 - y2) / denom
                    delta = max(-0.5, min(0.5, delta))         
                    refined_peaks.append(p + delta)
                else:
                    refined_peaks.append(float(p))
            else:
                refined_peaks.append(float(p))
        refined_peaks = np.array(refined_peaks)
        rr_intervals_refined = np.diff(refined_peaks) / fs
        bpm_refined = 60.0 / rr_intervals_refined
        bpm_refined = bpm_refined[(bpm_refined >= 40) & (bpm_refined <= 120)]
        
        if len(bpm_refined) >= 2:
            rr_sec = 60.0 / bpm_refined
            rr_ms  = rr_sec * 1000
                                                                             
            valid_mask_rr = np.ones(len(rr_ms), dtype=bool)
            for i in range(1, len(rr_ms)):
                if abs(rr_ms[i] - rr_ms[i-1]) / rr_ms[i-1] > 0.30:
                    valid_mask_rr[i] = False
            rr_clean = rr_ms[valid_mask_rr]
            if len(rr_clean) >= 2:
                diffs = np.diff(rr_clean)
                rmssd = float(np.sqrt(np.mean(diffs**2)))
                                                               
                if 5 <= rmssd <= 200:
                    feats["hrv_simple"] = rmssd
                else:
                    feats["hrv_simple"] = np.nan
            else:
                feats["hrv_simple"] = np.nan
        else:
            feats["hrv_simple"] = np.nan
    else:
        feats["hrv_simple"] = np.nan

                                                            
                                                  
                                                                 
                                
                                                          
                                                                    
    if len(red_valid) > 0 and np.mean(ir) > 1000:
        dc_red = np.mean(red_valid)
        dc_ir  = np.mean(ir)

        if dc_red > 0 and dc_ir > 0 and len(ir) >= 10:
                                                                                  
            try:
                ir_bp  = bandpass_filter(ir - dc_ir, 0.5, 3.5, fs)
                red_bp = bandpass_filter(red_valid - dc_red, 0.5, 3.5, fs)
                ac_ir  = np.sqrt(np.mean(ir_bp**2))                          
                ac_red = np.sqrt(np.mean(red_bp**2))
            except Exception:
                ac_ir  = np.std(ir)
                ac_red = np.std(red_valid)

            if ac_ir > 0:
                ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
                                                                              
                                                                          
                raw_spo2 = 104 - 17 * ratio + SPO2_OFFSET
                feats["spo2_estimated"] = float(np.clip(raw_spo2, 85, 100))
            else:
                feats["spo2_estimated"] = np.nan
        else:
            feats["spo2_estimated"] = np.nan
    else:
        feats["spo2_estimated"] = np.nan

    return feats


                                              

def compute_imu_features(chunk: pd.DataFrame) -> dict:
    """Extrage features de miscare din IMU.

    IMPORTANT: motion_level NU se calculeaza din |acc_mag - 1g| pentru ca
    bratara stationara are acc_mag ≈ 1g constant (gravitatia), rezultand in
    motion_level ≈ 0 chiar cand exista micro-miscari perceptibile pe axe.

    Solutia: motion_level = deviatia standard a magnitudinii (std of mag)
    + contributie din gyro. Asta captureaza micro-miscari chiar cand acc_mag = 1g.
    """
    feats = {}
    cols_acc  = ["acc_x","acc_y","acc_z"]
    cols_gyro = ["gyro_x","gyro_y","gyro_z"]

    acc  = chunk[cols_acc].dropna().values.astype(float)
    gyro = chunk[cols_gyro].dropna().values.astype(float)

    if len(acc) < 3:
        return {k: np.nan for k in [
            "motion_level","motion_per_min","motion_variation",
            "dominant_position","gyro_energy","acc_jerk"]}

                                   
    acc_mag = np.linalg.norm(acc, axis=1)

                                                                            
                                                                  
                                                     
                                                          
    motion_std = float(np.std(acc_mag))

                                                               
                                                      
    if len(acc_mag) >= 2:
        jerk = np.diff(acc_mag)
        acc_jerk = float(np.std(jerk))
    else:
        acc_jerk = 0.0

                                                
                                                                       
    feats["motion_level"] = motion_std + acc_jerk * 0.5

                                                                
                                                               
    jerk_events = np.sum(np.abs(np.diff(acc_mag)) > 0.01) if len(acc_mag) >= 2 else 0
    feats["motion_per_min"] = float(jerk_events / WINDOW_SEC * 60)

                                                    
    feats["motion_variation"] = float(np.std(acc_mag) / (np.mean(acc_mag) + 1e-6))

                                   
    feats["acc_jerk"] = acc_jerk

                                                               
    acc_mean = np.mean(acc, axis=0)
    dominant = np.argmax(np.abs(acc_mean))
    feats["dominant_position"] = float(dominant)

                                                            
                                                               
    if len(gyro) >= 3:
        gyro_mag = np.linalg.norm(gyro, axis=1)
                                                     
        gyro_centered = gyro_mag - np.median(gyro_mag)
        feats["gyro_energy"] = float(np.std(gyro_centered))
    else:
        feats["gyro_energy"] = np.nan

    return feats


                                            

def compute_flex_features(chunk: pd.DataFrame) -> dict:
    """Extrage features de respiratie din flex sensor.
    Metoda hibrida: prefera peak detection cand >=3 peaks, FFT ca fallback.
    Auto-detecteaza FS din chunk timestamps (nu se bazeaza pe FS_FLEX global)."""
    feats = {}
    flex = chunk["flex_delta"].dropna().values.astype(float)

    nan_result = {k: np.nan for k in [
        "resp_rate","resp_variability","resp_amplitude",
        "apnea_events_simple","breath_valid_ratio"]}

    if len(flex) < 5:
        return nan_result

                              
                                                                                    
                                                   
                                                     
                                                                    
    fs_local = FS_FLEX
    if "pc_timestamp" in chunk.columns and len(chunk) >= 3:
        ts_diffs = chunk["pc_timestamp"].diff().dt.total_seconds().dropna()
        if len(ts_diffs) > 0:
            med_int = float(ts_diffs.median())
            if 0.01 < med_int < 30:
                                                                            
                block_size_local = len(flex)                          
                n_blocks = len(chunk)
                if n_blocks > 0:
                    bs = block_size_local / n_blocks                     
                    fs_local = round(bs / med_int, 2)
                    fs_local = max(0.5, min(fs_local, 20.0))                            

                                                                                  
    amplitude = float(np.ptp(flex))
    if amplitude < 30:
        nan_result["resp_amplitude"] = amplitude
        nan_result["breath_valid_ratio"] = 0.0
        return nan_result

              
    if len(flex) >= 5:
        flex_smooth = pd.Series(flex).rolling(3, center=True, min_periods=1).mean().values
    else:
        flex_smooth = flex

                                                                    
    resp_rate_peaks = np.nan
    resp_var_peaks  = np.nan
    apnea_peaks     = np.nan

    min_dist = max(1, int(fs_local * 1.5))
    try:
        prom = max(np.std(flex_smooth) * 0.15, 0.5)
        peaks, _ = signal.find_peaks(flex_smooth, distance=min_dist,
                                     prominence=prom)
    except Exception:
        peaks = np.array([])

    if len(peaks) >= 2:
        breath_intervals = np.diff(peaks) / fs_local
        breath_rr = 60.0 / breath_intervals
        breath_rr = breath_rr[(breath_rr > 3) & (breath_rr < 40)]
        if len(breath_rr) > 0:
            resp_rate_peaks = float(np.mean(breath_rr))
            resp_var_peaks  = float(np.std(breath_rr)) if len(breath_rr) > 1 else np.nan
        long_pauses = np.sum(breath_intervals > 10)
        apnea_peaks = float(long_pauses)

                                                                    
    resp_rate_fft = np.nan
    try:
        flex_detrend = flex_smooth - np.linspace(flex_smooth[0], flex_smooth[-1], len(flex_smooth))
                                                                           
                                                   
        n_pad = max(512, len(flex_detrend))
        windowed = flex_detrend * np.hanning(len(flex_detrend))
        fft_vals = np.abs(np.fft.rfft(windowed, n=n_pad))
        freqs    = np.fft.rfftfreq(n_pad, d=1.0/fs_local)
        nyq = fs_local / 2.0
        f_min = 0.12
        f_max = min(0.5, nyq - 0.01)
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(mask):
            fft_band = fft_vals[mask]
            freq_band = freqs[mask]
            peak_idx = np.argmax(fft_band)
            resp_rate_fft = float(freq_band[peak_idx] * 60.0)
    except Exception:
        pass

                                                                   
                                                                                    
                                                  
    if len(peaks) >= 3 and not np.isnan(resp_rate_peaks):
        feats["resp_rate"]           = resp_rate_peaks
        feats["resp_variability"]    = resp_var_peaks
        feats["apnea_events_simple"] = apnea_peaks if fs_local >= 2.0 else 0.0
    elif not np.isnan(resp_rate_fft):
        feats["resp_rate"]           = resp_rate_fft
        feats["resp_variability"]    = resp_var_peaks                         
        feats["apnea_events_simple"] = 0.0                                     
    elif not np.isnan(resp_rate_peaks):
                                                   
        feats["resp_rate"]           = resp_rate_peaks
        feats["resp_variability"]    = resp_var_peaks
        feats["apnea_events_simple"] = 0.0
    else:
        feats["resp_rate"]           = np.nan
        feats["resp_variability"]    = np.nan
        feats["apnea_events_simple"] = 0.0

    feats["resp_amplitude"]    = amplitude
                                                                      
    noise_floor = max(amplitude * 0.1, 2.0)
    feats["breath_valid_ratio"] = float(np.mean(np.abs(flex - np.median(flex)) > noise_floor))

    return feats


                                            

def compute_temp_features(chunk: pd.DataFrame) -> dict:
    """Extrage features de temperatura."""
    temp = chunk["temp_c"].dropna().values.astype(float)
                                                               
    temp = temp[(temp >= 20) & (temp <= 45)]
    if len(temp) == 0:
        return {"temp_mean": np.nan, "temp_variation": np.nan}
    return {
        "temp_mean":      float(np.mean(temp)),
        "temp_variation": float(np.std(temp))
    }


                                            

def compute_global_snore(mic_df: pd.DataFrame) -> pd.DataFrame:
    """Calculeaza RMS per bloc MIC pe toata sesiunea si marcheaza sforaitul.

    Strategie stricta anti-fals-pozitiv:
    - Threshold = max(P97, median * 3.0, 150) — doar top 3% RMS-uri
    - Mediana * 3.0 asigura ca sforaitul e semnificativ peste zgomot ambient
    - Minim absolut 150 — evita declansarea pe silenta ambientala / respiratie normala
    - OBLIGATORIU: minim 2 blocuri consecutive peste threshold (sforait = sunet sustinut)
    """
    if mic_df.empty:
        return pd.DataFrame()
    results = []
    for _, row in mic_df.iterrows():
        try:
            vals = np.array([float(x) for x in str(row["samples"]).split(",") if x.strip()])
            if len(vals) < 10:
                continue
            rms = np.sqrt(np.mean((vals - np.mean(vals))**2))
            results.append({"pc_timestamp": row["pc_timestamp"], "rms": rms})
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)

    p97 = df["rms"].quantile(0.97)
    p50 = df["rms"].quantile(0.50)
                                                                           
    threshold = max(p97, p50 * 3.0, 150.0)

    above = (df["rms"] > threshold).astype(int).values

                                                               
                                                    
    is_snore = np.zeros(len(above), dtype=bool)
    for i in range(len(above)):
        if above[i]:
                                                                                       
            has_neighbor = False
            if i > 0 and above[i-1]:
                has_neighbor = True
            if i < len(above)-1 and above[i+1]:
                has_neighbor = True
            is_snore[i] = has_neighbor

    df["is_snore"] = is_snore
    print(f"[MIC]  RMS: median={p50:.1f}, P97={p97:.1f} | "
          f"Threshold sforait: {threshold:.1f} | Blocuri sforait: {df['is_snore'].sum()}/{len(df)} "
          f"(din {above.sum()} peste threshold, {df['is_snore'].sum()} consecutive)")
    return df


def find_mic_window(mic_df: pd.DataFrame, t_start, window_sec: int):
    """Gaseste blocul MIC cel mai apropiat de centrul ferestrei."""
    t_center = t_start + pd.Timedelta(seconds=window_sec / 2)
    mic_df = mic_df.copy()
    mic_df["dist"] = (mic_df["pc_timestamp"] - t_center).abs()
    closest = mic_df.nsmallest(1, "dist")
    if closest.empty or closest["dist"].iloc[0] > pd.Timedelta(seconds=60):
        return pd.DataFrame()
    return closest


def compute_mic_features(mic_chunk: pd.DataFrame) -> dict:
    """Detecteaza sforait din blocurile de microfon."""
    feats = {"snore_events_count": 0.0, "snore_intensity": 0.0}

    if mic_chunk.empty:
        return feats

                                                                 
    if "samples" in mic_chunk.columns:
        all_samples = []
        for row in mic_chunk["samples"]:
            if isinstance(row, str):
                vals = [float(x) for x in row.split(",") if x.strip().lstrip("-").isdigit()]
                all_samples.extend(vals)
        all_samples = np.array(all_samples)
    else:
        return feats

    if len(all_samples) < FS_MIC:
        return feats

                            
    all_samples = all_samples / 4096.0

                                  
    block_size = int(FS_MIC * 0.5)
    n_blocks = len(all_samples) // block_size
    rms_arr = []
    for i in range(n_blocks):
        blk = all_samples[i*block_size:(i+1)*block_size]
        rms = np.sqrt(np.mean((blk - np.mean(blk))**2))                             
        rms_arr.append(rms)

    rms_arr = np.array(rms_arr)
                                                           
                                              
    threshold = max(np.mean(rms_arr) + 1.0 * np.std(rms_arr), 80.0)
    snore_mask = rms_arr > threshold

    feats["snore_events_count"] = float(np.sum(snore_mask))
    feats["snore_intensity"]    = float(np.mean(rms_arr[snore_mask])) if np.any(snore_mask) else 0.0

    return feats


                                            

def extract_all_features(windows, mic_df: pd.DataFrame = None, raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extrage toate features din ferestrele de 30s."""
                                              
    snore_global = pd.DataFrame()
    if mic_df is not None and not mic_df.empty:
        snore_global = compute_global_snore(mic_df)

    records = []
    for t_start, chunk in windows:
        row = {"window_start": t_start}

        row.update(compute_ppg_features(chunk, t_start=t_start, t_end=t_start + pd.Timedelta(seconds=WINDOW_SEC)))
        row.update(compute_imu_features(chunk))

                                                                                   
        global _flex_4hz_df
        if not _flex_4hz_df.empty:
                                                  
            t_end_flex = t_start + pd.Timedelta(seconds=WINDOW_SEC)
            flex_chunk = _flex_4hz_df[
                (_flex_4hz_df["pc_timestamp"] >= t_start) &
                (_flex_4hz_df["pc_timestamp"] < t_end_flex)
            ]
            row.update(compute_flex_features(flex_chunk))
        elif FS_FLEX < 2.0 and raw_df is not None and "pc_timestamp" in raw_df.columns:
            flex_start = t_start - pd.Timedelta(seconds=WINDOW_SEC // 2)
            flex_end   = t_start + pd.Timedelta(seconds=WINDOW_SEC + WINDOW_SEC // 2)
            flex_chunk = raw_df[
                (raw_df["pc_timestamp"] >= flex_start) &
                (raw_df["pc_timestamp"] < flex_end)
            ]
            row.update(compute_flex_features(flex_chunk))
        else:
            row.update(compute_flex_features(chunk))

        row.update(compute_temp_features(chunk))

        if not snore_global.empty:
            t_end = t_start + pd.Timedelta(seconds=WINDOW_SEC)
                                                                                              
            snore_chunk = snore_global[
                (snore_global["pc_timestamp"] >= t_start) &
                (snore_global["pc_timestamp"] < t_end)
            ]
                                                                               
            if snore_chunk.empty:
                snore_chunk = snore_global[
                    (snore_global["pc_timestamp"] >= t_start - pd.Timedelta(seconds=15)) &
                    (snore_global["pc_timestamp"] < t_end + pd.Timedelta(seconds=15))
                ]
            row["snore_events_count"] = float(snore_chunk["is_snore"].sum())
            row["snore_intensity"]    = float(snore_chunk.loc[snore_chunk["is_snore"],"rms"].mean()) if snore_chunk["is_snore"].any() else 0.0
                                                                                      
                                                                       
            row["mic_rms_mean"] = float(snore_chunk["rms"].mean()) if not snore_chunk.empty else 0.0
        else:
            row.update({"snore_events_count": np.nan, "snore_intensity": np.nan})

                          
        if "session_id" in chunk.columns:
            row["session_id"] = chunk["session_id"].iloc[0]
        if "subject_id" in chunk.columns:
            row["subject_id"] = chunk["subject_id"].iloc[0]

        records.append(row)

    df = pd.DataFrame(records)
    print(f"[FEAT] {len(df)} ferestre cu {len(df.columns)-3} features extrase")

                        
    if "snore_events_count" in df.columns:
        total_snore_per_window = int(df["snore_events_count"].fillna(0).sum())
        windows_with_snore = int((df["snore_events_count"].fillna(0) > 0).sum())
        print(f"[FEAT] Sforait atribuit: {total_snore_per_window} evenimente "
              f"in {windows_with_snore} ferestre")

    return df

                                               
                           
                                               

def get_feature_cols(df):
    """Selecteaza dinamic features disponibile - exclude flex daca e deconectat,
    exclude hrv_simple daca FS PPG prea mic (toate NaN)."""
    all_cols = [
        "bpm_mean","bpm_std","bpm_min","bpm_max",
        "spo2_estimated",
        "motion_level","motion_per_min","motion_variation",
        "dominant_position","gyro_energy","acc_jerk",
        "temp_mean","temp_variation",
                                                                        
                                                                
                                                                             
    ]

                                                                   
    if "hrv_simple" in df.columns:
        valid_hrv = df["hrv_simple"].notna().mean()
        if valid_hrv > 0.1:
            all_cols.insert(4, "hrv_simple")                
            print(f"[FEAT] HRV inclus ({valid_hrv*100:.0f}% valid)")
        else:
            print(f"[FEAT] HRV exclus din features (FS PPG prea mic, {valid_hrv*100:.0f}% valid)")

    flex_cols = ["resp_rate","resp_variability","resp_amplitude",
                 "apnea_events_simple","breath_valid_ratio"]

                                                            
    if "resp_rate" in df.columns:
        valid_resp = df["resp_rate"].notna().mean()
        if valid_resp > 0.1:
            all_cols += flex_cols
            print(f"[FEAT] Flex inclus ({valid_resp*100:.0f}% valid)")
        else:
            print(f"[FEAT] Flex exclus din features (prea putine date valide)")

    return [c for c in all_cols if c in df.columns]

FEATURE_COLS = [
    "bpm_mean","bpm_std","bpm_min","bpm_max","hrv_simple",
    "spo2_estimated",
    "motion_level","motion_per_min","motion_variation",
    "dominant_position","gyro_energy","acc_jerk",
    "resp_rate","resp_variability","resp_amplitude",
    "apnea_events_simple","breath_valid_ratio",
    "temp_mean","temp_variation",
]


def detect_apnea_multisensor(df: pd.DataFrame) -> pd.DataFrame:
    """Detectie apnee multi-senzor: SpO2 desaturare + amplitudine flex scazuta.

    Metoda clinica standard: apnee = desaturare SpO2 >= 4% sub baseline.
    Complement: amplitudine respiratorie scazuta DOAR cand flex-ul functiona
                (breath_valid_ratio > 0 = senzor conectat).

    Reguli anti-fals-pozitiv:
    - SpO2 desat >= 4% (nu 3%) deoarece estimarea MAX30102 are std ~2.5%
    - Amplitudine flex: se ignora cand breath_valid_ratio == 0 (flex deconectat ≠ apnee)
    - Se foloseste P75 ca baseline amplitudine (mai robust la sesiuni cu semnal intermitent)
    - Se numara doar ferestrele care au CEL PUTIN un criteriu confirmat (nu adunare dubla)
    """
    df = df.copy()

    spo2_flag = pd.Series(False, index=df.index)
    amp_flag  = pd.Series(False, index=df.index)

                                                       
                                                                   
                                                                 
    if "spo2_estimated" in df.columns:
        spo2 = pd.to_numeric(df["spo2_estimated"], errors="coerce")
        quality_mask = pd.Series(True, index=df.index)
        if "contact_valid_ratio" in df.columns:
            cvr = pd.to_numeric(df["contact_valid_ratio"], errors="coerce")
            quality_mask = cvr >= 0.7
        if "bpm_mean" in df.columns:
            bpm = pd.to_numeric(df["bpm_mean"], errors="coerce")
            quality_mask = quality_mask & (bpm > 0)
        spo2_good = spo2[quality_mask].dropna()
        spo2_good = spo2_good[spo2_good > 85.0]
        if len(spo2_good) >= 3:
            spo2_baseline = float(spo2_good.median())
            desat = spo2_baseline - spo2
            raw_desat = (desat >= 4.0) & quality_mask & (spo2 > 85.0)
            raw_desat = raw_desat.fillna(False).values
                                                                                    
            consec = np.zeros(len(raw_desat), dtype=bool)
            for i in range(len(raw_desat)):
                if raw_desat[i]:
                    has_neighbor = (i > 0 and raw_desat[i-1]) or                                   (i < len(raw_desat)-1 and raw_desat[i+1])
                    consec[i] = has_neighbor
            spo2_flag = pd.Series(consec, index=df.index)
            print(f"[APNEA] SpO2 desat >= 4% sub {spo2_baseline:.0f}%: "
                  f"{raw_desat.sum()} brut → {consec.sum()} consecutive")

                                                                           
                                                                                
                                                                               
    if "resp_amplitude" in df.columns and "breath_valid_ratio" in df.columns:
        amp = pd.to_numeric(df["resp_amplitude"], errors="coerce")
        bvr = pd.to_numeric(df["breath_valid_ratio"], errors="coerce")

                                                             
        amp_working = amp[bvr > 0.3].dropna()
        if len(amp_working) >= 5:
            amp_baseline = float(amp_working.quantile(0.75))                  
            if amp_baseline > 30:
                                                                                    
                low_amp_mask = (amp < amp_baseline * 0.15) & (bvr > 0.3)
                amp_flag = low_amp_mask.fillna(False)
                n_low = amp_flag.sum()
                if n_low > 0:
                    print(f"[APNEA] Amplitudine flex scazuta (<{amp_baseline*0.15:.0f} ADC, "
                          f"baseline P75={amp_baseline:.0f}): {n_low} ferestre")

                                                  
    apnea_mask = spo2_flag | amp_flag
    df["apnea_events_simple"] = apnea_mask.astype(float)
    total = apnea_mask.sum()
    print(f"[APNEA] Total evenimente apnee (multi-senzor): {total:.0f} "
          f"(SpO2={spo2_flag.sum()}, Flex={amp_flag.sum()}, ambele={int((spo2_flag & amp_flag).sum())})")

    return df


def preprocess_features(df: pd.DataFrame):
    """
    Curata, imputa si standardizeaza features.
    Returneaza: (df_clean, X_scaled, scaler, feature_cols_used)
    """
    df = df.copy()

                                                                       
                                            
    if "ppg_valid" not in df.columns:
                                                         
        if "contact_valid_ratio" in df.columns:
            df["ppg_valid"] = (df["contact_valid_ratio"].fillna(0) >= 0.5).astype(float)
        else:
            df["ppg_valid"] = 1.0

                                                      
    feat_cols = get_feature_cols(df)

                                              
    for col in feat_cols:
        median_val = df[col].median()
        if col in ["bpm_mean","bpm_std","bpm_min","bpm_max","spo2_estimated"]:
                                                                 
            df[col] = df[col].fillna(0)
        elif col == "hrv_simple":
                                                                                           
                                                                                    
                                                                               
                                                                                     
            if not np.isnan(median_val) and median_val > 0:
                df[col] = df[col].fillna(median_val)
                                                                         
        else:
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

                                                  
    df = df.dropna(subset=feat_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError("Nu exista date suficiente dupa curatare!")

    X = df[feat_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

                                                               
                                                                            
                                                                              
                                                                            
    weights = {
        "bpm_mean":            1.0,                                  
        "bpm_std":             1.8,                                    
        "bpm_min":             1.0,
        "bpm_max":             1.0,
        "hrv_simple":          1.5,                            
        "spo2_estimated":      1.2,
        "motion_level":        2.5,                      
        "motion_per_min":      2.0,
        "motion_variation":    0.5,                                                               
        "gyro_energy":         1.5,
        "acc_jerk":            2.5,                                    
        "dominant_position":   0.3,                               
        "resp_rate":           1.8,
        "resp_variability":    1.5,
        "resp_amplitude":      1.0,
        "apnea_events_simple": 1.0,
        "breath_valid_ratio":  0.5,
        "temp_mean":           0.8,                          
        "temp_variation":      0.5,
                                 
                                                                            
                                                                
    }
    weight_arr = np.array([weights.get(c, 1.0) for c in feat_cols])
    X_scaled = X_scaled * weight_arr
    print(f"[PREP] Feature weighting rebalansat (motion x2.5 dominant, BPM x1.0 redus)")

    print(f"[PREP] {len(df)} ferestre valide | {len(feat_cols)} features")
    return df, X_scaled, scaler, feat_cols

                                               
                         
                                               

                                                                
STAGE_LABELS = {
    0: "Light Sleep",
    1: "Deep Sleep",
    2: "REM Sleep",
    3: "Wake / Movement"
}

STAGE_COLORS = {
    "Light Sleep":         "#4FC3F7",
    "Deep Sleep":          "#1565C0",
    "REM Sleep":           "#AB47BC",
    "Wake / Movement":     "#EF5350",
    "Invalid / No Contact": "#444444"                                    
}


def assign_stage_labels(df: pd.DataFrame, cluster_centers, feat_cols: list) -> pd.DataFrame:
    """
    Atribuie etichete in DOUA FAZE:

    FAZA 1 - LABELING INITIAL pe cluster (pe motion):
      - Cluster cu motion cel mai mare = Wake
      - Restul = "SleepCandidate" (vor fi subclasificate in faza 2)

    FAZA 2 - SUBCLASIFICARE PE FEREASTRA cu percentile relative la sesiune:
      Praguri calculate DIN DATELE SESIUNII (nu absolute):
        - Deep    = BPM in percentila inferioara (<=P40) + motion mica + bpm_std < P45
        - REM     = BPM in percentila superioara (>=P65) SI bpm_std >= P55
                    (ambele trebuie ridicate — OR singur supraevalua REM)
        - Light   = restul
      Cap fiziologic: Deep max 25%, REM max 30%.
    """
    df = df.copy()

                                                
    ids = sorted([i for i in df["kmeans_cluster"].unique() if i != -1])

    cluster_motion = {}
    for i in ids:
        sub = df[df["kmeans_cluster"] == i]
        cluster_motion[i] = float(sub["motion_level"].median()) if len(sub) > 0 else 0.0

    wake_cluster_id = max(cluster_motion, key=cluster_motion.get)

                                          
    df["sleep_stage"] = "SleepCandidate"
                                    
    df.loc[df["kmeans_cluster"] == wake_cluster_id, "sleep_stage"] = "Wake / Movement"

    n_wake = (df["sleep_stage"] == "Wake / Movement").sum()
    n_sleep = (df["sleep_stage"] == "SleepCandidate").sum()
    print(f"[LABELS] Faza 1 (pe cluster): Wake={n_wake}, SleepCandidate={n_sleep}")

                                                         
    sleep_mask = df["sleep_stage"] == "SleepCandidate"
    if sleep_mask.sum() == 0:
        return df

    sleep_df = df[sleep_mask]

                                                                         
    bpm_valid_mask = sleep_df["bpm_mean"] > 0
    bpm_valid = sleep_df.loc[bpm_valid_mask, "bpm_mean"]
    bpm_std_valid = sleep_df.loc[bpm_valid_mask, "bpm_std"]

                                                   
    motion_p25 = float(sleep_df["motion_level"].quantile(0.25))
    motion_p50 = float(sleep_df["motion_level"].quantile(0.50))
    motion_p95 = float(sleep_df["motion_level"].quantile(0.95))
    wake_motion_threshold = max(motion_p95, 0.05)

                                        
    if len(bpm_valid) >= 10:
        bpm_p40 = float(bpm_valid.quantile(0.40))
        bpm_p65 = float(bpm_valid.quantile(0.65))
        bpm_std_p45 = float(bpm_std_valid.quantile(0.45))
        bpm_std_p55 = float(bpm_std_valid.quantile(0.55))
    else:
                                              
        bpm_p40, bpm_p65 = 58, 68
        bpm_std_p45, bpm_std_p55 = 8, 15

    print(f"[LABELS] Thresholds: motion_P25={motion_p25:.4f}, "
          f"motion_P50={motion_p50:.4f}, wake_motion>={wake_motion_threshold:.4f}")
    print(f"[LABELS] BPM percentile: P40={bpm_p40:.1f}, P65={bpm_p65:.1f}, "
          f"bpm_std P45={bpm_std_p45:.1f}, P55={bpm_std_p55:.1f}")

    def classify_row(row):
        bm = row["bpm_mean"] if not np.isnan(row["bpm_mean"]) else 0
        bs = row["bpm_std"] if not np.isnan(row["bpm_std"]) else 0
        m  = row["motion_level"] if not np.isnan(row["motion_level"]) else 0.05
        mv = row["motion_variation"] if not np.isnan(row.get("motion_variation", float("nan"))) else 0
        hrv = row["hrv_simple"] if "hrv_simple" in row and not np.isnan(row["hrv_simple"]) else 0

                                                        
        if bm <= 0:
            return "Light Sleep"               

                                                           
        if m >= wake_motion_threshold and mv > 0.01:
            return "Wake / Movement"

                                                                             
        if bm <= bpm_p40 and m <= motion_p50 and bs <= bpm_std_p45:
            return "Deep Sleep"

                                                                            
                                                                      
        if bm >= bpm_p65 and bs >= bpm_std_p55:
            return "REM Sleep"

                             
        return "Light Sleep"

    new_labels = sleep_df.apply(classify_row, axis=1)
    df.loc[sleep_mask, "sleep_stage"] = new_labels

                                                           
    n_total = sleep_mask.sum()
    n_deep_current = (df["sleep_stage"] == "Deep Sleep").sum()

    max_deep_allowed = int(n_total * 0.25)

    if n_deep_current > max_deep_allowed:
        n_to_move = n_deep_current - max_deep_allowed
        deep_rows = df[df["sleep_stage"] == "Deep Sleep"].sort_values(
            "motion_level", ascending=False
        )
        idx_to_move = deep_rows.index[:n_to_move]
        df.loc[idx_to_move, "sleep_stage"] = "Light Sleep"
        print(f"[LABELS] Cap Deep 25%: {n_deep_current} -> {max_deep_allowed} "
              f"({n_to_move} ferestre mutate la Light)")

                                                          
    n_rem_current = (df["sleep_stage"] == "REM Sleep").sum()
    max_rem_allowed = int(n_total * 0.30)
    if n_rem_current > max_rem_allowed:
        n_to_move = n_rem_current - max_rem_allowed
                                                               
                                     
        rem_rows = df[df["sleep_stage"] == "REM Sleep"].sort_values(
            "bpm_std", ascending=True
        )
        idx_to_move = rem_rows.index[:n_to_move]
        df.loc[idx_to_move, "sleep_stage"] = "Light Sleep"
        print(f"[LABELS] Cap REM 30%: {n_rem_current} -> {max_rem_allowed} "
              f"({n_to_move} ferestre mutate la Light)")

                            
    final_dist = df["sleep_stage"].value_counts()
    print("[LABELS] Distributie dupa subclasificare:")
    for stage, cnt in final_dist.items():
        pct = cnt / len(df) * 100
        print(f"         {stage:<22}: {cnt:>4} ({pct:.1f}%)")

                                                                             
    remaining = (df["sleep_stage"] == "SleepCandidate").sum()
    if remaining > 0:
        df.loc[df["sleep_stage"] == "SleepCandidate", "sleep_stage"] = "Light Sleep"
        print(f"[LABELS] Safety: {remaining} SleepCandidate -> Light Sleep")

    return df


def run_kmeans(df: pd.DataFrame, X_scaled: np.ndarray, feat_cols: list,
               n_clusters: int = N_CLUSTERS):
    """Ruleaza K-Means si atribuie etapele de somn.
    - Exclude ferestrele cu ppg_valid=0 din fit (altfel trage centroidul)
    - Ferestrele invalide primesc eticheta "Invalid / No Contact"
    - Daca N_CLUSTERS=0 sau 'auto', testeaza 2-6 clustere cu silhouette
    """
    df = df.copy()

                                               
    if "ppg_valid" in df.columns:
        valid_mask = df["ppg_valid"].fillna(0) > 0.5
    else:
                                                 
        valid_mask = df.get("contact_valid_ratio", pd.Series([1]*len(df))).fillna(0) > 0.3

    n_valid = int(valid_mask.sum())
    n_invalid = len(df) - n_valid
    print(f"[KMEANS] Ferestre valide: {n_valid}/{len(df)} | invalide (PPG deconectat): {n_invalid}")

    if n_valid < 20:
        print(f"[KMEANS] AVERTISMENT: prea putine ferestre valide ({n_valid}) - ruleaza pe tot setul")
        valid_mask = pd.Series([True]*len(df), index=df.index)
        n_valid = len(df)

    X_valid = X_scaled[valid_mask.values]

                                        
    if n_clusters == 0 or str(n_clusters).lower() == "auto":
        best_k, best_sil, best_km, best_labels_valid = 4, -1, None, None
        max_k = min(6, n_valid // 2)
        print("[KMEANS] Selectie automata N_CLUSTERS pe ferestre valide...")
        for k in range(2, max_k + 1):
            km_try = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            lbl = km_try.fit_predict(X_valid)
            if len(set(lbl)) < 2:
                continue
            try:
                sil = silhouette_score(X_valid, lbl)
            except Exception:
                sil = -1
            print(f"         k={k}: silhouette={sil:.3f}")
            if sil > best_sil:
                best_sil, best_k, best_km, best_labels_valid = sil, k, km_try, lbl
        n_clusters = best_k
        km = best_km
        labels_valid = best_labels_valid
        print(f"[KMEANS] Ales k={n_clusters} (silhouette={best_sil:.3f})")
    else:
                        
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
        labels_valid = km.fit_predict(X_valid)

                                                 
    labels = np.full(len(df), -1, dtype=int)
    labels[valid_mask.values] = labels_valid
    df["kmeans_cluster"] = labels

                                                     
    if len(set(labels_valid)) > 1 and len(labels_valid) > len(set(labels_valid)):
        try:
            sil = silhouette_score(X_valid, labels_valid)
            print(f"[KMEANS] Silhouette score final: {sil:.3f}  (>0.5 = bun)")
        except Exception:
            print("[KMEANS] Silhouette score: N/A")

    df = assign_stage_labels(df, km.cluster_centers_, feat_cols)

                                                              
    df.loc[df["kmeans_cluster"] == -1, "sleep_stage"] = "Invalid / No Contact"
    n_invalid_final = (df["sleep_stage"] == "Invalid / No Contact").sum()
    if n_invalid_final > 0:
        print(f"[KMEANS] {n_invalid_final} ferestre marcate 'Invalid / No Contact' (excluse din scor)")

                                                                           
                                                                     
                                                  
    df["sleep_stage_raw"] = df["sleep_stage"].copy()
    df["sleep_stage"] = smooth_hypnogram(
        df["sleep_stage"], window=3,
        preserve_labels=["Invalid / No Contact", "Wake / Movement"]
    )
    n_changed = (df["sleep_stage"] != df["sleep_stage_raw"]).sum()
    print(f"[SMOOTH] {n_changed} ferestre netezite (tranzitii izolate eliminate)")

                                        
    dist = df["sleep_stage"].value_counts()
    print("[KMEANS] Distributie etape somn (dupa smoothing):")
    for stage, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"         {stage:<20}: {cnt:>4} ferestre ({pct:.1f}%)")

    return df, km

                                               
                      
                                               

def run_knn(df: pd.DataFrame, X_scaled: np.ndarray, k: int = 5):
    """
    KNN auto-antrenat pe labelele K-Means.
    Adauga o coloana 'knn_stage' cu predictia KNN (mai stabila la margini).
    """
    y = df["sleep_stage"].values
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
    knn.fit(X_scaled, y)
    df = df.copy()
    df["knn_stage"] = knn.predict(X_scaled)

                                    
    proba = knn.predict_proba(X_scaled)
    classes = knn.classes_
    for i, cls in enumerate(classes):
        safe_name = cls.replace(" ", "_").replace("/","_")
        df[f"prob_{safe_name}"] = proba[:, i]

    print(f"[KNN]  k={k} | antrenat pe {len(df)} ferestre")
    return df, knn

                                               
                                
                                               

def compute_sleep_score(df: pd.DataFrame) -> dict:
    """
    Calculeaza un scor global al calitatii somnului (0-100)
    pe baza a 6 componente ponderate.
    EXCLUDE ferestrele 'Invalid / No Contact' din calcul.
    """
    scores = {}
    details = {}

    total_windows_all = len(df)
    if total_windows_all == 0:
        return {"total_score": 0, "components": {}}

                                                           
    df_valid = df[df["sleep_stage"] != "Invalid / No Contact"].copy()
    total_windows = len(df_valid)
    n_invalid = total_windows_all - total_windows
    if n_invalid > 0:
        print(f"[SCORE] Excluse {n_invalid} ferestre 'Invalid / No Contact' din scoring")

    if total_windows == 0:
        return {"total_score": 0, "components": {}, "details": {"error": "nicio fereastra valida"}}

                                                    
    stage_pct = df_valid["sleep_stage"].value_counts(normalize=True).to_dict()
                                                           
    df = df_valid

                                                     
                                                                    
    deep_pct  = stage_pct.get("Deep Sleep", 0)
    rem_pct   = stage_pct.get("REM Sleep", 0)
    light_pct = stage_pct.get("Light Sleep", 0)
    wake_pct  = stage_pct.get("Wake / Movement", 0)

    arch_score  = min(deep_pct  / 0.25, 1.0) * 10                   
    arch_score += min(rem_pct   / 0.25, 1.0) * 10                  
    arch_score += max(0, 1 - wake_pct / 0.15) * 10                   
    scores["architecture"] = round(arch_score, 1)
    details["architecture"] = f"Deep={deep_pct*100:.0f}%, REM={rem_pct*100:.0f}%, Wake={wake_pct*100:.0f}%"

                                                 
    ppg_score = 0
    bpm_avg = None
    spo2 = None
    if "bpm_mean" in df.columns:
        bpm_series = df["bpm_mean"]
        bpm_series = bpm_series[bpm_series > 0]                                           
        if len(bpm_series) > 0:
            bpm_avg = float(bpm_series.median())
            if 45 <= bpm_avg <= 70:
                ppg_score += 10
            elif 35 <= bpm_avg <= 85:
                ppg_score += 6
            else:
                ppg_score += 2
    if "spo2_estimated" in df.columns:
        spo2_series = df["spo2_estimated"]
        spo2_series = spo2_series[spo2_series > 0]
        if len(spo2_series) > 0:
            spo2 = float(spo2_series.median())
            if spo2 >= 95:
                ppg_score += 10
            elif spo2 >= 90:
                ppg_score += 6
            else:
                ppg_score += 2
    scores["ppg_quality"] = round(ppg_score, 1)
    bpm_str  = f"{bpm_avg:.0f}" if bpm_avg is not None else "N/A"
    spo2_str = f"{spo2:.0f}%"   if spo2    is not None else "N/A"
    details["ppg_quality"] = f"BPM median={bpm_str}, SpO2 median={spo2_str}"

                                               
    resp_score = 0
    rr = None
    apnea_total = 0
    if "resp_rate" in df.columns:
        rr_series = df["resp_rate"].dropna()
        if len(rr_series) > 0:
            rr = float(rr_series.median())
            if 10 <= rr <= 18:
                resp_score += 10
            elif 8 <= rr <= 22:
                resp_score += 6
            else:
                resp_score += 2
    if "apnea_events_simple" in df.columns:
        apnea_total = float(df["apnea_events_simple"].sum())
                                                                           
                                                                   
        apnea_pct = apnea_total / max(len(df), 1) * 100
        if apnea_pct < 2:
            apnea_score = 10
        elif apnea_pct < 5:
            apnea_score = 7
        elif apnea_pct < 15:
            apnea_score = 4
        else:
            apnea_score = 1
        resp_score += apnea_score
    scores["respiration"] = round(resp_score, 1)
    rr_str = f"{rr:.1f}" if rr is not None else "N/A"
    details["respiration"] = f"RR median={rr_str}, Apnee={apnea_total:.0f}"

                                            
    mov_score = 0
    motion_avg = 0.0
    if "motion_level" in df.columns:
        motion_series = df["motion_level"].dropna()
        if len(motion_series) > 0:
            motion_avg = float(motion_series.median())
            if motion_avg < 0.05:
                mov_score = 15
            elif motion_avg < 0.15:
                mov_score = 10
            elif motion_avg < 0.30:
                mov_score = 5
            else:
                mov_score = 0
    scores["movement"] = round(mov_score, 1)
    details["movement"] = f"Motion median={motion_avg:.3f}"

                                        
    hrv_score = 0
    hrv_avg = 0.0
    if "hrv_simple" in df.columns:
        hrv_vals = df["hrv_simple"][(df["hrv_simple"] > 0) & df["hrv_simple"].notna()]
        hrv_avg = float(hrv_vals.mean()) if len(hrv_vals) > 0 else 0.0
        if len(hrv_vals) == 0:
            hrv_score = 5
            details["hrv"] = "HRV indisponibil (date PPG insuficiente)"
        elif 30 <= hrv_avg <= 100:
            hrv_score = 10
            details["hrv"] = f"HRV index={hrv_avg:.1f}ms (bun)"
        elif 15 <= hrv_avg <= 150:
            hrv_score = 6
            details["hrv"] = f"HRV index={hrv_avg:.1f}ms (moderat)"
        else:
            hrv_score = 3
            details["hrv"] = f"HRV index={hrv_avg:.1f}ms (scazut)"

    scores["hrv"] = round(hrv_score, 1)
                            
                            
                                                             
                                         
                                        
                                                  
    snore_score = 5.0
    total_snore = 0
    if "snore_events_count" in df.columns:
        snore_series = df["snore_events_count"].fillna(0)
        total_snore = int(snore_series.sum())
        if total_snore == 0:
            snore_score = 5.0
        elif total_snore <= 20:
            snore_score = 4.0
        elif total_snore <= 50:
            snore_score = 3.0
        elif total_snore <= 100:
            snore_score = 2.0
        else:
            snore_score = 1.0
    scores["snoring"] = round(snore_score, 1)
    details["snoring"] = f"Evenimente sforait total={total_snore}"

                                  
                                          
                                      
                                                                    
    duration_min = total_windows * WINDOW_SEC / 60
    if duration_min < 240:                 
        duration_penalty = 15
    elif duration_min < 360:             
        duration_penalty = round((360 - duration_min) / 12, 1)
    else:
        duration_penalty = 0                             

    if duration_penalty > 0:
        scores["duration_penalty"] = -duration_penalty
        details["duration_penalty"] = f"Durata={duration_min:.0f}min (sub 6h = -{duration_penalty}pt)"

                      
    total = sum(scores.values())
    total = round(min(max(total, 0), 100), 1)

                                                            
    scores.pop("duration_penalty", None)

                       
    if total >= 85:
        grade = "Excelent 🌟"
    elif total >= 70:
        grade = "Bun 👍"
    elif total >= 55:
        grade = "Mediu ⚠️"
    elif total >= 40:
        grade = "Slab 😴"
    else:
        grade = "Foarte slab ❌"

    result = {
        "total_score": total,
        "grade": grade,
        "components": scores,
        "details": details,
        "session_duration_min": round(total_windows * WINDOW_SEC / 60, 1),
        "session_duration_total_min": round(total_windows_all * WINDOW_SEC / 60, 1),
        "total_windows": total_windows,
        "total_windows_all": total_windows_all,
        "valid_ppg_ratio": round(total_windows / total_windows_all, 3) if total_windows_all else 0,
        "stage_distribution_pct": {k: round(v*100, 1) for k, v in stage_pct.items()}
    }

    print(f"\n{'='*50}")
    print(f"  SCOR CALITATE SOMN: {total}/100  –  {grade}")
    print(f"  Durata sesiune: {result['session_duration_min']} minute")
    print(f"{'='*50}")
    for comp, val in scores.items():
        print(f"  {comp:<20}: {val:>5.1f} pt  |  {details[comp]}")
    print(f"{'='*50}\n")

    return result

                                               
             
                                               

def smooth_hypnogram(stages_series, window=3, preserve_labels=None):
    """Netezeste hypnogram-ul in doua pasaje.

    Pas 1 — Elimina ferestre izolate (singure intre doi vecini identici):
      [Light, REM, Light] → [Light, Light, Light]
      [Light, REM, REM]   → neschimbat
      Deep Sleep NU se sterge in Pas 1 — chiar si e izolat poate fi real.

    Pas 2 — Elimina blocuri prea scurte (< 2 ferestre = 1 min) pt Light/REM/Wake.
      Deep Sleep NU se sterge in Pas 2 — fiziologic o fereastra de 30s Deep e reala.

    preserve_labels: etichete care NU se modifica niciodata.
    """
    PROTECT_STAGES = {"Deep Sleep"}                                              
                                                                               
                                                                             
                                                               
    MIN_BLOCK = 2                                       

    if preserve_labels is None:
        preserve_labels = ["Invalid / No Contact"]

    stages = stages_series.tolist()

                                   
                                                                                
                                                                          
                                                                          
                                                                                         
    smoothed = stages.copy()
    for i in range(1, len(stages) - 1):
        if stages[i] in preserve_labels:
            continue
        left, right = stages[i-1], stages[i+1]
        if left in preserve_labels or right in preserve_labels:
            continue
        if left == right and left != stages[i]:
            smoothed[i] = left

                                                      
    def get_blocks(s):
        blocks = []
        if not s:
            return blocks
        start = 0
        for i in range(1, len(s)):
            if s[i] != s[i-1]:
                blocks.append((s[i-1], start, i-1))
                start = i
        blocks.append((s[-1], start, len(s)-1))
        return blocks

    changed = True
    while changed:
        changed = False
        blocks = get_blocks(smoothed)
        for b_idx, (label, bstart, bend) in enumerate(blocks):
            if label in preserve_labels:
                continue
            if label in PROTECT_STAGES:
                continue                      
            blen = bend - bstart + 1
            if blen < MIN_BLOCK:
                                                                     
                left_label  = blocks[b_idx-1][0] if b_idx > 0 else None
                right_label = blocks[b_idx+1][0] if b_idx < len(blocks)-1 else None
                                                     
                left_len  = (blocks[b_idx-1][2] - blocks[b_idx-1][1] + 1) if left_label else 0
                right_len = (blocks[b_idx+1][2] - blocks[b_idx+1][1] + 1) if right_label else 0
                if left_label in preserve_labels: left_len = 0
                if right_label in preserve_labels: right_len = 0
                replacement = left_label if left_len >= right_len else right_label
                if replacement and replacement != label:
                    for i in range(bstart, bend+1):
                        smoothed[i] = replacement
                    changed = True
                    break                           

    return pd.Series(smoothed, index=stages_series.index)


def plot_hypnogram(df: pd.DataFrame, score_result: dict, session_id: str):
    """Hypnogram + scor + distributie etape."""
    stage_order = ["Wake / Movement", "Light Sleep", "REM Sleep", "Deep Sleep"]
    stage_y = {s: i for i, s in enumerate(stage_order)}

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

                        
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0D1117")

    times = df["window_start"]
                                                                    
    stages_raw    = df["sleep_stage"]
    stages_smooth = smooth_hypnogram(stages_raw, window=9)
    stages = stages_smooth.map(lambda s: stage_y.get(s, 1))

                                                  
    for i in range(len(df) - 1):
        stage = stages_smooth.iloc[i]
        color = STAGE_COLORS.get(stage, "#888888")
        ax1.fill_between([times.iloc[i], times.iloc[i+1]],
                         [stages.iloc[i]] * 2, 0,
                         color=color, alpha=0.4)

    ax1.step(times, stages, color="white", linewidth=1.2, where="post")
    ax1.set_yticks(range(len(stage_order)))
    ax1.set_yticklabels(stage_order, color="white", fontsize=9)
    ax1.set_xlabel("Timp", color="gray", fontsize=9)
    ax1.set_title(f"Hypnogram – Sesiune {session_id}", color="white", fontsize=12, fontweight="bold")
    ax1.tick_params(colors="gray")
    ax1.spines[:].set_color("#333333")
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    plt.setp(ax1.get_xticklabels(), color="gray", fontsize=8)

                          
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0D1117")
    if "bpm_mean" in df.columns:
        ax2.plot(times, df["bpm_mean"], color="#4FC3F7", linewidth=1.0, label="BPM")
        ax2.fill_between(times, df["bpm_mean"], alpha=0.15, color="#4FC3F7")
    ax2.set_title("Puls (BPM)", color="white", fontsize=10)
    ax2.set_ylabel("BPM", color="gray", fontsize=8)
    ax2.tick_params(colors="gray")
    ax2.spines[:].set_color("#333333")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    plt.setp(ax2.get_xticklabels(), color="gray", fontsize=7, rotation=30)

                                 
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0D1117")
    if "resp_rate" in df.columns:
        ax3.plot(times, df["resp_rate"], color="#66BB6A", linewidth=1.0)
        ax3.fill_between(times, df["resp_rate"], alpha=0.15, color="#66BB6A")
    ax3.set_title("Rata Respiratorie (resp/min)", color="white", fontsize=10)
    ax3.set_ylabel("Resp/min", color="gray", fontsize=8)
    ax3.tick_params(colors="gray")
    ax3.spines[:].set_color("#333333")
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    plt.setp(ax3.get_xticklabels(), color="gray", fontsize=7, rotation=30)

                                      
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#0D1117")
    comp = score_result["components"]
    comp_colors = ["#4FC3F7","#AB47BC","#66BB6A","#FFA726","#EF5350","#78909C"]
    wedges, texts, autotexts = ax4.pie(
        list(comp.values()),
        labels=[k.replace("_"," ").title() for k in comp.keys()],
        colors=comp_colors[:len(comp)],
        autopct="%1.0f%%",
        textprops={"color": "white", "fontsize": 7},
        startangle=90
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax4.set_title(f"Componente Scor\nTotal: {score_result['total_score']}/100 – {score_result['grade']}",
                  color="white", fontsize=10)

                                      
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#0D1117")
    dist = score_result["stage_distribution_pct"]
    bars = ax5.bar(
        dist.keys(), dist.values(),
        color=[STAGE_COLORS.get(k, "#888") for k in dist.keys()],
        edgecolor="#333333", width=0.6
    )
    for bar, val in zip(bars, dist.values()):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", va="bottom", color="white", fontsize=8)
    ax5.set_ylabel("% din sesiune", color="gray", fontsize=8)
    ax5.set_title("Distributie Etape Somn", color="white", fontsize=10)
    ax5.tick_params(colors="gray")
    ax5.spines[:].set_color("#333333")
    plt.setp(ax5.get_xticklabels(), color="white", fontsize=7, rotation=20, ha="right")

    out_path = os.path.join(OUTPUT_DIR, f"hypnogram_{session_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[PLOT] Hypnogram salvat: {out_path}")


def plot_features_pca(df: pd.DataFrame, X_scaled: np.ndarray, feat_cols: list, session_id: str):
    """PCA 2D al tuturor ferestrelor, colorat dupa etapa de somn.
    Aplica clipping pe P99 pentru a elimina outlieri extremi care
    tragoanele axele si fac clusterele invizibile.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

                                                              
    x_lo, x_hi = np.percentile(X_pca[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(X_pca[:, 1], [1, 99])
                                     
    x_margin = (x_hi - x_lo) * 0.1
    y_margin = (y_hi - y_lo) * 0.1
    ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
    ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

                                                                              
                                              
    plot_order = ["Invalid / No Contact", "Wake / Movement",
                  "Light Sleep", "Deep Sleep", "REM Sleep"]
    for stage in plot_order:
        mask = df["sleep_stage"] == stage
        if mask.sum() == 0:
            continue
                                                                                     
        alpha = 0.25 if stage == "Invalid / No Contact" else 0.65
        size  = 12   if stage == "Invalid / No Contact" else 25
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=STAGE_COLORS.get(stage, "#888"),
                   label=f"{stage} (n={mask.sum()})",
                   alpha=alpha, s=size, edgecolors="none")

                                                      
    n_outliers = int(((X_pca[:, 0] < x_lo) | (X_pca[:, 0] > x_hi) |
                      (X_pca[:, 1] < y_lo) | (X_pca[:, 1] > y_hi)).sum())

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color="gray")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color="gray")
    title = "PCA – Clustere Etape Somn"
    if n_outliers > 0:
        title += f"   ({n_outliers} outlieri in afara)"
    ax.set_title(title, color="white", fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
              fontsize=8, loc="best")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333333")
    ax.grid(True, alpha=0.1, color="#333333", linestyle="--")

    out_path = os.path.join(OUTPUT_DIR, f"pca_{session_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[PLOT] PCA salvat: {out_path}")


def plot_motion_snore(df: pd.DataFrame, session_id: str):
    """Miscare + semnal microfon cu evenimente sforait evidentiate."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.patch.set_facecolor("#0D1117")

    times = df["window_start"]

                           
    ax1 = axes[0]
    ax1.set_facecolor("#0D1117")
    if "motion_level" in df.columns:
        ax1.plot(times, df["motion_level"], color="#FFA726", linewidth=1.0)
        ax1.fill_between(times, df["motion_level"], alpha=0.2, color="#FFA726")
    ax1.set_ylabel("Motion Level (g)", color="gray", fontsize=9)
    ax1.set_title("Miscare + Sforait in Timp", color="white", fontsize=11)
    ax1.tick_params(colors="gray")
    ax1.spines[:].set_color("#333333")

                                                                             
    ax2 = axes[1]
    ax2.set_facecolor("#0D1117")

    if "snore_events_count" in df.columns:
        snore_mask = df["snore_events_count"].fillna(0) > 0

                                                                       
        if "mic_rms_mean" in df.columns:
            rms_signal = df["mic_rms_mean"].fillna(0)
            ax2.plot(times, rms_signal, color="#EF9A9A", linewidth=0.8,
                     alpha=0.5, label="Nivel sunet MIC")
            ax2.fill_between(times, rms_signal, alpha=0.15, color="#EF9A9A")

                               
            valid_rms = rms_signal[rms_signal > 5]
            if len(valid_rms) > 10:
                p50 = float(valid_rms.median())
                threshold_line = max(valid_rms.quantile(0.97), p50 * 3.0, 150.0)
                ax2.axhline(y=threshold_line, color="#FF7043", linewidth=1.0,
                            linestyle="--", alpha=0.8,
                            label=f"Prag sforait ({threshold_line:.0f} RMS)")

                                                  
        if snore_mask.any():
            snore_heights = df.loc[snore_mask, "snore_intensity"].fillna(
                df.loc[snore_mask, "mic_rms_mean"] if "mic_rms_mean" in df.columns else 50
            )
            ax2.bar(times[snore_mask], snore_heights,
                    width=pd.Timedelta(seconds=25),
                    color="#EF5350", alpha=0.9, label="Sforait detectat", zorder=3)

        ax2.set_ylabel("Intensitate Sunet (RMS)", color="gray", fontsize=9)
        ax2.legend(loc="upper right", fontsize=7,
                   facecolor="#1A1F2E", labelcolor="white", framealpha=0.7)
    else:
        ax2.set_ylabel("Semnal Microfon", color="gray", fontsize=9)

    ax2.tick_params(colors="gray")
    ax2.spines[:].set_color("#333333")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    plt.setp(ax2.get_xticklabels(), color="gray", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"motion_snore_{session_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[PLOT] Motion+Sforait salvat: {out_path}")

                                               
                
                                               

def export_features_csv(df: pd.DataFrame, session_id: str):
    """Salveaza features + etape intr-un CSV."""
    out_path = os.path.join(OUTPUT_DIR, f"features_{session_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"[CSV]  Features exportate: {out_path}")
    return out_path


def export_score_json(score_result: dict, session_id: str):
    """Salveaza scorul calitate somn intr-un JSON."""
    out_path = os.path.join(OUTPUT_DIR, f"score_{session_id}.json")
    with open(out_path, "w") as f:
        json.dump(score_result, f, indent=2, default=str)
    print(f"[JSON] Scor exportat: {out_path}")
    return out_path

                                               
                         
                                               

def analyze_session(db_path: str, session_id: str = None):
    """
    Ruleaza pipeline-ul complet pentru o sesiune de somn.
    Daca session_id este None, proceseaza toate sesiunile disponibile.
    """
    print(f"\n{'='*60}")
    print(f"  SLEEP ANALYSIS PIPELINE")
    print(f"  DB: {db_path} | Sesiune: {session_id or 'TOATE'}")
    print(f"  Fereastra: {WINDOW_SEC}s | Clustere: {N_CLUSTERS}")
    print(f"{'='*60}\n")

                       
    raw_df = load_raw_data(db_path, session_id)
    mic_df = load_mic_data(db_path, session_id)

    if raw_df.empty:
        print("EROARE: Nu exista date in baza de date!")
        return

                                                                   
    if session_id is None:
        sessions = get_sessions(db_path)
        if not sessions:
            print("EROARE: Nu exista sesiuni in DB!")
            return
        session_id = sessions[0]
        print(f"Auto-selectata sesiunea: {session_id}")
        raw_df = load_raw_data(db_path, session_id)
        mic_df = load_mic_data(db_path, session_id)

                                           
    global FS_FLEX
    if len(raw_df) >= 3:
        _diffs = raw_df["pc_timestamp"].diff().dt.total_seconds().dropna()
        _median_int = float(_diffs.median())
        if _median_int > 0:
            FS_FLEX = round(1.0 / _median_int, 2)
    print(f"[FLEX] FS_FLEX detectat: {FS_FLEX} Hz (interval={1/FS_FLEX:.1f}s)")

                        
    windows = split_windows(raw_df, WINDOW_SEC)

                                   
    global _ppg_50hz_df
    ppg_blocks = load_ppg_data(DB_PATH, session_id)
    if not ppg_blocks.empty:
        _ppg_50hz_df = expand_ppg_blocks(ppg_blocks)
    else:
        _ppg_50hz_df = pd.DataFrame()

                                                                 
    global _flex_4hz_df
    flex_blocks = load_flex_data(DB_PATH, session_id)
    if not flex_blocks.empty:
        _flex_4hz_df = expand_flex_blocks(flex_blocks)
                                                                          
        if len(_flex_4hz_df) >= 3:
            _fd = _flex_4hz_df["pc_timestamp"].diff().dt.total_seconds().dropna()
            _mi = float(_fd.median())
            if _mi > 0:
                FS_FLEX = round(1.0 / _mi, 2)
                print(f"[FLEX] FS_FLEX actualizat din flex_data: {FS_FLEX} Hz")
    else:
        _flex_4hz_df = pd.DataFrame()
        print("[FLEX] Tabel flex_data inexistent — folosesc flex din raw_data")
    if not windows:
        print("EROARE: Nu s-au putut crea ferestre de analiza!")
        return

                           
    features_df = extract_all_features(windows, mic_df, raw_df=raw_df)

                                                                          
                                                                
    features_df = detect_apnea_multisensor(features_df)

                     
    df_clean, X_scaled, scaler, feat_cols = preprocess_features(features_df)

                
    df_clean, km_model = run_kmeans(df_clean, X_scaled, feat_cols)

            
    df_clean, knn_model = run_knn(df_clean, X_scaled)

                      
    score_result = compute_sleep_score(df_clean)

                   
    export_features_csv(df_clean, session_id)
    export_score_json(score_result, session_id)

                
    plot_hypnogram(df_clean, score_result, session_id)
    plot_features_pca(df_clean, X_scaled, feat_cols, session_id)
    plot_motion_snore(df_clean, session_id)

    print(f"\n✅  Analiza completa! Fisierele sunt in: ./{OUTPUT_DIR}/")
    return df_clean, score_result


                                               
              
                                               
if __name__ == "__main__":
                
                                                                                 
                                                                              

    db = DB_PATH
    sess = sys.argv[1] if len(sys.argv) > 1 else None

    if not os.path.exists(db):
        print(f"EROARE: Fisierul de baza de date '{db}' nu a fost gasit!")
        print("Modifica variabila DB_PATH din configuratie.")
        sys.exit(1)

    analyze_session(db, sess)
