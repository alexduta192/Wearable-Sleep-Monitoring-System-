"""
raw_data_analysis.py
====================
Analizeaza calitatea datelor brute de la senzori.
Detecteaza automat noise-ul si aplica filtre daca e necesar.

Utilizare:
  python3 raw_data_analysis.py <session_id>
  python3 raw_data_analysis.py subject_01_20260329_221149
"""

import sqlite3
import numpy as np
import pandas as pd
import sys
import os
from scipy import signal
from scipy.stats import kurtosis, skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DB_PATH    = "/home/pi/raw_sensor_data.db"
OUTPUT_DIR = "sleep_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

                                               
                    
                                               

def load_session(session_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM raw_data WHERE session_id=? ORDER BY pc_timestamp ASC",
        conn, params=(session_id,)
    )
    conn.close()
    df["pc_timestamp"] = pd.to_datetime(df["pc_timestamp"])
    print(f"[LOAD] {len(df)} randuri | sesiune: {session_id}")
    return df

                                               
                             
                                               

def analyze_signal_quality(series: pd.Series, name: str, fs: float = 1.0) -> dict:
    """
    Calculeaza metrici de calitate pentru un semnal:
    - SNR (Signal-to-Noise Ratio)
    - % outlieri (>3 sigma)
    - Kurtosis (valori mari = outlieri multi)
    - Flat segments (semnal blocat = senzor deconectat)
    - Missing values
    """
    s = series.dropna().values.astype(float)
    if len(s) < 10:
        return {"status": "INSUFICIENT", "n": len(s)}

    mean_v  = np.mean(s)
    std_v   = np.std(s)
    snr     = abs(mean_v) / std_v if std_v > 0 else 0

                        
    outliers_pct = np.mean(np.abs(s - mean_v) > 3 * std_v) * 100

                                                    
    diffs = np.diff(s)
    flat  = np.sum(np.abs(diffs) < 1e-6) / len(diffs) * 100

                                           
    kurt = float(kurtosis(s))

                    
    missing_pct = series.isna().mean() * 100

                    
    issues = []
    if outliers_pct > 5:  issues.append(f"outlieri {outliers_pct:.1f}%")
    if flat > 20:          issues.append(f"semnal plat {flat:.1f}%")
    if missing_pct > 10:   issues.append(f"missing {missing_pct:.1f}%")
    if std_v < 1e-6:       issues.append("semnal constant")

    if not issues:
        status = "OK"
    elif len(issues) == 1:
        status = "ATENTIE"
    else:
        status = "PROBLEMATIC"

    return {
        "name":         name,
        "n":            len(s),
        "mean":         round(float(mean_v), 3),
        "std":          round(float(std_v), 3),
        "min":          round(float(np.min(s)), 3),
        "max":          round(float(np.max(s)), 3),
        "snr":          round(snr, 2),
        "outliers_pct": round(outliers_pct, 2),
        "flat_pct":     round(flat, 2),
        "kurtosis":     round(kurt, 2),
        "missing_pct":  round(missing_pct, 2),
        "status":       status,
        "issues":       issues,
    }


def apply_filter(series: pd.Series, filter_type: str,
                 lowcut=None, highcut=None, fs=10.0) -> pd.Series:
    """
    Aplica filtru pe semnal:
    - lowpass:  elimina zgomot de inalta frecventa
    - highpass: elimina drift DC
    - bandpass: pastreaza doar banda de interes
    - median:   elimina spike-uri izolate
    - savgol:   netezire Savitzky-Golay (pastreaza forma)
    """
    s = series.ffill().bfill().values.astype(float)
    nyq = fs / 2.0

    if filter_type == "median":
        from scipy.signal import medfilt
        filtered = medfilt(s, kernel_size=5)

    elif filter_type == "savgol":
        from scipy.signal import savgol_filter
        wl = min(11, len(s) // 4 * 2 + 1)
        filtered = savgol_filter(s, window_length=wl, polyorder=3)

    elif filter_type == "lowpass" and highcut:
        hi = min(highcut / nyq, 0.999)
        b, a = signal.butter(3, hi, btype="low")
        filtered = signal.filtfilt(b, a, s)

    elif filter_type == "highpass" and lowcut:
        lo = max(lowcut / nyq, 0.001)
        b, a = signal.butter(3, lo, btype="high")
        filtered = signal.filtfilt(b, a, s)

    elif filter_type == "bandpass" and lowcut and highcut:
        lo = max(lowcut / nyq, 0.001)
        hi = min(highcut / nyq, 0.999)
        b, a = signal.butter(3, [lo, hi], btype="band")
        filtered = signal.filtfilt(b, a, s)

    else:
        filtered = s

    return pd.Series(filtered, index=series.index)


                                               
                                
                                               

def plot_raw_analysis(df: pd.DataFrame, session_id: str):
    """
    Grafic principal cu datele brute + versiuni filtrate + metrici calitate.
    """
                                                     
    df_plot = df.head(min(500, len(df))).copy()
    t = df_plot["pc_timestamp"]
    fs_raw = 10.0                  

    fig = plt.figure(figsize=(18, 20))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#0D1117")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color="gray", fontsize=8)
        ax.tick_params(colors="gray", labelsize=7)
        ax.spines[:].set_color("#333333")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.get_xticklabels(), color="gray", fontsize=7, rotation=20)

                                       
    ax1 = fig.add_subplot(gs[0, :])
    ir_raw = df_plot["ir"].astype(float)
    ir_filt = apply_filter(ir_raw, "bandpass", lowcut=0.5, highcut=3.5, fs=fs_raw)
    ax1.plot(t, ir_raw,  color="#333366", linewidth=0.5, alpha=0.6, label="IR brut")
    ax1.plot(t, ir_filt, color="#4FC3F7", linewidth=1.0, label="IR filtrat (0.5-3.5 Hz)")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    style_ax(ax1, "PPG — Semnal IR (MAX30102): Brut vs Filtrat Bandpass", "IR Value")

    q_ir = analyze_signal_quality(ir_raw, "IR", fs_raw)
    status_color = {"OK": "#66BB6A", "ATENTIE": "#FFA726", "PROBLEMATIC": "#EF5350"}
    ax1.text(0.01, 0.92,
             f"Status: {q_ir['status']} | SNR={q_ir['snr']:.1f} | "
             f"Outlieri={q_ir['outliers_pct']:.1f}% | Flat={q_ir['flat_pct']:.1f}%",
             transform=ax1.transAxes, color=status_color.get(q_ir["status"], "white"),
             fontsize=8, bbox=dict(facecolor="#1a1a2e", alpha=0.7, edgecolor="none"))

                            
    ax2 = fig.add_subplot(gs[1, 0])
    for col, color, lbl in [("acc_x","#EF5350","X"), ("acc_y","#66BB6A","Y"), ("acc_z","#4FC3F7","Z")]:
        ax2.plot(t, df_plot[col].astype(float), color=color, linewidth=0.7, label=lbl, alpha=0.8)
    ax2.axhline(0, color="#444", linewidth=0.5, linestyle="--")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    style_ax(ax2, "Accelerometru (MPU6050) — X/Y/Z (g)", "g")

    q_acc = analyze_signal_quality(df_plot["acc_z"], "AccZ", fs_raw)
    ax2.text(0.01, 0.88, f"AccZ: {q_acc['status']} | std={q_acc['std']:.3f}g",
             transform=ax2.transAxes, color=status_color.get(q_acc["status"], "white"), fontsize=8)

                       
    ax3 = fig.add_subplot(gs[1, 1])
    for col, color, lbl in [("gyro_x","#EF5350","X"), ("gyro_y","#66BB6A","Y"), ("gyro_z","#4FC3F7","Z")]:
        gyro = apply_filter(df_plot[col].astype(float), "median")
        ax3.plot(t, gyro, color=color, linewidth=0.7, label=lbl, alpha=0.8)
    ax3.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    style_ax(ax3, "Giroscop (MPU6050) — X/Y/Z (deg/s) filtrat median", "deg/s")

                          
    ax4 = fig.add_subplot(gs[2, 0])
    flex_raw  = df_plot["flex_raw"].astype(float)
    flex_filt = apply_filter(flex_raw, "savgol")
    ax4.plot(t, flex_raw,  color="#555555", linewidth=0.5, alpha=0.5, label="Brut")
    ax4.plot(t, flex_filt, color="#66BB6A", linewidth=1.2, label="Savitzky-Golay")
    ax4.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    style_ax(ax4, "Flex Sensor (Respiratie) — Brut vs Filtrat", "ADC Value")

    q_flex = analyze_signal_quality(flex_raw, "Flex", fs_raw)
    ax4.text(0.01, 0.88,
             f"Status: {q_flex['status']} | std={q_flex['std']:.1f} | outlieri={q_flex['outliers_pct']:.1f}%",
             transform=ax4.transAxes, color=status_color.get(q_flex["status"], "white"), fontsize=8)

                          
    ax5 = fig.add_subplot(gs[2, 1])
    temp = df_plot["temp_c"].astype(float)
    temp_filt = apply_filter(temp, "lowpass", highcut=0.1, fs=fs_raw)
    ax5.plot(t, temp,      color="#555555", linewidth=0.5, alpha=0.5, label="Brut")
    ax5.plot(t, temp_filt, color="#FFA726", linewidth=1.2, label="Lowpass 0.1Hz")
    ax5.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    style_ax(ax5, "Temperatura Piele (BMP280) — Brut vs Filtrat", "°C")

    q_temp = analyze_signal_quality(temp, "Temp", fs_raw)
    ax5.text(0.01, 0.88,
             f"Status: {q_temp['status']} | mean={q_temp['mean']:.1f}°C | std={q_temp['std']:.3f}",
             transform=ax5.transAxes, color=status_color.get(q_temp["status"], "white"), fontsize=8)

                                                   
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor("#0D1117")
    ir_clean = ir_raw.fillna(ir_raw.median()).values
    if len(ir_clean) > 10:
        freqs = np.fft.rfftfreq(len(ir_clean), d=1.0/fs_raw)
        fft_vals = np.abs(np.fft.rfft(ir_clean - np.mean(ir_clean)))
        ax6.plot(freqs, fft_vals, color="#4FC3F7", linewidth=0.8)
        ax6.axvspan(0.5, 3.5, alpha=0.15, color="#66BB6A", label="Banda BPM (0.5-3.5 Hz)")
        ax6.axvline(1.0, color="#FFA726", linewidth=1, linestyle="--", alpha=0.7, label="~60 BPM")
        ax6.axvline(1.5, color="#EF5350", linewidth=1, linestyle="--", alpha=0.7, label="~90 BPM")
        ax6.set_xlim(0, min(5, freqs.max()))
        ax6.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=7)
    ax6.set_title("Spectru Frecventa — IR (PPG)", color="white", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Frecventa (Hz)", color="gray", fontsize=8)
    ax6.set_ylabel("Amplitudine FFT", color="gray", fontsize=8)
    ax6.tick_params(colors="gray", labelsize=7)
    ax6.spines[:].set_color("#333333")

                                       
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor("#0D1117")
    for col, color, lbl in [("acc_x","#EF5350","AccX"), ("acc_y","#66BB6A","AccY"), ("acc_z","#4FC3F7","AccZ")]:
        vals = df_plot[col].dropna().astype(float)
        ax7.hist(vals, bins=40, alpha=0.5, color=color, label=lbl, density=True)
    ax7.set_title("Distributie Accelerometru", color="white", fontsize=10, fontweight="bold")
    ax7.set_xlabel("Valoare (g)", color="gray", fontsize=8)
    ax7.set_ylabel("Densitate", color="gray", fontsize=8)
    ax7.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
    ax7.tick_params(colors="gray", labelsize=7)
    ax7.spines[:].set_color("#333333")

                                     
    ax8 = fig.add_subplot(gs[4, :])
    ax8.set_facecolor("#0D1117")
    ax8.axis("off")

    signals_to_check = [
        ("ir",       "IR (PPG)",          fs_raw),
        ("red",      "RED (PPG)",         fs_raw),
        ("acc_x",    "Accelerometru X",   fs_raw),
        ("acc_z",    "Accelerometru Z",   fs_raw),
        ("gyro_x",   "Giroscop X",        fs_raw),
        ("flex_raw", "Flex Sensor",       fs_raw),
        ("temp_c",   "Temperatura",       fs_raw),
    ]

    table_data = []
    col_labels = ["Semnal", "N", "Mean", "Std", "SNR", "Outlieri%", "Flat%", "Kurtosis", "Status"]

    for col, name, fs in signals_to_check:
        if col in df.columns:
            q = analyze_signal_quality(df[col], name, fs)
            table_data.append([
                name,
                q["n"],
                f"{q['mean']:.2f}",
                f"{q['std']:.3f}",
                f"{q['snr']:.1f}",
                f"{q['outliers_pct']:.1f}%",
                f"{q['flat_pct']:.1f}%",
                f"{q['kurtosis']:.1f}",
                q["status"],
            ])

    if table_data:
        tbl = ax8.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

                          
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#1F3864")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

                                      
        status_colors_tbl = {"OK": "#1B3A1B", "ATENTIE": "#3A2E00", "PROBLEMATIC": "#3A0000"}
        for i, row in enumerate(table_data, 1):
            status = row[-1]
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(status_colors_tbl.get(status, "#111111"))
                tbl[i, j].set_text_props(
                    color="#66BB6A" if status=="OK" else "#FFA726" if status=="ATENTIE" else "#EF5350"
                )

    ax8.set_title("Metrici Calitate Semnal — Toate Canalele", color="white",
                  fontsize=11, fontweight="bold", pad=60)

    plt.suptitle(f"Analiza Date Brute — {session_id}",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    out = os.path.join(OUTPUT_DIR, f"raw_quality_{session_id}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[PLOT] Salvat: {out}")
    return out


                                               
                          
                                               

def print_quality_report(df: pd.DataFrame, session_id: str):
    """Printeaza un raport detaliat cu recomandarile de filtrare."""

    signals = {
        "ir":       ("IR PPG",          10.0, "bandpass", 0.5,  3.5),
        "red":      ("RED PPG",         10.0, "bandpass", 0.5,  3.5),
        "acc_x":    ("AccX",            10.0, "median",   None, None),
        "acc_y":    ("AccY",            10.0, "median",   None, None),
        "acc_z":    ("AccZ",            10.0, "median",   None, None),
        "gyro_x":   ("GyroX",          10.0, "median",   None, None),
        "flex_raw": ("Flex",            5.0,  "savgol",   None, None),
        "temp_c":   ("Temperatura",     1.0,  "lowpass",  None, 0.05),
    }

    print(f"\n{'='*60}")
    print(f"  RAPORT CALITATE DATE BRUTE — {session_id}")
    print(f"  Total randuri: {len(df)}")
    print(f"{'='*60}\n")

    all_ok = True
    for col, (name, fs, filt, lo, hi) in signals.items():
        if col not in df.columns:
            continue
        q = analyze_signal_quality(df[col], name, fs)
        icon = "✅" if q["status"] == "OK" else "⚠️ " if q["status"] == "ATENTIE" else "❌"
        print(f"{icon} {name:<20} | Status: {q['status']:<12} | "
              f"SNR={q['snr']:>6.1f} | outlieri={q['outliers_pct']:>5.1f}% | "
              f"flat={q['flat_pct']:>5.1f}%")
        if q["issues"]:
            print(f"   Probleme: {', '.join(q['issues'])}")
            print(f"   Filtru recomandat: {filt}"
                  + (f" ({lo}-{hi} Hz)" if lo or hi else ""))
            all_ok = False

    print()
    if all_ok:
        print("✅  Toate semnalele sunt curate! Nu e nevoie de filtrare suplimentara.")
    else:
        print("⚠️   Unele semnale necesita filtrare.")
        print("     Filtrele sunt aplicate automat in sleep_pipeline.py la extragerea features.")
        print("     Pentru date mai curate: verifica contactul senzorilor si conexiunile.")

    print(f"\n[PLOT] Grafic salvat in sleep_output/raw_quality_{session_id}.png")


                                               
              
                                               

if __name__ == "__main__":
    if len(sys.argv) < 2:
                                                   
        conn = sqlite3.connect(DB_PATH)
        sessions = pd.read_sql_query(
            "SELECT DISTINCT session_id FROM raw_data ORDER BY session_id DESC LIMIT 5",
            conn
        )["session_id"].tolist()
        conn.close()
        if not sessions:
            print("EROARE: Nu exista sesiuni in DB!")
            sys.exit(1)
        print(f"Sesiuni disponibile: {sessions}")
        session_id = sessions[0]
        print(f"Folosesc: {session_id}")
    else:
        session_id = sys.argv[1]

    df = load_session(session_id)
    if df.empty:
        print("EROARE: Nu exista date pentru aceasta sesiune!")
        sys.exit(1)

    print_quality_report(df, session_id)
    plot_raw_analysis(df, session_id)
    print(f"\n✅  Analiza completa! Vezi: sleep_output/raw_quality_{session_id}.png")
