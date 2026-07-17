"""
alert_monitor.py  –  Monitor alerte live
==========================================
Se integreaza cu server.py pentru detectie in timp real.

Alerte detectate:
  - APNEE:        flex_delta constant (variatie < 3) mai mult de 10s
  - APNEE_SEVERA: flex_delta constant mai mult de 20s
  - BPM_LOW:      BPM < 40
  - BPM_HIGH:     BPM > 120
  - SPO2_LOW:     SpO2 < 90%
  - SFORAIT:      RMS microfon peste prag, sustinut > 3s
  - TEMP_LOW:     Temperatura < 28°C (senzor detasat probabil)
  - TEMP_HIGH:    Temperatura > 38°C
  - MISCARE:      Gyroscop > 50 deg/s (trezire/miscare mare)
  - FLEX_FLAT:    Flex constant > 15s = senzor detasat
  - PPG_FLAT:     IR constant > 15s = senzor detasat

Utilizare in server.py:
  from alert_monitor import AlertMonitor
  monitor = AlertMonitor(DB_PATH, session_id, subject_id)
  # in insert_raw():  monitor.check_raw(data)
  # in insert_mic():  monitor.check_mic(data)
  # la stop_session: monitor.print_summary()
"""

import sqlite3
import time
import math
import threading
from datetime import datetime
from collections import deque


class AlertMonitor:

    def __init__(self, db_path, session_id, subject_id):
        self.db_path    = db_path
        self.session_id = session_id
        self.subject_id = subject_id
        self._lock      = threading.Lock()

                                                        
        self.APNEA_VARIATION      = 3                                                
        self.APNEA_TIME_SEC       = 10                   
        self.APNEA_SEVERE_SEC     = 20                   
        self.BPM_LOW              = 40
        self.BPM_HIGH             = 120
        self.SPO2_LOW             = 90.0
        self.TEMP_LOW             = 28.0
        self.TEMP_HIGH            = 38.0
        self.GYRO_MOVEMENT        = 50.0          
        self.SNORE_RMS_FACTOR     = 3.0                         
        self.SNORE_DURATION_SEC   = 3
        self.PPG_FLAT_SEC         = 15                                   
        self.FLEX_FLAT_SEC        = 15                                     

                                                        
        self.COOLDOWN_SEC = {
            "APNEE":        30,
            "APNEE_SEVERA": 60,
            "BPM_LOW":      60,
            "BPM_HIGH":     60,
            "SPO2_LOW":     60,
            "SFORAIT":      15,
            "TEMP_LOW":     120,
            "TEMP_HIGH":    120,
            "MISCARE":      10,
            "FLEX_FLAT":    120,
            "PPG_FLAT":     120,
        }
        self._last_alert_time = {}                                  

                                                        
        self.flex_history   = deque(maxlen=30)                               
        self.ir_history     = deque(maxlen=20)                       
        self.bpm_history    = deque(maxlen=10)                         
        self.mic_rms_history = deque(maxlen=10)                

                                                       
        self._ppg_ir_buffer  = deque(maxlen=300)                 
        self._last_bpm       = 0
        self._last_spo2      = 0

                                                       
        self._apnea_start    = None                                               
        self._apnea_alerted  = False                                       
        self._apnea_severe_alerted = False

                                                       
        self._snore_start    = None
        self._mic_baseline_rms = None
        self._mic_baseline_std = None
        self._mic_calibration_count = 0
        self._mic_calibration_sum   = 0.0
        self._mic_calibration_sq    = 0.0
        self.MIC_CALIBRATION_SAMPLES = 30                                  

                                                       
        self.alert_counts = {}
        self.total_alerts = 0

                                
        self._init_alerts_table()

    def _init_alerts_table(self):
        """Creeaza tabelul alerts daca nu exista."""
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                pc_timestamp  TEXT,
                sensor_time   TEXT,
                alert_type    TEXT,
                severity      TEXT,
                value         REAL,
                threshold     REAL,
                duration_sec  REAL,
                message       TEXT,
                subject_id    TEXT,
                session_id    TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id, pc_timestamp)")
        conn.commit()
        conn.close()

    def _can_alert(self, alert_type):
        """Verifica cooldown — returneaza True daca se poate genera alerta."""
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        cooldown = self.COOLDOWN_SEC.get(alert_type, 30)
        return (now - last) >= cooldown

    def _save_alert(self, sensor_time, alert_type, severity, value, threshold, duration, message):
        """Salveaza alerta in DB si printeaza in consola."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_alert_time[alert_type] = time.time()

                        
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("""
                INSERT INTO alerts (
                    pc_timestamp, sensor_time, alert_type, severity,
                    value, threshold, duration_sec, message,
                    subject_id, session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                now_str, sensor_time, alert_type, severity,
                value, threshold, duration, message,
                self.subject_id, self.session_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [ALERT DB ERROR] {e}")

                    
        self.alert_counts[alert_type] = self.alert_counts.get(alert_type, 0) + 1
        self.total_alerts += 1

                                  
        severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(severity, "❓")
        print(f"\n  {severity_icon} [{alert_type}] {message}")
        print(f"     Valoare: {value:.1f} | Prag: {threshold:.1f} | Severitate: {severity}")
        if duration and duration > 0:
            print(f"     Durata: {duration:.0f}s")

                                                 
                                               
                                                 
    def check_raw(self, data):
        """Analizeaza datele senzorilor la fiecare secunda."""
        with self._lock:
            sensor_time = data.get("t", "")
            flex_delta  = data.get("fd", 0)
            ir_val      = data.get("ir", 0)
            red_val     = data.get("red", 0)
            temp_c      = data.get("tmp", 0.0)
            gyro_x      = data.get("gx", 0.0)
            gyro_y      = data.get("gy", 0.0)
            gyro_z      = data.get("gz", 0.0)

                               
            self.flex_history.append(flex_delta)
            self.ir_history.append(ir_val)

                                                       
            self._check_apnea(sensor_time)

                                                       
            self._check_temperature(sensor_time, temp_c)

                                                       
            self._check_movement(sensor_time, gyro_x, gyro_y, gyro_z)

                                                        
            self._check_flex_flat(sensor_time)

                                                        
            self._check_ppg_flat(sensor_time)

    def _check_apnea(self, sensor_time):
        """Detecteaza apnee pe baza variatiei flex_delta."""
        if len(self.flex_history) < self.APNEA_TIME_SEC:
            return

                                    
        recent = list(self.flex_history)[-self.APNEA_TIME_SEC:]
        variation = max(recent) - min(recent)

        if variation < self.APNEA_VARIATION:
                                            
            if self._apnea_start is None:
                self._apnea_start = time.time()

            elapsed = time.time() - self._apnea_start

                                 
            if elapsed >= self.APNEA_SEVERE_SEC and not self._apnea_severe_alerted:
                if self._can_alert("APNEE_SEVERA"):
                    self._save_alert(sensor_time, "APNEE_SEVERA", "CRITICAL",
                                     elapsed, self.APNEA_SEVERE_SEC, elapsed,
                                     f"Apnee severa! Fara respiratie de {elapsed:.0f}s")
                    self._apnea_severe_alerted = True

                                 
            elif elapsed >= self.APNEA_TIME_SEC and not self._apnea_alerted:
                if self._can_alert("APNEE"):
                    self._save_alert(sensor_time, "APNEE", "WARNING",
                                     elapsed, self.APNEA_TIME_SEC, elapsed,
                                     f"Posibila apnee — fara respiratie de {elapsed:.0f}s")
                    self._apnea_alerted = True
        else:
                                             
            if self._apnea_start is not None:
                self._apnea_start = None
                self._apnea_alerted = False
                self._apnea_severe_alerted = False

    def _check_temperature(self, sensor_time, temp_c):
        """Detecteaza temperatura anormala."""
        if temp_c <= 0:
            return                        

        if temp_c < self.TEMP_LOW and self._can_alert("TEMP_LOW"):
            self._save_alert(sensor_time, "TEMP_LOW", "WARNING",
                             temp_c, self.TEMP_LOW, 0,
                             f"Temperatura scazuta: {temp_c:.1f}°C (senzor detasat?)")

        elif temp_c > self.TEMP_HIGH and self._can_alert("TEMP_HIGH"):
            self._save_alert(sensor_time, "TEMP_HIGH", "WARNING",
                             temp_c, self.TEMP_HIGH, 0,
                             f"Temperatura ridicata: {temp_c:.1f}°C")

    def _check_movement(self, sensor_time, gx, gy, gz):
        """Detecteaza miscare excesiva (trezire)."""
        gyro_magnitude = math.sqrt(gx*gx + gy*gy + gz*gz)

        if gyro_magnitude > self.GYRO_MOVEMENT and self._can_alert("MISCARE"):
            self._save_alert(sensor_time, "MISCARE", "INFO",
                             gyro_magnitude, self.GYRO_MOVEMENT, 0,
                             f"Miscare mare detectata: {gyro_magnitude:.1f} deg/s")

    def _check_flex_flat(self, sensor_time):
        """Detecteaza flex sensor detasat (valoare constanta)."""
        if len(self.flex_history) < self.FLEX_FLAT_SEC:
            return

        recent = list(self.flex_history)[-self.FLEX_FLAT_SEC:]
                                                                  
        if len(set(recent)) <= 2 and self._can_alert("FLEX_FLAT"):
            self._save_alert(sensor_time, "FLEX_FLAT", "WARNING",
                             recent[-1], 0, self.FLEX_FLAT_SEC,
                             f"Flex sensor posibil detasat — semnal constant {self.FLEX_FLAT_SEC}s")

    def _check_ppg_flat(self, sensor_time):
        """Detecteaza MAX30102 detasat (IR constant)."""
        if len(self.ir_history) < self.PPG_FLAT_SEC:
            return

        recent = list(self.ir_history)[-self.PPG_FLAT_SEC:]
                                            
        variation = max(recent) - min(recent)
        if (variation < 10 or recent[-1] == 0) and self._can_alert("PPG_FLAT"):
            self._save_alert(sensor_time, "PPG_FLAT", "WARNING",
                             recent[-1], 0, self.PPG_FLAT_SEC,
                             f"MAX30102 posibil detasat — IR constant {self.PPG_FLAT_SEC}s")

                                                 
                                               
                                                 
    def check_mic(self, data):
        """Analizeaza blocul microfon pentru sforait."""
        with self._lock:
            sensor_time = data.get("t", "")
            samples_str = data.get("s", "")

            if not samples_str:
                return

            try:
                samples = [int(x) for x in samples_str.split(",") if x.strip()]
            except ValueError:
                return

            if len(samples) < 10:
                return

                            
            mean_val = sum(samples) / len(samples)
            sq_sum = sum((s - mean_val) ** 2 for s in samples)
            rms = math.sqrt(sq_sum / len(samples))

                                            
            if self._mic_calibration_count < self.MIC_CALIBRATION_SAMPLES:
                self._mic_calibration_count += 1
                self._mic_calibration_sum += rms
                self._mic_calibration_sq  += rms * rms
                if self._mic_calibration_count == self.MIC_CALIBRATION_SAMPLES:
                    mean_rms = self._mic_calibration_sum / self.MIC_CALIBRATION_SAMPLES
                    var_rms  = (self._mic_calibration_sq / self.MIC_CALIBRATION_SAMPLES) - mean_rms * mean_rms
                    self._mic_baseline_rms = mean_rms
                    self._mic_baseline_std = math.sqrt(max(var_rms, 0.01))
                    print(f"  [MIC] Calibrare completa: baseline RMS={mean_rms:.1f}, std={self._mic_baseline_std:.1f}")
                return

            if self._mic_baseline_rms is None:
                return

                                                    
            snore_threshold = self._mic_baseline_rms + self.SNORE_RMS_FACTOR * self._mic_baseline_std

            self.mic_rms_history.append(rms)

            if rms > snore_threshold:
                if self._snore_start is None:
                    self._snore_start = time.time()
                elapsed = time.time() - self._snore_start
                if elapsed >= self.SNORE_DURATION_SEC and self._can_alert("SFORAIT"):
                    self._save_alert(sensor_time, "SFORAIT", "INFO",
                                     rms, snore_threshold, elapsed,
                                     f"Sforait detectat: RMS={rms:.1f} (prag={snore_threshold:.1f})")
                    self._snore_start = None                     
            else:
                self._snore_start = None

                                                 
                                               
                                                 
    def check_ppg(self, data):
        """Analizeaza blocul PPG pentru BPM si SpO2."""
        with self._lock:
            sensor_time = data.get("t", "")
            ir_str  = data.get("ir", "")
            red_str = data.get("red", "")

            if not ir_str or not red_str:
                return

            try:
                ir_samples  = [int(x) for x in ir_str.split(",") if x.strip()]
                red_samples = [int(x) for x in red_str.split(",") if x.strip()]
            except ValueError:
                return

            if len(ir_samples) < 10:
                return

                                                       
            bpm = self._calc_bpm(ir_samples, fs=50)
            if bpm and bpm > 0:
                self._last_bpm = bpm
                self.bpm_history.append(bpm)

                if bpm < self.BPM_LOW and self._can_alert("BPM_LOW"):
                    self._save_alert(sensor_time, "BPM_LOW", "CRITICAL",
                                     bpm, self.BPM_LOW, 0,
                                     f"BPM foarte scazut: {bpm:.0f} BPM")

                elif bpm > self.BPM_HIGH and self._can_alert("BPM_HIGH"):
                    self._save_alert(sensor_time, "BPM_HIGH", "WARNING",
                                     bpm, self.BPM_HIGH, 0,
                                     f"BPM ridicat: {bpm:.0f} BPM")

                                                       
            spo2 = self._calc_spo2(ir_samples, red_samples)
            if spo2 and spo2 > 0:
                self._last_spo2 = spo2

                if spo2 < self.SPO2_LOW and self._can_alert("SPO2_LOW"):
                    self._save_alert(sensor_time, "SPO2_LOW", "CRITICAL",
                                     spo2, self.SPO2_LOW, 0,
                                     f"SpO2 scazut: {spo2:.1f}%")

    def _calc_bpm(self, ir_samples, fs=50):
        """Calculeaza BPM din IR samples prin peak detection simplu."""
        if len(ir_samples) < 20:
            return None

                                     
        n = len(ir_samples)
        smooth = []
        w = 3
        for i in range(n):
            start = max(0, i - w)
            end = min(n, i + w + 1)
            smooth.append(sum(ir_samples[start:end]) / (end - start))

                                      
        mean_val = sum(smooth) / len(smooth)
        peaks = []
        for i in range(1, len(smooth) - 1):
            if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
                if smooth[i] > mean_val:
                    peaks.append(i)

        if len(peaks) < 2:
            return None

                                                 
        intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
                                      
        intervals = [iv for iv in intervals if 15 < iv < 75]                    

        if not intervals:
            return None

        avg_interval = sum(intervals) / len(intervals)
        bpm = (fs / avg_interval) * 60
        return bpm

    def _calc_spo2(self, ir_samples, red_samples):
        """Calculeaza SpO2 simplificat din ratio of ratios."""
        if len(ir_samples) < 10 or len(red_samples) < 10:
            return None

                     
        ir_mean  = sum(ir_samples) / len(ir_samples)
        red_mean = sum(red_samples) / len(red_samples)

        if ir_mean == 0 or red_mean == 0:
            return None

        ir_ac  = math.sqrt(sum((s - ir_mean)**2 for s in ir_samples) / len(ir_samples))
        red_ac = math.sqrt(sum((s - red_mean)**2 for s in red_samples) / len(red_samples))

        if ir_ac == 0:
            return None

                                                 
        R = (red_ac / red_mean) / (ir_ac / ir_mean)

                                                     
        spo2 = 110 - 25 * R

        if spo2 < 50 or spo2 > 100:
            return None

        return spo2

                                                 
            
                                                 
    def print_summary(self):
        """Afiseaza sumarul alertelor la sfarsitul sesiunii."""
        print(f"\n{'─'*55}")
        print(f"  SUMAR ALERTE")
        print(f"{'─'*55}")
        if self.total_alerts == 0:
            print(f"  Nicio alerta inregistrata.")
        else:
            print(f"  Total alerte: {self.total_alerts}")
            for alert_type, count in sorted(self.alert_counts.items()):
                print(f"    {alert_type:20s}: {count}")
        if self._last_bpm > 0:
            print(f"\n  Ultim BPM calculat: {self._last_bpm:.0f}")
        if self._last_spo2 > 0:
            print(f"  Ultim SpO2: {self._last_spo2:.1f}%")
        print(f"{'─'*55}")

    def get_stats(self):
        """Returneaza statisticile curente (pentru dashboard)."""
        with self._lock:
            return {
                "total_alerts": self.total_alerts,
                "alert_counts": dict(self.alert_counts),
                "last_bpm":     self._last_bpm,
                "last_spo2":    self._last_spo2,
            }
