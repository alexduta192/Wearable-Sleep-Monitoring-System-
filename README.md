# Wearable Sleep Monitoring System

End-to-end wearable system for overnight sleep tracking, built as a Bachelor's thesis at the Faculty of Electronics, Telecommunications and Information Technology (POLITEHNICA University of Bucharest).

**Thesis title:** *Developing a System for Physiological Signal Acquisition and Analysis for Sleep Quality Assessment*
**Supervisor:** Ph.D. Ovidiu Grigore

The system covers the full pipeline: custom wearable hardware, firmware, a backend server, and a signal processing and classification pipeline that assigns sleep stages (Wake, Light, Deep, REM) to each 30-second window of a recording session.

---

## Overview

A wristband and chest band collect physiological signals overnight (heart rate/SpO2, motion, temperature, respiration, and snoring), stream them to a local server, and a Python pipeline processes the data offline to classify sleep stages using unsupervised and supervised machine learning.

## Hardware

| Sensor | Signal | Location |
|---|---|---|
| MAX30102 | PPG / SpO2 | Wristband |
| MPU6050 | Motion (accelerometer/gyroscope) | Wristband |
| BMP280 | Temperature / pressure | Wristband |
| ADS1015 + flex sensor | Respiration | Chest band |
| MAX4466 | Snoring (microphone) | Wristband |

- **MCU:** ESP32-C3 Super Mini
- **Power:** LiPo battery with TP4056 charging circuit
- **Server:** Raspberry Pi, running the backend and database

## Firmware

Built on the Arduino/ESP32 framework. Because the ESP32-C3 is single-core, the original design (three separate HTTP POST requests per cycle) caused significant data loss under load. The firmware was redesigned to consolidate all sensor data into a single combined `/data` POST endpoint, cutting per-cycle latency from ~1500ms to ~500ms and nearly doubling the effective sensor sampling rate.

## Backend

- Python `http.server` (`ThreadingHTTPServer` / `BaseHTTPRequestHandler`) — no external web framework
- SQLite database in WAL mode, with tables for raw data, PPG, flex/respiration, microphone, and alerts
- Live web dashboard (Chart.js) for real-time visualization of incoming sensor data and detected sleep stages

## Signal Processing & Classification

- 30-second analysis windows, 21 extracted features per window
- **K-Means clustering** (k=4) for unsupervised sleep stage segmentation, with post-hoc physiological rules (session-relative percentile thresholds) applied to label clusters as Wake, Light, Deep, or REM
- **Distance-weighted KNN** (k=5) trained on the K-Means labels, with majority-vote smoothing across adjacent windows
- HRV proxy derived from beat-to-beat interval variability where PPG sampling rate is too low for clinical-grade HRV metrics

## Validation

The system was tested across 10 overnight recording sessions with 8 subjects. Known limitations are documented in the thesis, including the effective PPG sampling rate constraint (a direct consequence of the ESP32-C3's single-core architecture) and the absence of polysomnography (PSG) ground truth for validation.

## Tech Stack

`C++ (Arduino/ESP32)` · `Python` · `SQLite` · `pandas` · `scikit-learn` · `NumPy` · `SciPy` · `Chart.js`

## Repository Structure

```
firmware/         ESP32-C3 firmware (Arduino IDE)
server/           Python backend (HTTP server, database)
pipeline/         Signal processing and sleep stage classification
dashboard/        Live web dashboard
```

---

*This project was developed as a Bachelor's thesis and is shared for portfolio purposes.*
