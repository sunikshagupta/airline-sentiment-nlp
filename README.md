# Crystal Growth Environment Monitor

An automated data acquisition and analysis system for monitoring environmental
conditions during crystal growth experiments. Built to demonstrate real-time
I2C sensor integration, serial communication, and high-throughput experimental
workflow automation.

---

## Overview

This project implements an end-to-end pipeline for experimental screening of
crystal growth conditions — from hardware-level sensor acquisition to automated
anomaly detection and report generation.

**Hardware stack:**
- Arduino (I2C master) → reads BME280 (temperature/humidity) and BH1750 (light)
- Raspberry Pi (host controller) → receives serial data, logs and analyzes

**Software stack:**
- Arduino C (sensor polling, I2C communication, serial output)
- Python 3 (data logging, time-series analysis, report generation)

---

## Architecture

```
┌───────────────────────────────────────┐
│            Arduino (I2C Master)        │
│                                       │
│  BME280 (0x76) ──┐                    │
│                  ├─ I2C Bus ──► Sketch │
│  BH1750 (0x23) ──┘            │       │
│                                ▼      │
│              Serial (9600 baud) ──────┼──► Raspberry Pi
└───────────────────────────────────────┘         │
                                                   ▼
                                        crystal_logger.py
                                        (CSV data logging)
                                                   │
                                                   ▼
                                       crystal_analysis.py
                                  (statistics, anomaly detection,
                                        HTML report)
```

---

## Files

| File | Description |
|---|---|
| `crystal_monitor.ino` | Arduino sketch — I2C sensor polling, serial output |
| `crystal_logger.py`   | Python data logger — serial reader, CSV writer |
| `crystal_analysis.py` | Analysis script — plots, anomaly detection, HTML report |

---

## Quick Start

### Simulation mode (no hardware required)

```bash
# Install dependencies
pip install pyserial pandas matplotlib

# Run logger in simulation mode for 10 minutes
python crystal_logger.py --simulate --duration 600

# Analyze the output and generate a report
python crystal_analysis.py --input logs/crystal_run_<timestamp>.csv --report
```

### With real hardware (Raspberry Pi + Arduino)

```bash
# Connect Arduino via USB, identify port
ls /dev/tty*          # Linux/macOS
# or check Device Manager on Windows

# Run logger
python crystal_logger.py --port /dev/ttyUSB0 --duration 3600

# Analyze
python crystal_analysis.py --input logs/crystal_run_<timestamp>.csv --report
```

---

## Sensor Configuration

| Sensor | Parameter | I2C Address | Interface |
|---|---|---|---|
| BME280 | Temperature (°C), Humidity (%RH) | 0x76 | I2C |
| BH1750 | Illuminance (lux) | 0x23 | I2C |

To swap in real sensors, replace the simulation block in `crystal_monitor.ino`
with the appropriate sensor library calls (Adafruit BME280, BH1750FVI).

---

## Anomaly Detection

The analysis pipeline uses a **rolling z-score** method to flag environmental
deviations that could indicate:
- Crystallization onset (temperature/humidity shifts)
- Equipment fluctuations
- External disturbances

Default parameters: window = 10 readings, threshold = 2.5σ. Both are
configurable in `crystal_analysis.py`.

---

## Output

- **`logs/crystal_run_<timestamp>.csv`** — raw time-series data
- **`logs/environment_plot.png`** — three-panel time-series plot with anomaly markers
- **`logs/report.html`** — full summary report with statistics and embedded plot

---

## Extending This Project

- Add HPLC or UV-Vis spectrometer output parsing
- Integrate motion control (stepper motor) for automated crystal stage positioning
- Connect to a LabVIEW DAQ system via serial bridge
- Add nucleation event detection using optical density thresholding

---

## Requirements

```
pyserial>=3.5
pandas>=1.3
matplotlib>=3.4
```

---

## Background

Developed as part of an automated experimental screening workflow for
optoelectronic material characterization. Designed for extensibility —
the serial/I2C abstraction layer makes it straightforward to swap simulated
data for real sensors or integrate additional analytical instruments.
