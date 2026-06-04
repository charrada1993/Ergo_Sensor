# Hardware Setup Guide

This guide explains how to build and configure the wearable sensor nodes for Ergo Sensor.

---

## 🛠️ Components List

Each sensor node requires:
- **Microcontroller**: ESP32-WROOM-32 (Dual-core with Wi-Fi).
- **IMU Sensor**: BNO085 (9-axis) or MPU-6050 (6-axis).
- **Power**: 3.7V LiPo Battery (500mAh recommended).
- **Case**: 3D-printed enclosure with elastic straps.

---

## 🔌 Wiring Diagram (I2C)

| ESP32 Pin | Sensor Pin | Description |
| :--- | :--- | :--- |
| 3.3V | VCC | Power Supply |
| GND | GND | Ground |
| GPIO 21 | SDA | I2C Data |
| GPIO 22 | SCL | I2C Clock |

---

## 💻 Firmware Installation

1. Install [Arduino IDE](https://www.arduino.cc/en/software) or PlatformIO.
2. Install dependencies:
   - `WiFi.h`
   - `HTTPClient.h`
   - `ArduinoJson.h`
   - `Adafruit_BNO08x.h` (if using BNO085)
3. Flash the code located in `firmware/esp32_imu_sender.ino`.
4. Configure your `SSID` and `PASSWORD` in the code.

---

## 🏃 Sensor Placement Strategy

For a complete 12-sensor audit, place the nodes as follows:

| Segment | Location | Importance |
| :--- | :--- | :--- |
| **Neck** | C7 Vertebra (base of neck) | High (Posture) |
| **Trunk** | T12 Vertebra (mid-back) | High (Core) |
| **Arms** | Bilateral Biceps & Forearms | High (RULA) |
| **Legs** | Bilateral Thighs & Shanks | Medium (REBA) |

> 💡 **Tip**: Use medical-grade Velcro straps to ensure the sensors don't move during high-intensity tasks.

---

## 🎯 Calibration Procedure

1. **Power On**: Turn on all sensor nodes.
2. **Neutral Stance**: Have the worker stand upright with arms by their side (All-Angle Position 0).
3. **Trigger API**: Click **"Calibrate"** on the Web Dashboard.
4. **Verification**: Check the 3D Digital Twin to ensure the virtual skeleton matches the physical worker.
