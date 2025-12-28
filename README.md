# Smart Overtake Assist System

**This application** is a Python-based Advanced Driver Assistance System (ADAS) designed to assist drivers with safe overtaking maneuvers. It combines modern computer vision techniques (YOLOv8) with robust lane detection and temporal logic to assess traffic risks in real-time.

## 🚀 Features

### 1. Vision Perception
- **Vehicle Detection**: Utilizes **YOLOv8** for accurate detection of cars, trucks, and motorcycles.
- **Lane Detection**: A robust approach using **Hough Transform** with advanced spatial and temporal filtering.
    - Includes **HLS Color Filtering** for robustness against lighting changes.
    - Features **Temporal Smoothing** and **Forecasting** to stabilize lane lines and prevent jitter.

### 2. Safety Logic
- **Risk Assessment**: Analyzes vehicle position relative to the lane markings.
    - **Risky Lane Monitoring**: Identifies if the overtaking lane (right lane) is occupied.
    - **Adaptive Thresholds**: Uses dynamic overlap thresholds (5% for distant/small vehicles, 10% for nearby/large vehicles) to catch potential threats early.
- **Trend Analysis**: Tracks vehicle width expansion over time to determine if a vehicle is **Approaching** (dangerous) or **Receding** (safe).
    - **Status Output**: Classifies situations as `SAFE`, `WARNING`, or `RISKY`.

### 3. Ablation Study Framework
- A dedicated module to evaluating the contribution of individual system components.
- Supports configurable experiments to test the impact of:
    - **Skip-3 Optimization**: Running detection every 3rd frame vs. every frame.
    - **Trend Analysis**: Temporal tracking vs. static frame analysis.
    - **Adaptive Thresholds**: Dynamic vs. fixed overlap ratios.
    - **Temporal Smoothing**: Stabilized vs. raw lane detection.

## 📂 Project Structure

```
application/
├── data/                   # Models and Test Videos
├── src/
│   ├── main.py             # Main application entry point
│   ├── ablation/           # Ablation study modules and runner
│   │   ├── ablation_configs.py
│   │   ├── run_ablation.py
│   │   └── ...
│   ├── logic/              # Safety decision logic
│   │   └── safety_checker.py
│   ├── perception/         # Vision processing
│   │   ├── lane_detector.py
│   │   └── vehicle_detector.py
│   └── scripts/            # Evaluation and utility scripts
│       ├── evaluate.py     # Evaluation with visual tuning
│       ├── evaluate_baseline.py
│       └── ...
├── config.py               # Global configuration settings
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd "application"
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages: `ultralytics`, `opencv-python`, `numpy`, `torch`.*

## 🚦 Usage

### 1. Running the Main Application
To start the live system with visualization:
```bash
python src/main.py
```

### 2. Evaluation
To evaluate system performance against ground truth labels:

**Option A: Baseline Evaluation (Headless)**
```bash
python src/scripts/evaluate_baseline.py <path_to_video> <path_to_ground_truth_csv>
```

**Option B: Visual Evaluation with Tuning**
Includes a parameter tuning window before starting.
```bash
python src/scripts/evaluate.py <path_to_video> <path_to_ground_truth_csv>
```

### 3. Running an Ablation Study
To run the full suite of experiments (Ablation Study) to benchmark different configurations:

```bash
python src/ablation/run_ablation.py
```
*Note: This script includes a **Tuning Mode** at the start. Adjust the lane detection parameters using the sliders and press **'s'** to start the experiments.*

### 4. Tuning Parameters
The system uses `src/config.py` for global defaults. You can modify:
- **ROI Vertices**: Region of Interest for lane detection.
- **Safety Thresholds**: Distances and TTC values.
- **Model Paths**: Path to the YOLO `.pt` model web weights.

## 📊 Methodology

The system follows a pipeline approach:
1.  **Input**: Video frame.
2.  **Perception**:
    *   YOLOv8 detects vehicles.
    *   Lane Detector identifies the overtaking lane boundaries.
3.  **State Tracking**:
    *   Vehicles are tracked across frames to calculate expansion rates (Velocity/Approaching status).
    *   Lane positions are smoothed using a forecasting model.
4.  **Logic**:
    *   The `SafetyChecker` checks for overlaps between detected vehicles and the overtaking lane.
    *   If an "Approaching" vehicle overlaps significantly, a `RISKY` alert is issued.

## 📝 License
[MIT License](LICENSE)
