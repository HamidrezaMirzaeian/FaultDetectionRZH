
# Wind turbine fault detection using machine learning

This is the second of two final projects in the 2nd Specific Course of the TWEED Doctoral Network, held at DTU during March 9-13, 2026.

Building upon the forecasting project, the objective here is to develop a Python module for **wind turbine fault detection and condition monitoring** using machine learning. You and your group will leverage a real-world SCADA dataset from an onshore wind farm in Portugal (EDP open data) to build models that can detect incipient failures, moving from reactive to predictive maintenance. You code will preprocess data, implement an unsupervised learning model, and evaluate detection performance with a focus on building a trustworthy intelligent alarm system.

## Background

Wind turbines operate under harsh and variable conditions, making them susceptible to component failures (e.g., in gearboxes, generators, bearings, pitch systems). Unplanned downtime due to failures is a major cost driver in wind farm operation. Fault detection aims to identify anomalies in turbine behavior as early as possible, enabling condition-based maintenance. This reduces maintenance costs, increases turbine availability, and prevents catastrophic, costly damage.

### From Forecasting to Fault Detection: A Shift in Perspective

In the first project, you used meteorological variables to predict **power output**. In this project, you will use the **same type of sensor data** (temperatures, speeds, power, etc.) but now to predict the **health state** of the turbine. The core idea is that developing faults leave a subtle but detectable signature in the SCADA data before they trigger an official alarm or cause a failure.

### Machine Learning Strategies for Fault Detection

Fault detection can be approached in several ways, each with its own assumptions and data requirements:

1.  **Supervised Classification:**
    - **Concept:** Treat fault detection as a binary (fault/normal) or multi-class (specific fault type) classification problem.
    - **Requirement:** Requires a historical dataset where faults have been explicitly logged and labeled. The provided dataset includes an `event_info.csv` file with labels (`anomaly` or `normal`) and the exact time windows of the events.
    - **Challenge:** Faults are rare events, leading to heavily imbalanced datasets. A model that always predicts "normal" would be highly accurate but useless. You must address this imbalance.

2.  **Unsupervised Anomaly Detection:**
    - **Concept:** Model the "normal" behavior of a turbine using only data from healthy operation. Any significant deviation from this learned normal behavior is flagged as a potential fault (anomaly).
    - **Requirement:** Only needs a clean dataset of normal operation. You can extract such periods using the `status_type` column (e.g., status 0 and 2 indicate normal operation) and by excluding the labeled event windows.
    - **Advantage:** Much more realistic, as labeled fault data is scarce. Can detect novel, unforeseen fault types.
    - **Common Approach:** Autoencoders, One-Class SVM, Isolation Forests.

3.  **Remaining Useful Life (RUL) Prediction:**
    - **Concept:** A more advanced form of regression where the goal is to predict the time left before a component is likely to fail.
    - **Note:** This is beyond the scope of this introductory project but represents the cutting edge.

### Key Challenges in Data-Driven Fault Detection

- **False Alarms:** A system that triggers too many false alarms will be ignored by operators. The goal is high precision (when it says "fault", it is likely a fault) and high recall (it catches most real faults).
- **Data Imbalance:** As mentioned, the vast majority of data represents normal operation.
- **Operational Variability:** A turbine behaves differently at different wind speeds and under different control modes (e.g., derated operation, idling). A model must distinguish between a change in behavior due to a fault and a change due to a different but normal operating condition. The `status_type` column can help here.
- **Trustworthiness:** Operators need to trust the alarm. A "black box" model that flags an anomaly without explanation is less useful than a system that can provide a confidence level or highlight which sensors are driving the alarm.

## Provided Data

For this project, we will use a **subset** of the full dataset: **all data from Wind Farm A, Turbine 0**. This subset is sufficient to build and evaluate meaningful fault detection models. The data originates from the EDP open data platform and has been structured specifically for fault detection research.

The dataset for Wind Farm A is organized as follows:

- **Wind Farm A**
  - `datasets/` — Contains 5 CSV files, each representing one event (either an anomaly event or a normal operation period) for **Turbine 0**.
  - `event_info.csv` — Provides labels and time windows for all events.
  - `feature_description.csv` — Describes the anonymized sensors, their statistics, units, and whether they represent angles or counters.

### Event Data (CSV files)

Each CSV file in `datasets/` contains 10‑minute SCADA data for a single turbine over a period that includes both training data (normal operation) and the event window. The columns are:

- `time_stamp`: Anonymized timestamps at 10‑minute frequency.
- `asset_id`: Anonymized wind turbine ID. For our project, we will filter for `asset_id` corresponding to Turbine 0.
- `id`: An additional unique ID for each timestamp (can be ignored).
- `train_test`: A string indicating whether the timestamp belongs to the training part (`"train"`) or the prediction part (`"test"`) of the dataset. The training part is meant for learning normal behavior; the test part contains the event to be predicted.
- `status_type`: Operational status of the turbine (see table below).
- Sensor columns: Named `sensor_x_avg`, `sensor_x_min`, `sensor_x_max`, `sensor_x_std` (where `x` is an anonymized sensor index). Not all sensors have all four statistics.

**Status types:**

| Status | Description |
|--------|-------------|
| 0 | Normal Operation – turbine in normal power production mode |
| 1 | Derated Operation – power generation with a restriction |
| 2 | Idling – turbine is idling and waiting to operate again |
| 3 | Service – turbine is in service mode / service team on site |
| 4 | Downtime – turbine is down due to a fault or other reasons |
| 5 | Other – other operational states |

**Important:** For the purpose of this project, we consider **status 0 and 2 as normal operation** (healthy behavior). Status 1, 3, 4, and 5 represent abnormal or non-productive states that should either be excluded from training or treated as potential fault indicators, depending on the context.

### Event Information (`event_info.csv`)

This file contains one row per event (i.e., per CSV file in `datasets/`). The columns are:

- `event_id`: Matches the name of the corresponding CSV file (without `.csv`).
- `event_label`: Either `"anomaly"` (the event is a fault leading up to a failure) or `"normal"` (the event is a period of normal behavior).
- `event_start`: Start timestamp of the event (within the test part of the CSV).
- `event_end`: End timestamp of the event.
- `event_start_id`: The `id` of the start timestamp.
- `event_end_id`: The `id` of the end timestamp.
- `event_description`: Additional information about the root cause for some anomaly events (may be empty).

### Feature Description (`feature_description.csv`)

This file describes each sensor and its available statistics:

- `sensor_name`: Anonymized name (e.g., `sensor_1`).
- `statistic_type`: Whether the column provides the average (`"avg"`), minimum (`"min"`), maximum (`"max"`), or standard deviation (`"std"`).
- `description`: Brief description of what the sensor measures (e.g., "generator bearing temperature").
- `unit`: Unit of measurement.
- `is_angle`: Boolean – `True` if the measurement is an angle (e.g., wind direction, blade pitch).
- `is_counter`: Boolean – `True` if the measurement is a counter (e.g., number of rotations, energy production).

### Data Subset for the Project

To keep the project manageable, we will focus exclusively on **Turbine 0 in Wind Farm A**. You will need to:

- Identify all CSV files in `Wind Farm A/datasets/` that contain data for `asset_id == 0`.
- Merge these files (they are separate events) into a single time‑series DataFrame, being careful with overlapping time periods (if any).
- Use `event_info.csv` to create a target column `fault_flag` that is 1 during the `[event_start, event_end]` window for anomaly events, and 0 elsewhere (including all normal events).

**Note:** The `train_test` column indicates which part of each CSV is intended for training (normal behavior) and which part contains the event. You may choose to use this hint, but you are also free to define your own training/validation splits.

## Requirements

### Functional Requirements

Your module should be able to perform the following tasks. You are expected to build upon the code structure from the first project, but now tailored to fault detection.

1.  **Load and Parse Data:**
    - Write a function to load all relevant CSV files for **Turbine 0** from Wind Farm A.
    - Concatenate them into a single DataFrame, sorted by timestamp.
    - Merge with `event_info.csv` to add a binary column `fault_flag` (1 for anomaly event windows, 0 otherwise).
    - Handle the `status_type` column appropriately (e.g., you may filter out non‑operational states when training a normal behavior model).

2.  **Plot Sensor Timeseries with Event Overlay:**
    - Create a function that, given a sensor name (e.g., `sensor_1_avg`) and a time window, plots the sensor values.
    - On the same plot, highlight the periods where `fault_flag == 1` (anomaly events) with a shaded background.
    - This is essential for exploratory data analysis.

3.  **Compute Detection Metrics:**
    - Implement functions to calculate key performance metrics for a binary classifier, especially those suited for imbalanced data:
        - **Precision:** `TP / (TP + FP)`
        - **Recall (True Positive Rate):** `TP / (TP + FN)`
        - **F1-Score:** Harmonic mean of precision and recall.
        - **False Alarm Rate (FAR):** `FP / (FP + TN)`
    - Also implement a function to plot a confusion matrix.

4.  **Train an Unsupervised Anomaly Detector (using `scikit-learn` or optionally TensorFlow):**
    - **Baseline (scikit-learn):** Implement a One-Class SVM or an Isolation Forest to model normal behavior. Train it **only on data from normal operation periods** (e.g., where `fault_flag == 0` and `status_type` is 0 or 2). Then, for each point in the test set, obtain an anomaly score. Choose a threshold (e.g., 95th percentile of scores on normal validation data) to classify points as anomalous.
    - **Optional (TensorFlow/Keras):** Implement an autoencoder. Train it to reconstruct normal operation data. Use the reconstruction error (MSE) as the anomaly score. Set a threshold on the validation normal data.
    - In either case, generate binary predictions (`1` if anomaly score > threshold) and evaluate against the true `fault_flag`.

5.  **Alarm Generation with Persistence Filter:**
    - Raw anomaly scores may lead to many false alarms (single spikes). Implement a simple persistence filter: only raise an alarm if the anomaly score exceeds the threshold for **at least 3 consecutive time steps (30 minutes)**.
    - Evaluate how this filter affects precision and recall.

6.  **Plot Results:**
    - Create a diagnostic plot for a given time window showing:
        - The actual power output (if available, or a key sensor).
        - The anomaly score (reconstruction error or One-Class SVM score).
        - The threshold line.
        - The generated alarms (as shaded regions).
        - The true fault windows (if available) for comparison.

### Formal Requirements

- Your code should be well-documented, modular.
- Use `scikit-learn` as the primary machine learning library. If you attempt the autoencoder, you may use TensorFlow/Keras or PyTorch, but this is optional.

### Other Tips

- **Start with the supervised approach** using a Random Forest. It will give you a quick baseline and help you understand feature importance.
- **Feature engineering:** Re‑use the cyclical encoding for any angular variables (`is_angle=True`). Consider creating lagged features (e.g., value 1 hour ago) to capture trends. Rolling statistics (e.g., moving average) can also help.
- **Train/test split:** Because the data contains multiple events over time, always split temporally (e.g., train on the first 80% of time, test on the last 20%) to avoid look‑ahead bias.
- **Scaling:** For distance‑based models (SVM, autoencoders), scale your features to zero mean and unit variance using `StandardScaler`. Fit the scaler only on the training data (normal operation data for unsupervised methods).
- **Think like an operator:** A single spike is noise; a persistent pattern is an alarm. Your persistence filter is a simple but effective way to reduce false alarms.

## References

- The dataset is derived from the **EDP open data platform** (https://www.edp.com/en/innovation/open-data/data). The specific curation into events is part of a research dataset used in several fault detection studies.
- Gück, Christian, Cyriana MA Roelofs, and Stefan Faulstich. "Care to compare: a real-world benchmark dataset for early fault detection in wind turbine data." Data 9.12 (2024): 138.
- Roelofs, Cyriana MA, et al. "Autoencoder-based anomaly root cause analysis for wind turbines." Energy and AI 4 (2021): 100065.

## Extra Materials

- **Imbalanced-learn documentation:** [User guide](https://imbalanced-learn.org/stable/user_guide.html).
- **Autoencoders (optional):** [Keras autoencoder tutorial](https://blog.keras.io/building-autoencoders-in-keras.html).
- **Time series cross-validation:** [Scikit-learn's `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Good luck, and remember: a good fault detection system is one that operators trust and act upon.