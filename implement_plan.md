# AI-Driven Bus Scheduling Optimization Implementation Plan
## Ho Chi Minh City Public Transportation System

**Project:** AI-Driven Optimization of Bus Scheduling in Ho Chi Minh City
**Author:** Pham Quang Nhan - 2470885
**Supervisor:** Assoc. Prof. Dr. Tran Minh Quang
**Institution:** HCMC University of Technology - Faculty of Computer Science and Engineering

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Detailed Implementation Plan](#3-detailed-implementation-plan)
4. [Technical Stack & Dependencies](#4-technical-stack--dependencies)
5. [Data Pipeline Architecture](#5-data-pipeline-architecture)
6. [Evaluation Metrics & Success Criteria](#6-evaluation-metrics--success-criteria)
7. [Code Templates & Starting Points](#7-code-templates--starting-points)
8. [Risk Assessment & Mitigation](#8-risk-assessment--mitigation)
9. [Next Steps & Deliverables](#9-next-steps--deliverables)

---

## 1. Project Overview

### Problem Statement

Ho Chi Minh City's public bus system faces:
- **Overcrowding during peak hours** with buses at capacity
- **Underutilization in off-peak periods** (avg 18-19 passengers/trip in 2023)
- **Fixed schedules** that don't adapt to real-time demand
- **Suboptimal vehicle allocation** across 120 routes and 2,052 vehicles
- **Poor passenger experience** with long, unpredictable wait times

### Solution Approach

An AI-driven framework integrating:
- **LSTM Neural Networks** for passenger demand forecasting
- **Ant Colony Optimization (ACO)** for dynamic schedule optimization
- **Real-time data ingestion** from GPS, ticketing, weather, and events
- **Closed-loop feedback** for continuous improvement

### Expected Outcomes

- **Forecast accuracy:** >85% (MAPE < 15%)
- **Waiting time reduction:** ≥20%
- **Occupancy improvement:** 70-80% during peak and off-peak
- **Cost savings:** 10-15% reduction in operational costs
- **Real-time updates:** 15-30 minute schedule adjustments

---

## 2. System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                  │
│  ┌──────────┬──────────┬──────────┬─────────────────┐  │
│  │ Ticketing│   GPS    │ Weather  │  Event Data     │  │
│  │  System  │ Tracking │   API    │  (Web Scraping) │  │
│  └──────────┴──────────┴──────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING & STORAGE                │
│  • Data cleaning & normalization                         │
│  • Feature engineering (temporal, spatial, contextual)   │
│  • Time-series database (InfluxDB/TimescaleDB)          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              DEMAND FORECASTING MODULE (LSTM)            │
│  • Input: Historical + Real-time data                    │
│  • Output: 15-30 min ahead demand forecasts              │
│  • Target Accuracy: >85% (MAE, RMSE)                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│         SCHEDULE OPTIMIZATION MODULE (ACO)               │
│  • Input: Demand forecasts + Constraints                 │
│  • Output: Dynamic schedules (headways, allocations)     │
│  • Objective: Min wait time, Max occupancy (70-80%)     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  OUTPUT & DEPLOYMENT                     │
│  ┌──────────────────────┬──────────────────────────┐   │
│  │  Control Center      │  Passenger-Facing App    │   │
│  │  Dashboard           │  (BusMap Integration)    │   │
│  └──────────────────────┴──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Problem-Solution Mapping

| Problem | Proposed Solution |
|---------|-------------------|
| Fixed schedules don't reflect temporal demand variations | LSTM-based forecasting with historical data, GPS, weather, and events |
| Long waiting times due to capacity mismatches | ACO to optimize frequency, headways, and vehicle allocation |
| Suboptimal vehicle allocation across routes | ACO reallocation + shuttle services for high-demand segments |
| Reactive system lacking real-time adaptation | Real-time framework with streaming data and continuous updates |
| Poor passenger experience | Passenger-facing app with arrival times and capacity information |

---

## 3. Detailed Implementation Plan

### Project Timeline: 10 Weeks

```
Phase 1: Data Collection & Preprocessing       [Weeks 1-2]
Phase 2: LSTM Model Development                [Weeks 3-5]
Phase 3: ACO Schedule Optimization             [Weeks 6-8]
Phase 4: Evaluation & Deployment Proposal      [Weeks 9-10]
```

---

### PHASE 1: Data Collection & Preprocessing (2 weeks)

#### Week 1: Data Source Identification & Initial Collection

**Tasks:**

1. **Passenger Data Acquisition**
   - Contact HCMC Transport Authority for electronic ticketing data
   - If unavailable, implement synthetic data generation:
     - Poisson process for arrival patterns
     - Historical route patterns from xebuyt.net
     - Peak/off-peak hour distributions (7-9 AM, 5-7 PM peaks)
     - Weekend vs. weekday patterns

2. **GPS & Operational Data**
   - **Web scraping setup for xebuyt.net:**
     - Route information (120 routes)
     - Schedule data (departure times, headways)
     - Stop locations and sequences
     - Current timetables
   - **Google Maps API integration:**
     - Real-time travel time estimates
     - Route geometry data
     - Historical traffic patterns
     - Distance calculations

3. **Weather Data Integration**
   - OpenWeatherMap API setup and authentication
   - Historical weather data retrieval (12+ months)
   - Fields: temperature, rainfall, humidity, wind speed, conditions
   - Hourly granularity

4. **Event Data Collection**
   - Web crawler development for Vietnamese news sites:
     - VnExpress (vnexpress.net)
     - Tuoi Tre (tuoitre.vn)
     - Thanh Nien (thanhnien.vn)
   - Target events: concerts, sports, festivals, public holidays
   - Build event calendar with location tagging (geocoding)

**Deliverables:**
- Data collection scripts (Python)
- Initial raw datasets
- Data source documentation

---

#### Week 2: Data Preprocessing & Integration

**Tasks:**

1. **Data Cleaning Pipeline**
   - **Missing value handling:**
     - Time-series: Forward fill / interpolation
     - Categorical: Mode imputation or separate category
   - **Outlier removal:**
     - Z-score method (threshold: |z| > 3)
     - IQR method for non-normal distributions
   - **Timestamp standardization:**
     - Convert all timestamps to UTC+7 (Vietnam timezone)
     - Align to common intervals (5-minute bins)

2. **Feature Engineering**

   **Temporal Features:**
   - `hour_of_day` (0-23)
   - `day_of_week` (0-6, Monday=0)
   - `month` (1-12)
   - `is_weekend` (binary)
   - `is_holiday` (binary, Vietnamese calendar)
   - `is_rush_hour` (7-9 AM, 5-7 PM)
   - `time_since_last_bus` (minutes)

   **Spatial Features:**
   - `route_id` (categorical, 120 routes)
   - `stop_id` (categorical)
   - `distance_to_center` (km from District 1)
   - `district` (categorical, 24 districts)
   - Route embeddings (learned during training)

   **Contextual Features:**
   - `temperature` (°C, normalized)
   - `rainfall` (mm/hour, normalized)
   - `humidity` (%, normalized)
   - `is_raining` (binary)
   - `event_nearby` (binary, within 2km)
   - `event_distance` (km, if event exists)
   - `traffic_index` (0-1, from Google Maps)

3. **Database Setup**
   - **Install TimescaleDB:**
     - Extension of PostgreSQL for time-series data
     - Automatic data partitioning by time
   - **Schema design:**
     ```sql
     CREATE TABLE passenger_counts (
         time TIMESTAMPTZ NOT NULL,
         route_id INTEGER,
         stop_id INTEGER,
         boarding_count INTEGER,
         alighting_count INTEGER,
         occupancy INTEGER
     );

     CREATE TABLE gps_traces (
         time TIMESTAMPTZ NOT NULL,
         bus_id INTEGER,
         route_id INTEGER,
         latitude DOUBLE PRECISION,
         longitude DOUBLE PRECISION,
         speed DOUBLE PRECISION,
         heading DOUBLE PRECISION
     );

     CREATE TABLE weather_conditions (
         time TIMESTAMPTZ NOT NULL,
         temperature DOUBLE PRECISION,
         rainfall DOUBLE PRECISION,
         humidity DOUBLE PRECISION,
         wind_speed DOUBLE PRECISION,
         conditions VARCHAR(50)
     );

     CREATE TABLE events (
         event_id SERIAL PRIMARY KEY,
         name VARCHAR(255),
         location VARCHAR(255),
         latitude DOUBLE PRECISION,
         longitude DOUBLE PRECISION,
         start_time TIMESTAMPTZ,
         end_time TIMESTAMPTZ
     );

     SELECT create_hypertable('passenger_counts', 'time');
     SELECT create_hypertable('gps_traces', 'time');
     SELECT create_hypertable('weather_conditions', 'time');
     ```
   - **Create indexes** for efficient querying:
     ```sql
     CREATE INDEX idx_passenger_route ON passenger_counts(route_id, time DESC);
     CREATE INDEX idx_gps_bus ON gps_traces(bus_id, time DESC);
     ```

4. **Data Validation**
   - Statistical summaries (mean, std, min, max, quantiles)
   - Temporal consistency checks (no future timestamps)
   - Correlation analysis between features
   - Visualization of patterns (daily, weekly, seasonal)

**Deliverables:**
- Clean, integrated dataset
- TimescaleDB database with historical data
- Feature engineering pipeline (reusable)
- Data quality report

---

### PHASE 2: LSTM Model Development (3 weeks)

#### Week 3: Model Architecture Design

**Tasks:**

1. **Data Preparation for LSTM**
   - **Sliding window creation:**
     - Lookback: 168 hours (1 week) = 168 time steps
     - Forecast horizon: 6 steps (30 minutes in 5-min intervals)
   - **Train/Validation/Test split:**
     - Training: First 70% (temporal order preserved)
     - Validation: Next 15%
     - Test: Final 15%
   - **Feature normalization:**
     - MinMaxScaler for bounded features (0-1)
     - StandardScaler for unbounded features (z-score)
     - Fit only on training data, transform all splits

2. **LSTM Architecture Design**

   ```
   Input Layer
   ├── Temporal Input: (batch, 168, 1) - Historical demand
   ├── Route Input: (batch, 1) - Route ID
   ├── Stop Input: (batch, 1) - Stop ID
   └── Exogenous Input: (batch, 168, 10) - Weather, time features, etc.
         ↓
   Embedding Layers
   ├── Route Embedding: 120 routes → 32 dimensions
   └── Stop Embedding: ~500 stops → 16 dimensions
         ↓
   Feature Concatenation: [temporal + exogenous]
         ↓
   LSTM Layer 1: 128 units, return_sequences=True
         ↓
   Dropout: 0.2
         ↓
   LSTM Layer 2: 64 units, return_sequences=True
         ↓
   Dropout: 0.2
         ↓
   LSTM Layer 3: 32 units
         ↓
   Concatenate: [LSTM output + route_embedding + stop_embedding]
         ↓
   Dense Layer: 64 units, ReLU activation
         ↓
   Output Layer: 6 units (6 future time steps), ReLU activation
   ```

3. **Loss Function & Optimizer**
   - **Loss:** Mean Squared Error (MSE) or Huber Loss
   - **Optimizer:** Adam (learning_rate=0.001, beta_1=0.9, beta_2=0.999)
   - **Metrics:** MAE, MAPE for monitoring
   - **Early stopping:** patience=10 epochs on validation loss
   - **Learning rate schedule:** ReduceLROnPlateau (factor=0.5, patience=5)

**Deliverables:**
- Model architecture code
- Data preprocessing pipeline
- Training configuration

---

#### Week 4-5: Model Training & Hyperparameter Tuning

**Tasks:**

1. **Baseline Model Training**
   - Train initial model on 3-5 high-traffic routes
   - Monitor training metrics:
     - Training loss, validation loss
     - MAE, RMSE, MAPE
   - Validate convergence (loss plateau)
   - Compare against naive baselines:
     - Moving average (7-day, 14-day)
     - Same-hour-last-week
     - ARIMA/SARIMA

2. **Hyperparameter Optimization**

   **Search space:**
   - Number of LSTM layers: [2, 3, 4]
   - Units per layer: [32, 64, 128, 256]
   - Dropout rate: [0.1, 0.2, 0.3]
   - Lookback window: [24h, 72h, 168h]
   - Batch size: [32, 64, 128]
   - Learning rate: [0.0001, 0.001, 0.01]

   **Optimization approach:**
   - Grid search for critical parameters (layers, units)
   - Bayesian optimization (Optuna) for fine-tuning
   - 5-fold time-series cross-validation

   **Evaluation metric for selection:**
   - Validation MAPE (primary)
   - Validation RMSE (secondary)
   - Inference time (constraint: < 1 second per route)

3. **Multi-Route Training**
   - **Approach 1:** Single global model for all 120 routes
     - Pros: More training data, better generalization
     - Cons: May not capture route-specific patterns
   - **Approach 2:** Route-specific models
     - Pros: Captures unique route characteristics
     - Cons: Data scarcity for low-traffic routes
   - **Approach 3:** Hierarchical model (recommended)
     - Global model + route-specific fine-tuning
     - Transfer learning for low-data routes
     - Cluster similar routes (by demand profile)

4. **Exogenous Feature Integration**
   - **Ablation study:**
     - Baseline: Only historical demand
     - +Temporal: Add time features
     - +Weather: Add weather data
     - +Events: Add event indicators
     - Full model: All features
   - **Feature importance analysis:**
     - Permutation importance
     - SHAP values (for interpretability)
   - **Determine optimal feature set** (accuracy vs. complexity tradeoff)

5. **Model Validation**
   - **Test set evaluation:**
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)
     - MAPE (Mean Absolute Percentage Error)
     - R² score
   - **Target:** MAPE < 15% (>85% accuracy)
   - **Temporal validation:**
     - Forecast 15 min, 30 min, 60 min ahead
     - Analyze degradation over forecast horizon
   - **Stratified analysis:**
     - Peak vs. off-peak performance
     - Weekday vs. weekend
     - Weather conditions (normal, rain, heavy rain)
     - Special events
   - **Error distribution analysis:**
     - Identify systematic biases
     - Check for heteroscedasticity

**Deliverables:**
- Trained LSTM model(s)
- Hyperparameter tuning report
- Model evaluation report
- Serialized model files (.h5 or .pb)

---

### PHASE 3: ACO Schedule Optimization (3 weeks)

#### Week 6: ACO Algorithm Design

**Tasks:**

1. **Problem Formulation**

   **Decision Variables:**
   - `headway_r` ∈ [5, 60] minutes for each route r
   - `vehicles_r` ∈ [1, fleet_size] for each route r
   - `departure_times` for each trip on each route

   **Constraints:**
   - Total vehicles: Σ vehicles_r ≤ 2,052
   - Vehicle capacity: passengers per trip ≤ capacity (e.g., 50)
   - Driver shifts: Max 8 hours, breaks required
   - Minimum headway: ≥ 5 minutes (operational safety)
   - Maximum headway: ≤ 60 minutes (service quality)
   - Route length constraints: vehicles_r ≥ round_trip_time_r / headway_r

   **Objective Function:**
   ```
   Minimize: α × AvgWaitingTime + β × UnderutilizationPenalty

   Where:
   AvgWaitingTime = Σ_r (headway_r / 2) × demand_r / Σ_r demand_r

   UnderutilizationPenalty = Σ_r penalty(occupancy_r, target=0.75)

   occupancy_r = (demand_r × headway_r) / (vehicles_r × capacity)

   penalty(occ, target) =
     if occ < 0.7: (0.7 - occ)²
     elif occ > 0.9: 2 × (occ - 0.9)²
     else: 0

   α = 0.7 (waiting time weight)
   β = 0.3 (utilization weight)
   ```

2. **ACO Implementation Design**

   **Algorithm Parameters:**
   - Number of ants: 50
   - Iterations: 100
   - Evaporation rate (ρ): 0.1
   - Pheromone influence (α): 1.0
   - Heuristic influence (β): 2.0

   **Pheromone Matrix:**
   - Dimensions: [num_routes × num_headway_options]
   - Example: [120 × 56] (headways from 5 to 60 minutes)
   - Initial value: τ_0 = 1.0

   **Heuristic Information:**
   - η_ij = desirability of assigning headway j to route i
   - Based on forecasted demand:
     ```
     target_occupancy = 0.75
     expected_occupancy = (demand_i × headway_j) / capacity
     η_ij = 1 / (|expected_occupancy - target_occupancy| + ε)
     ```

   **Solution Construction:**
   - Each ant builds a complete schedule
   - For each route, select headway probabilistically:
     ```
     P_ij = (τ_ij^α × η_ij^β) / Σ_k (τ_ik^α × η_ik^β)
     ```
   - Check constraints after each assignment
   - Repair infeasible solutions (penalty function or greedy repair)

   **Pheromone Update:**
   ```
   τ_ij ← (1 - ρ) × τ_ij + Δτ_ij

   Δτ_ij = Σ_k (Q / cost_k) if ant k used (i, j)
            0 otherwise

   Q = quality constant (e.g., 100)
   cost_k = objective function value for ant k's solution
   ```

**Deliverables:**
- ACO algorithm specification
- Mathematical formulation document
- Pseudocode

---

#### Week 7: ACO Implementation & Testing

**Tasks:**

1. **ACO Algorithm Implementation**
   - Code ACO class in Python
   - Implement solution construction method
   - Implement constraint checking
   - Implement pheromone update rules
   - Add logging and monitoring

2. **Constraint Handling**
   - **Penalty functions:** Add large cost for constraint violations
   - **Feasibility repair operators:**
     - If fleet exceeded: Increase headways proportionally
     - If occupancy too high: Reduce headway (if vehicles available)
   - **Priority rules:**
     - High-demand routes get priority for vehicle allocation
     - Essential routes (hospitals, universities) have minimum service

3. **Unit Testing**
   - Test on single route scenarios
   - Test constraint satisfaction (fleet limit, capacity, etc.)
   - Verify solution quality improves over iterations
   - Test edge cases (very high demand, very low fleet)
   - Validate pheromone matrix updates

**Deliverables:**
- ACO implementation code
- Unit test suite
- Debugging logs

---

#### Week 8: Integration & Optimization

**Tasks:**

1. **LSTM-ACO Integration**
   - Connect LSTM forecast output to ACO input
   - Handle forecast uncertainty:
     - Use confidence intervals (mean ± 2×std)
     - Robust optimization (worst-case demand)
   - Build end-to-end pipeline:
     ```
     Input (current state) → LSTM forecast → ACO optimization → Output (new schedule)
     ```
   - Test on historical data (backtesting)

2. **Performance Optimization**
   - **Parallel ant evaluation:**
     - Use multiprocessing (Python `multiprocessing` module)
     - Distribute ants across CPU cores
   - **Caching:**
     - Cache demand forecasts (valid for 15 min)
     - Cache route distance calculations
   - **Lightweight heuristics for rapid updates:**
     - For < 15-minute updates: Use greedy adjustments
     - For major changes: Run full ACO

3. **Scenario Testing**
   - **Peak hour scenarios:**
     - 7-9 AM: Morning rush (university, office commutes)
     - 5-7 PM: Evening rush (return home)
   - **Special event scenarios:**
     - Stadium events (50,000+ attendees)
     - Festivals (Tet, Mid-Autumn)
     - Concerts at major venues
   - **Weather disruption scenarios:**
     - Heavy rain (increased demand for public transport)
     - Flooding (route closures)
   - **Comparison with fixed schedules:**
     - Baseline: Current HCMC bus schedules
     - Metrics: Waiting time, occupancy, cost

**Deliverables:**
- Integrated LSTM-ACO pipeline
- Performance benchmarks
- Scenario test results
- Comparison report (AI vs. fixed schedules)

---

### PHASE 4: Evaluation & Deployment Proposal (2 weeks)

#### Week 9: Comprehensive Evaluation

**Tasks:**

1. **Forecast Accuracy Evaluation**
   - **Per-route analysis:**
     - MAE, RMSE, MAPE for each of 120 routes
     - Identify best and worst performing routes
   - **Temporal breakdown:**
     - Peak vs. off-peak accuracy
     - Weekday vs. weekend
     - Month-by-month (detect seasonal patterns)
   - **Weather condition stratification:**
     - Clear weather
     - Light rain
     - Heavy rain
   - **Error distribution analysis:**
     - Histogram of errors
     - Check for bias (over/under-prediction)

2. **Schedule Performance Evaluation**

   **Waiting Time Analysis:**
   - Average waiting time: current vs. optimized
   - Target: ≥20% reduction
   - 95th percentile waiting times (worst-case)
   - Time-of-day breakdown (hourly)
   - Route-level breakdown

   **Occupancy Analysis:**
   - Average occupancy rate: target 70-80%
   - Peak load factor (max occupancy per trip)
   - Empty seat-kilometers reduction
   - Comparison: current (45%) vs. optimized (75%)

   **Operational Efficiency:**
   - Vehicle utilization rates (hours in service / total hours)
   - Total vehicle-hours required per day
   - Cost savings estimation:
     - Fuel: 12% reduction
     - Labor: 10% reduction
     - Maintenance: 8% reduction

3. **Comparative Analysis**
   - **Baseline vs. AI-optimized:**
     - A/B comparison on historical scenarios
   - **Statistical significance testing:**
     - Paired t-tests for waiting time
     - Chi-square test for occupancy distribution
   - **Sensitivity analysis:**
     - Vary demand by ±10%, ±20%
     - Vary fleet size by ±5%, ±10%
     - Analyze robustness of solutions

**Deliverables:**
- Comprehensive evaluation report
- Statistical analysis results
- Visualization dashboards

---

#### Week 10: Documentation & Deployment Roadmap

**Tasks:**

1. **Technical Documentation**
   - **System architecture document:**
     - Component diagrams
     - Data flow diagrams
     - Technology stack
   - **API specifications:**
     - RESTful API for forecast queries
     - WebSocket API for real-time updates
     - Authentication & authorization
   - **Model training procedures:**
     - Step-by-step guide
     - Hyperparameter recommendations
     - Troubleshooting guide
   - **Deployment requirements:**
     - Hardware: CPU, RAM, GPU, storage
     - Software: OS, dependencies, versions
     - Network: bandwidth, latency requirements

2. **Deployment Roadmap**

   **Phase 1 - Pilot (Months 1-2):**
   - Deploy on 5-10 high-traffic routes:
     - Route 1 (Ben Thanh - Suoi Tien)
     - Route 13 (Ben Thanh - Binh Tien)
     - Routes near universities
   - Objectives:
     - Validate real-world performance
     - Gather operator feedback
     - Identify integration issues

   **Phase 2 - Expansion (Months 3-4):**
   - Scale to 50 routes
   - Include diverse route types:
     - Urban core routes
     - Suburban routes
     - Express routes
   - Objectives:
     - Test scalability
     - Refine algorithms based on pilot feedback

   **Phase 3 - Full Deployment (Months 5-6):**
   - All 120 routes
   - Full integration with control center
   - BusMap integration
   - Objectives:
     - Achieve city-wide optimization
     - Demonstrate full cost savings

3. **Stakeholder Materials**
   - **Executive summary:**
     - 2-page overview for HCMC Transport Authority
     - Key findings, ROI analysis
   - **Cost-benefit analysis:**
     - Implementation costs (development, hardware, training)
     - Annual operational savings
     - Payback period (estimated 1-2 years)
   - **User interface mockups:**
     - Control center dashboard (Figma/Sketch)
     - Passenger app screens
   - **Training materials:**
     - Operator training manual
     - Video tutorials
     - FAQ document

4. **Risk Assessment & Mitigation**
   - **Data availability risks:**
     - Mitigation: Synthetic data generation, manual counts
   - **Model performance degradation:**
     - Mitigation: Weekly retraining, monitoring alerts
   - **System integration challenges:**
     - Mitigation: Phased rollout, API versioning
   - **Change management:**
     - Mitigation: Operator training, gradual transition

**Deliverables:**
- Technical documentation (100+ pages)
- Deployment roadmap
- Executive summary
- Cost-benefit analysis
- Training materials
- Final presentation (PowerPoint/PDF)

---

## 4. Technical Stack & Dependencies

### Programming Languages
- **Python 3.10+** (primary language)
- **SQL** (PostgreSQL/TimescaleDB queries)
- **JavaScript/TypeScript** (optional, for web dashboard)

### Machine Learning & Deep Learning

```python
# Core ML/DL Libraries
tensorflow==2.15.0          # LSTM implementation
keras==2.15.0               # High-level neural network API
scikit-learn==1.4.0         # Preprocessing, metrics
numpy==1.26.0               # Numerical operations
pandas==2.1.0               # Data manipulation

# Optimization
scipy==1.11.0               # Scientific computing
optuna==3.5.0               # Hyperparameter optimization
```

### Data Processing & Storage

```python
# Database
psycopg2==2.9.9             # PostgreSQL adapter
timescaledb                 # Time-series database extension

# Data Processing
apache-airflow==2.8.0       # Workflow orchestration
redis==5.0.1                # Caching layer
celery==5.3.4               # Distributed task queue
```

### Web Scraping & APIs

```python
requests==2.31.0            # HTTP requests
beautifulsoup4==4.12.0      # HTML parsing
selenium==4.16.0            # Dynamic web scraping
googlemaps==4.10.0          # Google Maps API client
```

### Visualization & Monitoring

```python
matplotlib==3.8.0           # Static plots
seaborn==0.13.0            # Statistical visualization
plotly==5.18.0             # Interactive plots
dash==2.14.0               # Dashboard framework
streamlit==1.29.0          # Alternative dashboard (simpler)

# Monitoring
mlflow==2.9.0              # Experiment tracking
prometheus-client==0.19.0   # Metrics collection
grafana                     # Metrics visualization (external)
```

### Testing & Quality Assurance

```python
pytest==7.4.0              # Unit testing
pytest-cov==4.1.0          # Coverage reporting
black==23.12.0             # Code formatting
flake8==7.0.0              # Linting
mypy==1.7.0                # Type checking
```

### Infrastructure & Deployment

```yaml
# Containerization
docker==24.0
docker-compose==2.23

# Cloud Services (Optional)
# - AWS: EC2, S3, RDS, Lambda
# - GCP: Compute Engine, Cloud Storage, Cloud SQL
# - Azure: VMs, Blob Storage, SQL Database
```

### Hardware Requirements

**Development Environment:**
- **CPU:** Intel i7/AMD Ryzen 7 (8+ cores)
- **RAM:** 32GB minimum
- **GPU:** NVIDIA RTX 3060+ (12GB VRAM) for LSTM training
- **Storage:** 500GB SSD

**Production Environment:**
- **CPU:** 16+ cores (Intel Xeon or AMD EPYC)
- **RAM:** 64GB+
- **GPU:** NVIDIA T4 or A10 (for real-time inference)
- **Storage:** 1TB SSD (for time-series data)
- **Network:** High-bandwidth connection for real-time data streams

---

## 5. Data Pipeline Architecture

### 5.1 Real-Time Data Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (External)                       │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│  E-Ticketing │  GPS Tracking│ OpenWeather  │  Event Websites  │
│    System    │  (xebuyt.net)│     API      │  (News Sites)    │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬─────────┘
       │              │              │                 │
       ↓              ↓              ↓                 ↓
┌─────────────────────────────────────────────────────────────────┐
│              DATA COLLECTORS (Apache Airflow DAGs)               │
├─────────────────────────────────────────────────────────────────┤
│  • Ticketing Collector (every 5 min)                            │
│  • GPS Collector (every 2 min)                                  │
│  • Weather Collector (every 30 min)                             │
│  • Event Scraper (daily)                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                 MESSAGE QUEUE (Apache Kafka/Redis)               │
│  Topics: ticketing_events, gps_updates, weather_data, events    │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│             STREAM PROCESSORS (Celery Workers)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Data Validation & Cleaning                                   │
│  • Feature Extraction                                           │
│  • Real-time Aggregation                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STORAGE LAYER (TimescaleDB)                      │
├─────────────────────────────────────────────────────────────────┤
│  Tables:                                                         │
│  • passenger_counts (route_id, stop_id, timestamp, count)       │
│  • gps_traces (bus_id, lat, lon, timestamp, speed)              │
│  • weather_conditions (timestamp, temp, rain, humidity)         │
│  • events (event_id, name, location, start_time, end_time)      │
│  • schedules (route_id, headway, vehicles, updated_at)          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Feature Engineering Pipeline

```python
# Airflow DAG: feature_engineering_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_temporal_features(**context):
    """Extract time-based features"""
    # Hour, day_of_week, is_weekend, is_holiday, etc.
    pass

def extract_spatial_features(**context):
    """Extract location-based features"""
    # Route embeddings, distance calculations, etc.
    pass

def extract_contextual_features(**context):
    """Extract context features"""
    # Weather aggregation, event proximity, etc.
    pass

def create_training_sequences(**context):
    """Create sliding window sequences for LSTM"""
    # Lookback window: 168 hours
    pass

default_args = {
    'owner': 'bus_optimization',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'feature_engineering',
    default_args=default_args,
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    catchup=False,
)

task1 = PythonOperator(
    task_id='extract_temporal',
    python_callable=extract_temporal_features,
    dag=dag,
)

task2 = PythonOperator(
    task_id='extract_spatial',
    python_callable=extract_spatial_features,
    dag=dag,
)

task3 = PythonOperator(
    task_id='extract_contextual',
    python_callable=extract_contextual_features,
    dag=dag,
)

task4 = PythonOperator(
    task_id='create_sequences',
    python_callable=create_training_sequences,
    dag=dag,
)

[task1, task2, task3] >> task4
```

### 5.3 Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING DATA PREPARATION                      │
├─────────────────────────────────────────────────────────────────┤
│  • Query historical data (last 12 months)                       │
│  • Train/Val/Test split (70/15/15)                              │
│  • Normalization (MinMax/Standard scaling)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LSTM MODEL TRAINING                         │
├─────────────────────────────────────────────────────────────────┤
│  • Hyperparameter tuning (Optuna)                               │
│  • Cross-validation (time-series split)                         │
│  • Early stopping (monitor val_loss)                            │
│  • Model checkpointing                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL EVALUATION & REGISTRY                    │
├─────────────────────────────────────────────────────────────────┤
│  • MLflow tracking (metrics, parameters, artifacts)             │
│  • Model versioning                                             │
│  • A/B testing framework                                        │
│  • Model deployment (if metrics > threshold)                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Inference & Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE LOOP                      │
└─────────────────────────────────────────────────────────────────┘
         ┌─────────────────────────────────────┐
         │  Every 15 minutes:                  │
         └─────────────────┬───────────────────┘
                           ↓
    ┌────────────────────────────────────────┐
    │  1. Fetch latest features from DB      │
    │     (last 168 hours + current context) │
    └────────────────┬───────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │  2. LSTM Model Prediction              │
    │     - Load model from cache            │
    │     - Generate 30-min forecasts        │
    │     - Output: demand per route/stop    │
    └────────────────┬───────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │  3. ACO Schedule Optimization          │
    │     - Input: forecasts + constraints   │
    │     - Run 50 ants × 100 iterations     │
    │     - Output: optimal schedules        │
    └────────────────┬───────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │  4. Schedule Update                    │
    │     - Write to schedules table         │
    │     - Trigger notifications            │
    │     - Update control center dashboard  │
    │     - Push to BusMap API               │
    └────────────────┬───────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │  5. Feedback Loop                      │
    │     - Collect actual passenger counts  │
    │     - Calculate forecast errors        │
    │     - Trigger retraining if needed     │
    └────────────────────────────────────────┘
```

---

## 6. Evaluation Metrics & Success Criteria

### 6.1 Demand Forecasting Performance

| Metric | Formula | Target | Excellent |
|--------|---------|--------|-----------|
| **MAE** | (1/n)Σ\|y_actual - y_pred\| | < 5 passengers | < 3 passengers |
| **RMSE** | √[(1/n)Σ(y_actual - y_pred)²] | < 8 passengers | < 5 passengers |
| **MAPE** | (100/n)Σ\|y_actual - y_pred\|/y_actual | < 15% | < 10% |
| **R² Score** | 1 - (SS_res/SS_tot) | > 0.85 | > 0.90 |

**Stratified Evaluation:**
- Peak hours (7-9 AM, 5-7 PM): MAPE < 12%
- Off-peak hours: MAPE < 18%
- Special events: MAPE < 20%
- Weather disruptions: MAPE < 25%

### 6.2 Schedule Optimization Performance

| Metric | Baseline | Target | Formula |
|--------|----------|--------|---------|
| **Avg. Waiting Time** | 15 min | < 12 min | Σ(headway_r / 2) × demand_r / Σdemand_r |
| **95th %ile Wait** | 30 min | < 24 min | Percentile calculation |
| **Avg. Occupancy** | 45% | 70-80% | (passengers / capacity) × 100 |
| **Peak Occupancy** | 85% | 75-85% | Max occupancy across all trips |
| **Empty Runs** | 25% | < 10% | (trips with <20% occupancy) / total trips |

### 6.3 Operational Efficiency

| Metric | Baseline | Target | Impact |
|--------|----------|--------|--------|
| **Vehicle-Hours** | 16,400/day | Reduce 10-15% | Cost savings |
| **Fuel Consumption** | Current | Reduce 12% | Environmental + cost |
| **Avg. Trip Distance** | 12 km | Maintain | Service coverage |
| **Route Coverage** | 120 routes | Maintain | Accessibility |

### 6.4 System Performance (Technical)

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| **Inference Latency** | < 30 seconds | Time from data fetch to schedule update |
| **Forecast Freshness** | 15-30 min | Age of predictions |
| **System Uptime** | > 99.5% | Downtime tracking |
| **Data Freshness** | < 5 min | Lag from event to database |
| **Model Retraining** | Weekly | Automated pipeline |

### 6.5 User Experience Metrics

**Control Center (Operators):**
- Dashboard load time: < 2 seconds
- Schedule update notification delay: < 1 minute
- Data visualization refresh: Real-time (< 10s)

**Passenger-Facing App:**
- Arrival time accuracy: ±2 minutes, 85% of the time
- App response time: < 1 second
- User satisfaction: > 4.0/5.0 (post-deployment survey)

### 6.6 Success Criteria Checklist

**Minimum Viable Product (MVP):**
- [ ] LSTM forecast accuracy: MAPE < 15% on test set
- [ ] ACO reduces waiting time by ≥ 15%
- [ ] System generates schedules within 2 minutes
- [ ] Database handles 1000+ transactions/second
- [ ] Dashboard displays real-time data

**Production-Ready System:**
- [ ] LSTM forecast accuracy: MAPE < 12% across all time periods
- [ ] Waiting time reduction: ≥ 20%
- [ ] Occupancy rates: 70-80% during peak and off-peak
- [ ] System uptime: > 99.5%
- [ ] Integration with BusMap API complete
- [ ] Operator training completed
- [ ] 3-month pilot successful on 10 routes

**Long-Term Excellence:**
- [ ] Forecast accuracy: MAPE < 10%
- [ ] Waiting time reduction: > 25%
- [ ] Occupancy optimization: 75-80% consistently
- [ ] Cost savings: > 15% reduction in operational costs
- [ ] Passenger satisfaction: > 4.2/5.0
- [ ] Deployment across all 120 routes

---

## 7. Code Templates & Starting Points

### 7.1 LSTM Model Architecture

```python
# models/lstm_forecaster.py

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate
from keras.models import Model

class BusDemandForecaster:
    def __init__(self, config):
        self.sequence_length = config['sequence_length']  # 168 hours
        self.num_routes = config['num_routes']  # 120
        self.num_stops = config['num_stops']
        self.prediction_horizon = config['prediction_horizon']  # 6 (30 min intervals)

    def build_model(self):
        # Temporal input (historical demand)
        temporal_input = keras.Input(
            shape=(self.sequence_length, 1),
            name='temporal_input'
        )

        # Categorical inputs
        route_input = keras.Input(shape=(1,), name='route_input')
        stop_input = keras.Input(shape=(1,), name='stop_input')

        # Exogenous features (weather, events, time features)
        exog_input = keras.Input(
            shape=(self.sequence_length, 10),
            name='exog_input'
        )

        # Embeddings for categorical features
        route_embedding = Embedding(
            input_dim=self.num_routes,
            output_dim=32
        )(route_input)
        route_embedding = tf.squeeze(route_embedding, axis=1)

        stop_embedding = Embedding(
            input_dim=self.num_stops,
            output_dim=16
        )(stop_input)
        stop_embedding = tf.squeeze(stop_embedding, axis=1)

        # Combine temporal and exogenous features
        combined_features = Concatenate(axis=2)([
            temporal_input,
            exog_input
        ])

        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True)(combined_features)
        dropout1 = Dropout(0.2)(lstm1)

        lstm2 = LSTM(64, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)

        lstm3 = LSTM(32)(dropout2)

        # Concatenate LSTM output with embeddings
        concat = Concatenate()([
            lstm3,
            route_embedding,
            stop_embedding
        ])

        # Dense layers
        dense1 = Dense(64, activation='relu')(concat)
        output = Dense(
            self.prediction_horizon,
            activation='relu',
            name='demand_output'
        )(dense1)

        model = Model(
            inputs=[temporal_input, route_input, stop_input, exog_input],
            outputs=output
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/best_lstm_model.h5',
            monitor='val_mae',
            save_best_only=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        return history
```

### 7.2 ACO Optimization Algorithm

```python
# optimization/aco_scheduler.py

import numpy as np
from typing import List, Dict, Tuple

class AntColonyScheduler:
    def __init__(self, config):
        self.num_ants = config['num_ants']  # 50
        self.num_iterations = config['num_iterations']  # 100
        self.evaporation_rate = config['evaporation_rate']  # 0.1
        self.alpha = config['alpha']  # Pheromone influence: 1.0
        self.beta = config['beta']  # Heuristic influence: 2.0
        self.fleet_size = config['fleet_size']  # 2052

        # Constraints
        self.min_headway = 5  # minutes
        self.max_headway = 60  # minutes
        self.vehicle_capacity = config['vehicle_capacity']  # e.g., 50 passengers

    def initialize_pheromones(self, num_routes):
        """Initialize pheromone matrix"""
        # Rows: routes, Columns: headway options (5-60 min)
        return np.ones((num_routes, 56))

    def calculate_heuristic(self, demand_forecast, route_id, headway):
        """Calculate heuristic information based on demand"""
        # Higher demand -> prefer shorter headway
        expected_passengers = demand_forecast[route_id]
        trips_per_hour = 60 / headway
        capacity_utilized = expected_passengers / (trips_per_hour * self.vehicle_capacity)

        # Heuristic favors headways that result in 70-80% capacity
        target_utilization = 0.75
        heuristic = 1.0 / (abs(capacity_utilized - target_utilization) + 0.1)

        return heuristic

    def construct_solution(self, pheromones, demand_forecast, routes):
        """Each ant constructs a complete schedule"""
        schedule = {}
        vehicles_used = 0

        for route_id in routes:
            # Calculate probabilities for each headway option
            probabilities = []

            for headway in range(self.min_headway, self.max_headway + 1):
                idx = headway - self.min_headway
                pheromone = pheromones[route_id, idx]
                heuristic = self.calculate_heuristic(
                    demand_forecast, route_id, headway
                )

                # ACO probability formula
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)

            # Normalize probabilities
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # Select headway based on probabilities
            headway = np.random.choice(
                range(self.min_headway, self.max_headway + 1),
                p=probabilities
            )

            # Calculate vehicles needed for this route
            route_length_hours = 2  # Example: 2-hour round trip
            vehicles_needed = int(np.ceil(route_length_hours * 60 / headway))

            # Check fleet constraint
            if vehicles_used + vehicles_needed <= self.fleet_size:
                schedule[route_id] = {
                    'headway': headway,
                    'vehicles': vehicles_needed
                }
                vehicles_used += vehicles_needed
            else:
                # Adjust to use remaining vehicles
                remaining = self.fleet_size - vehicles_used
                if remaining > 0:
                    adjusted_headway = int(route_length_hours * 60 / remaining)
                    schedule[route_id] = {
                        'headway': adjusted_headway,
                        'vehicles': remaining
                    }
                    vehicles_used = self.fleet_size
                    break

        return schedule

    def evaluate_solution(self, schedule, demand_forecast):
        """Calculate objective function value"""
        total_waiting_time = 0
        total_underutilization = 0

        for route_id, route_schedule in schedule.items():
            headway = route_schedule['headway']
            demand = demand_forecast[route_id]

            # Average waiting time = headway / 2
            avg_wait = headway / 2
            total_waiting_time += avg_wait * demand

            # Calculate underutilization
            trips_per_hour = 60 / headway
            vehicles = route_schedule['vehicles']
            capacity = trips_per_hour * self.vehicle_capacity
            utilization = demand / capacity if capacity > 0 else 0

            # Penalize if utilization < 70% or > 90%
            if utilization < 0.70:
                total_underutilization += (0.70 - utilization) * demand
            elif utilization > 0.90:
                total_underutilization += (utilization - 0.90) * demand * 2

        # Weighted objective (minimize)
        cost = 0.7 * total_waiting_time + 0.3 * total_underutilization
        return cost

    def update_pheromones(self, pheromones, all_solutions, all_costs):
        """Update pheromone matrix based on solution quality"""
        # Evaporation
        pheromones *= (1 - self.evaporation_rate)

        # Deposit pheromones (best ants deposit more)
        best_cost = min(all_costs)

        for solution, cost in zip(all_solutions, all_costs):
            # Deposit amount inversely proportional to cost
            deposit = best_cost / cost

            for route_id, route_schedule in solution.items():
                headway = route_schedule['headway']
                idx = headway - self.min_headway
                pheromones[route_id, idx] += deposit

        return pheromones

    def optimize(self, demand_forecast, routes):
        """Main ACO optimization loop"""
        num_routes = len(routes)
        pheromones = self.initialize_pheromones(num_routes)

        best_solution = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            all_solutions = []
            all_costs = []

            # Each ant constructs a solution
            for ant in range(self.num_ants):
                solution = self.construct_solution(
                    pheromones, demand_forecast, routes
                )
                cost = self.evaluate_solution(solution, demand_forecast)

                all_solutions.append(solution)
                all_costs.append(cost)

                # Track best solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

            # Update pheromones
            pheromones = self.update_pheromones(
                pheromones, all_solutions, all_costs
            )

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best cost = {best_cost:.2f}")

        return best_solution, best_cost
```

### 7.3 Data Collection Script

```python
# data_collection/gps_collector.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import psycopg2
from typing import List, Dict

class XeBuytScraper:
    def __init__(self, db_config):
        self.base_url = "https://xebuyt.net"
        self.db_config = db_config

    def get_route_list(self) -> List[str]:
        """Scrape list of all bus routes"""
        response = requests.get(f"{self.base_url}/tuyen")
        soup = BeautifulSoup(response.content, 'html.parser')

        routes = []
        for route_link in soup.find_all('a', class_='route-item'):
            route_id = route_link.get('href').split('/')[-1]
            routes.append(route_id)

        return routes

    def get_route_details(self, route_id: str) -> Dict:
        """Scrape detailed information for a specific route"""
        response = requests.get(f"{self.base_url}/tuyen/{route_id}")
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract route information
        route_data = {
            'route_id': route_id,
            'route_name': soup.find('h1', class_='route-name').text.strip(),
            'stops': [],
            'schedule': {},
            'scraped_at': datetime.now()
        }

        # Extract stops
        for stop in soup.find_all('div', class_='stop-item'):
            stop_name = stop.find('span', class_='stop-name').text.strip()
            route_data['stops'].append(stop_name)

        # Extract schedule
        schedule_table = soup.find('table', class_='schedule')
        if schedule_table:
            for row in schedule_table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    route_data['schedule'] = {
                        'first_trip': cols[0].text.strip(),
                        'last_trip': cols[1].text.strip(),
                        'headway': cols[2].text.strip()
                    }

        return route_data

    def save_to_database(self, route_data: Dict):
        """Save route data to PostgreSQL"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        # Insert route info
        cur.execute("""
            INSERT INTO routes (route_id, route_name, scraped_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (route_id) DO UPDATE
            SET route_name = EXCLUDED.route_name,
                scraped_at = EXCLUDED.scraped_at
        """, (
            route_data['route_id'],
            route_data['route_name'],
            route_data['scraped_at']
        ))

        # Insert stops
        for idx, stop_name in enumerate(route_data['stops']):
            cur.execute("""
                INSERT INTO stops (route_id, stop_name, stop_order)
                VALUES (%s, %s, %s)
                ON CONFLICT (route_id, stop_order) DO UPDATE
                SET stop_name = EXCLUDED.stop_name
            """, (route_data['route_id'], stop_name, idx))

        conn.commit()
        cur.close()
        conn.close()

    def collect_all_routes(self):
        """Main collection function"""
        routes = self.get_route_list()
        print(f"Found {len(routes)} routes")

        for route_id in routes:
            try:
                route_data = self.get_route_details(route_id)
                self.save_to_database(route_data)
                print(f"Collected data for route {route_id}")
            except Exception as e:
                print(f"Error collecting route {route_id}: {e}")

# Usage
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'bus_optimization',
        'user': 'postgres',
        'password': 'your_password'
    }

    scraper = XeBuytScraper(db_config)
    scraper.collect_all_routes()
```

### 7.4 Database Schema

```sql
-- database/schema.sql

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Routes table
CREATE TABLE routes (
    route_id INTEGER PRIMARY KEY,
    route_name VARCHAR(255) NOT NULL,
    route_type VARCHAR(50),
    distance_km DOUBLE PRECISION,
    avg_trip_time_min INTEGER,
    scraped_at TIMESTAMPTZ
);

-- Stops table
CREATE TABLE stops (
    stop_id SERIAL PRIMARY KEY,
    route_id INTEGER REFERENCES routes(route_id),
    stop_name VARCHAR(255) NOT NULL,
    stop_order INTEGER,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    UNIQUE(route_id, stop_order)
);

-- Passenger counts (time-series)
CREATE TABLE passenger_counts (
    time TIMESTAMPTZ NOT NULL,
    route_id INTEGER REFERENCES routes(route_id),
    stop_id INTEGER REFERENCES stops(stop_id),
    boarding_count INTEGER,
    alighting_count INTEGER,
    occupancy INTEGER,
    is_synthetic BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('passenger_counts', 'time');

-- GPS traces (time-series)
CREATE TABLE gps_traces (
    time TIMESTAMPTZ NOT NULL,
    bus_id INTEGER,
    route_id INTEGER REFERENCES routes(route_id),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    heading DOUBLE PRECISION
);

SELECT create_hypertable('gps_traces', 'time');

-- Weather conditions (time-series)
CREATE TABLE weather_conditions (
    time TIMESTAMPTZ NOT NULL,
    location VARCHAR(100) DEFAULT 'Ho Chi Minh City',
    temperature DOUBLE PRECISION,
    rainfall DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    conditions VARCHAR(50)
);

SELECT create_hypertable('weather_conditions', 'time');

-- Events table
CREATE TABLE events (
    event_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    location VARCHAR(255),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    expected_attendance INTEGER,
    event_type VARCHAR(50)
);

-- Schedules table (stores optimized schedules)
CREATE TABLE schedules (
    schedule_id SERIAL PRIMARY KEY,
    route_id INTEGER REFERENCES routes(route_id),
    headway_min INTEGER,
    vehicles_allocated INTEGER,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    optimization_cost DOUBLE PRECISION
);

-- Model performance tracking
CREATE TABLE model_performance (
    evaluation_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    evaluation_date TIMESTAMPTZ DEFAULT NOW(),
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    r2_score DOUBLE PRECISION,
    dataset_size INTEGER
);

-- Create indexes
CREATE INDEX idx_passenger_route_time ON passenger_counts(route_id, time DESC);
CREATE INDEX idx_gps_bus_time ON gps_traces(bus_id, time DESC);
CREATE INDEX idx_weather_time ON weather_conditions(time DESC);
CREATE INDEX idx_events_time ON events(start_time, end_time);
CREATE INDEX idx_schedules_route ON schedules(route_id, valid_from DESC);
```

---

## 8. Risk Assessment & Mitigation

### Critical Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| **Lack of real passenger data** | HIGH | HIGH | Generate synthetic data using Poisson processes + historical patterns from similar cities; validate with manual counts at select stops |
| **Model overfitting** | MEDIUM | MEDIUM | Cross-validation, regularization (L2, dropout), early stopping, ensemble methods |
| **ACO convergence issues** | MEDIUM | LOW | Tune parameters (α, β, ρ), implement local search operators, hybrid GA-ACO approach |
| **Real-time latency** | MEDIUM | MEDIUM | Model caching, GPU inference, lightweight heuristics for < 15 min updates |
| **Stakeholder adoption resistance** | HIGH | MEDIUM | Pilot program, operator training, demonstrate cost savings, gradual rollout with feedback |
| **API rate limits** (Google Maps, Weather) | LOW | MEDIUM | Caching strategies, request batching, backup API providers |
| **Data quality issues** | HIGH | HIGH | Validation rules, outlier detection, data imputation strategies, monitoring dashboards |
| **System integration challenges** | MEDIUM | MEDIUM | Phased integration, API versioning, comprehensive testing, rollback procedures |

### Technical Challenges

1. **Cold Start Problem:** New routes with no historical data
   - **Solution:** Transfer learning from similar routes (by demand profile, geography), use city-wide patterns, bootstrap with synthetic data

2. **Concept Drift:** Demand patterns change over time (new metro lines, urban development)
   - **Solution:** Weekly model retraining, online learning algorithms, ensemble of recent models, drift detection mechanisms

3. **Scalability:** 120 routes × 1000+ stops = large state space
   - **Solution:** Route clustering (similar demand patterns), hierarchical optimization, distributed computing (Ray, Dask)

4. **Real-time Performance:** Need <30s latency for inference + optimization
   - **Solution:** Model quantization, TensorRT optimization, pre-computed schedules with rapid adjustments

5. **Missing Data:** GPS outages, sensor failures
   - **Solution:** Imputation algorithms (KNN, interpolation), redundant data sources, graceful degradation

---

## 9. Next Steps & Deliverables

### Immediate Actions (Week 1)

1. **Environment Setup**
   - Install Python 3.10+, TensorFlow, PostgreSQL, TimescaleDB
   - Set up Git repository with proper structure
   - Configure virtual environment (venv or conda)

2. **Data Acquisition**
   - Begin xebuyt.net scraping for route information
   - Register for OpenWeatherMap API key
   - Set up Google Maps API access (if available)

3. **Project Structure**
   ```
   bus-scheduling-optimization/
   ├── data/
   │   ├── raw/
   │   ├── processed/
   │   └── synthetic/
   ├── models/
   │   ├── lstm_forecaster.py
   │   └── saved_models/
   ├── optimization/
   │   └── aco_scheduler.py
   ├── data_collection/
   │   ├── gps_collector.py
   │   ├── weather_collector.py
   │   └── event_scraper.py
   ├── preprocessing/
   │   └── feature_engineering.py
   ├── evaluation/
   │   └── metrics.py
   ├── database/
   │   └── schema.sql
   ├── notebooks/
   │   └── eda.ipynb
   ├── tests/
   ├── docs/
   ├── requirements.txt
   └── README.md
   ```

4. **Database Setup**
   - Install PostgreSQL and TimescaleDB
   - Create database schema (see Section 7.4)
   - Test data insertion and querying

5. **Initial Data Collection**
   - Run route scraper
   - Collect 1 week of weather data
   - Document data sources and limitations

### Key Deliverables by Phase

**Phase 1 (Weeks 1-2):**
- [ ] Clean, integrated dataset (passenger, GPS, weather, events)
- [ ] TimescaleDB database with 12+ months of historical data
- [ ] Feature engineering pipeline
- [ ] Exploratory data analysis report (Jupyter notebook)
- [ ] Data quality assessment

**Phase 2 (Weeks 3-5):**
- [ ] Trained LSTM model (>85% accuracy target)
- [ ] Model evaluation report (MAE, RMSE, MAPE by route, time, condition)
- [ ] Hyperparameter tuning results and recommendations
- [ ] Model serialization for deployment (.h5 files)
- [ ] Training scripts and documentation

**Phase 3 (Weeks 6-8):**
- [ ] ACO optimization algorithm implementation
- [ ] Integrated LSTM-ACO pipeline (end-to-end)
- [ ] Performance benchmarks (vs. fixed schedules)
- [ ] Simulation results on historical data
- [ ] Sensitivity analysis report

**Phase 4 (Weeks 9-10):**
- [ ] Comprehensive evaluation report (30+ pages)
- [ ] Dashboard prototype (Streamlit or Dash)
- [ ] Deployment roadmap document (phased rollout plan)
- [ ] Final presentation for HCMC Transport Authority (PowerPoint)
- [ ] Technical documentation (architecture, APIs, procedures)
- [ ] Research paper draft (optional, for academic publication)

### Long-Term Roadmap (Post-Development)

**Months 1-2 (Pilot):**
- Deploy on 5-10 routes
- Monitor performance daily
- Collect operator and passenger feedback
- Identify and fix bugs

**Months 3-4 (Expansion):**
- Scale to 50 routes
- Refine algorithms based on pilot learnings
- Develop operator training program
- Integrate with BusMap

**Months 5-6 (Full Deployment):**
- All 120 routes
- Full integration with control center
- Launch passenger-facing features
- Continuous monitoring and optimization

**Year 2+:**
- Expand to other cities in Vietnam
- Add advanced features (multi-modal integration, electric bus optimization)
- Real-time passenger information system
- Predictive maintenance integration

---

## Summary

This implementation plan provides a complete, actionable roadmap for developing the AI-driven bus scheduling optimization system for Ho Chi Minh City. The plan encompasses:

- **4-phase, 10-week development timeline** with clear milestones
- **Detailed technical architecture** integrating LSTM forecasting and ACO optimization
- **Comprehensive data pipeline** supporting real-time operations
- **Code templates** for immediate development kickstart
- **Quantified success metrics** (>85% forecast accuracy, 20%+ wait time reduction, 70-80% occupancy)
- **Risk mitigation strategies** for common challenges
- **Scalable infrastructure** for 120 routes and 250,000+ daily passengers

### Expected Impact

- **Passenger Experience:** Reduced wait times, more reliable service, real-time information
- **Operational Efficiency:** 10-15% cost reduction, better vehicle utilization
- **Environmental:** Lower emissions through optimized routing
- **Economic:** Improved public transport attractiveness, reduced private vehicle usage
- **Scalability:** Framework adaptable to other Vietnamese cities

### Success Factors

1. **Data Quality:** Ensure clean, comprehensive datasets
2. **Stakeholder Buy-in:** Engage operators and authority early
3. **Iterative Development:** Start small, validate, scale
4. **Continuous Monitoring:** Track performance, retrain models
5. **User-Centric Design:** Focus on both operators and passengers

---

**Next Step:** Begin Phase 1 - Data Collection & Preprocessing

**Contact:** Pham Quang Nhan (2470885) | Supervisor: Assoc. Prof. Dr. Tran Minh Quang
**Institution:** HCMC University of Technology - Faculty of Computer Science and Engineering

---

*Document Version: 2.0*
*Last Updated: January 2025*
*Status: Ready for Implementation*
