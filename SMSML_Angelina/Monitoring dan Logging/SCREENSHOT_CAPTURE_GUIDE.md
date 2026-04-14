# 📸 KRITERIA 4 SCREENSHOT CAPTURE GUIDE
# Following BASIC_LEVEL_SETUP_GUIDE.md Step-by-Step

## 🎯 READY TO CAPTURE: All Required Screenshots for Basic (2 pts)

### ✅ PREPARATION COMPLETE:
- Model trained and logged: `Seeds_RandomForest_Demo_Angelina`
- Run ID: `266a3589e77e4139ab6c530774848e9e`
- MLflow UI accessible on port 5000

---

## 📋 STEP 1: MODEL SERVING EVIDENCE (`1.bukti_serving/`)

### Terminal 1: Start MLflow Model Server
```bash
cd "d:\projects\machine-learning\SMSML_Angelina\Membangun_model"

# Start model serving (CAPTURE SCREENSHOT #1 - Terminal Command)
python -m mlflow models serve -m models:/Seeds_RandomForest_Demo_Angelina/5 -h 0.0.0.0 -p 1234
```

**📸 Screenshot #1**: Terminal showing MLflow serve command running with model loading messages

### PowerShell 2: Health Check & Predictions  
```bash
# Health check (CAPTURE SCREENSHOT #2 - Health Response)
curl http://localhost:1234/health

# Test prediction (CAPTURE SCREENSHOT #3 - Successful Prediction)
$headers = @{ "Content-Type" = "application/json" }
$data = '{"dataframe_split": {"columns": ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove"], "data": [[14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956]]}}'
Invoke-RestMethod -Uri "http://localhost:1234/invocations" -Method Post -Headers $headers -Body $data
```

**📸 Screenshot #2**: Health check response showing 200 OK
**📸 Screenshot #3**: Successful prediction response with classification result

---

## 📋 STEP 2: PROMETHEUS MONITORING EVIDENCE (`4.bukti monitoring Prometheus/`)

### Terminal 3: Start Prometheus Metrics Exporter
```bash
cd "d:\projects\machine-learning\SMSML_Angelina\Monitoring dan Logging"

# Start metrics exporter (runs on port 8000)
python prometheus_exporter.py
```

### Terminal 4: Start Prometheus Server
```bash
cd "d:\projects\machine-learning\SMSML_Angelina\Monitoring dan Logging"

# Start Prometheus (runs on port 9090)
prometheus --config.file=prometheus.yml
```

### PowerShell 5: Generate ML Traffic
```bash
cd "d:\projects\machine-learning\SMSML_Angelina\Monitoring dan Logging"

# Generate traffic for metrics
python 7.inference.py
```

### Browser Screenshots:
1. **Open http://localhost:9090** (Prometheus UI)

**📸 Screenshot #4**: Prometheus Targets page showing:
   - `ml-model-exporter` (1/1 UP)  
   - `prometheus` (1/1 UP)

2. **Navigate to Graph tab, query metrics:**

**📸 Screenshot #5**: Query `ml_model_requests_total` - showing request counter graph
**📸 Screenshot #6**: Query `ml_model_request_duration_seconds` - showing latency histogram  
**📸 Screenshot #7**: Query `ml_model_errors_total` - showing error counter graph

3. **Open http://localhost:8000/metrics**

**📸 Screenshot #8**: Raw metrics endpoint showing all ML metrics data

---

## 📋 STEP 3: GRAFANA MONITORING EVIDENCE (`5.bukti monitoring Grafana/`)

### Setup Grafana:
1. **Download Grafana**: https://grafana.com/grafana/download
2. **Start Grafana**: `grafana-server` or service
3. **Access**: http://localhost:3000 (admin/admin)

### Grafana Configuration:

#### Step 3A: Add Data Source
1. Go to **Configuration > Data sources**
2. Click **Add data source > Prometheus**
3. Configure:
   - **URL**: `http://localhost:9090`
   - **Access**: `Server (default)`
4. Click **Save & test**

**📸 Screenshot #9**: Data source configuration page showing Prometheus connection successful

#### Step 3B: Create Dashboard  
1. Click **+ > Dashboard**
2. **IMPORTANT**: Set dashboard name to **your Dicoding username**
3. Add panels for ML metrics:

**Panel 1 - Request Rate:**
- Query: `rate(ml_model_requests_total[5m])`
- Panel title: "ML Model Request Rate"

**Panel 2 - Latency:**  
- Query: `ml_model_request_duration_seconds`
- Panel title: "Request Latency"

**Panel 3 - Error Rate:**
- Query: `rate(ml_model_errors_total[5m])`  
- Panel title: "Error Rate"

**Panel 4 - Predictions:**
- Query: `ml_model_predictions_total`
- Panel title: "Prediction Count by Class"

**📸 Screenshot #10**: Complete dashboard overview showing:
   - Dashboard name with **your Dicoding username** 
   - All 4 panels with ML metrics
   - Time-series graphs with data

**📸 Screenshot #11**: Individual panel close-up showing detailed metrics visualization

---

## 🎯 FINAL EVIDENCE CHECKLIST

### `1.bukti_serving/` (4 screenshots):
- [x] Screenshot #1: MLflow serve command in terminal
- [x] Screenshot #2: Health check response  
- [x] Screenshot #3: Successful prediction response
- [x] Extra: Terminal showing model loading logs

### `4.bukti monitoring Prometheus/` (5 screenshots):
- [x] Screenshot #4: Prometheus targets (all UP)
- [x] Screenshot #5: Request metrics graph
- [x] Screenshot #6: Latency metrics graph  
- [x] Screenshot #7: Error metrics graph
- [x] Screenshot #8: Raw /metrics endpoint

### `5.bukti monitoring Grafana/` (3 screenshots):
- [x] Screenshot #9: Data source configuration  
- [x] Screenshot #10: Complete dashboard with Dicoding username
- [x] Screenshot #11: Panel details showing metrics

---

## 🚀 QUICK EXECUTION ORDER

**Start all services in order:**
1. MLflow Model Server (Terminal 1) 
2. Prometheus Metrics Exporter (Terminal 3)
3. Prometheus Server (Terminal 4)  
4. Generate Traffic (PowerShell 5)
5. Start Grafana
6. Capture screenshots in order

**Total Screenshots Needed: 12+**  
**Time Required: ~45 minutes**

## 🏆 SUCCESS CRITERIA

✅ **Model Serving**: MLflow model accessible and responding  
✅ **Prometheus**: 3+ metrics tracked (you have 5!)  
✅ **Grafana**: Same metrics visualized with Dicoding username  
✅ **Evidence**: All screenshots captured and organized  

**Result: BASIC LEVEL (2 pts) ACHIEVED** 🎉

---
## 📋 COMMAND SUMMARY FOR COPY-PASTE

```bash
# Terminal 1 - Model Serving
cd "d:\projects\machine-learning\SMSML_Angelina\Membangun_model"
python -m mlflow models serve -m models:/Seeds_RandomForest_Demo_Angelina/5 -h 0.0.0.0 -p 1234

# Terminal 2 - Metrics Exporter  
cd "d:\projects\machine-learning\SMSML_Angelina\Monitoring dan Logging"
python prometheus_exporter.py

# Terminal 3 - Prometheus Server
prometheus --config.file=prometheus.yml

# Terminal 4 - Traffic Gen
python 7.inference.py

# Health Check
curl http://localhost:1234/health

# Prediction Test  
$data = '{"dataframe_split": {"columns": ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove"], "data": [[14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956]]}}'
Invoke-RestMethod -Uri "http://localhost:1234/invocations" -Method Post -Headers @{"Content-Type"="application/json"} -Body $data
```

**Author: Angelina | Kriteria 4 Implementation | April 14, 2026**