# Chicago Crime — Big Data Analytics with PySpark

A single Spark pipeline over the **City of Chicago "Crimes – 2001 to Present"** dataset (≈7.8 M incidents, 1.84 GB CSV), enriched with **socio-economic indicators** for the 77 community areas and **daily weather** (Open-Meteo / ERA5). The notebook (`7002.ipynb`) covers four tasks:

1. **Distributed data engineering** — schema contract, typed casting, quality rules, quarantine, partitioned Parquet lake
2. **High-dimensional EDA with Spark SQL** — temporal, spatial, socio-economic, rolling-window and weather analysis; broadcast joins
3. **Unsupervised learning** — spatio-temporal hotspot clustering (K-Means vs Bisecting K-Means) and alignment with police districts
4. **Distributed predictive modelling** — arrest prediction (binary) and primary-type prediction (multiclass) under class imbalance

---

## Contents

- [Environment & configuration](#environment--configuration)
- [Data sources](#data-sources)
- [Task 1 — Data engineering pipeline](#task-1--distributed-data-engineering--pipeline-construction)
- [Task 2 — Exploratory analysis (Spark SQL)](#task-2--high-dimensional-exploratory-data-analysis-spark-sql)
- [Task 3 — Hotspot clustering](#task-3--unsupervised-learning-spatio-temporal-hotspot-clustering)
- [Task 4 — Predictive modelling](#task-4--distributed-predictive-modelling)
- [Key findings](#key-findings)
- [Known issues & caveats](#known-issues--caveats)
- [Outputs](#outputs)

---

## Environment & configuration

| Item | Value |
|---|---|
| Runtime | Google Colab (also runs locally) |
| Spark | 3.5.3, `local[*]`, 2 cores in the recorded run |
| Java | OpenJDK 17 (installed automatically on Colab) |
| Python libs | `pyspark`, `kagglehub`, `pandas`, `numpy`, `matplotlib`, `scipy` (optional) |
| Working dir | `/content/chicago_crime` on Colab, `./chicago_crime` locally (override with `CC_BASE`) |

**Spark session settings:** driver memory `8g`, 64 shuffle partitions, adaptive query execution on, session time zone `UTC` (so Chicago's local wall-clock timestamps are not shifted by DST), `CORRECTED` time-parser policy, Snappy Parquet compression.

**Lake layout**

```
chicago_crime/
├── raw/            # landing zone (immutable): crime/, weather/, socio/
├── lake/
│   ├── crimes_bronze_typed/   # typed copy of the CSV
│   ├── crimes_silver/         # curated, partitioned by year / district
│   ├── crimes_quarantine/     # rejected rows + reason
│   ├── weather_daily/         # partitioned by year
│   └── socio_economic/
├── figures/        # 12 PNGs
└── tables/         # 22 CSVs
```

**Tunable knobs** (environment variables in brackets)

| Parameter | Default | Purpose |
|---|---|---|
| `CLUSTER_YEAR` | latest complete year | Year used for clustering |
| `CLUSTER_SAMPLE_FRAC` (`CC_CLUSTER_FRAC`) | 1.0 | Sample fraction for clustering |
| `K_RANGE` | 2–15 | K values evaluated |
| `TIME_WEIGHT` | 1.0 | Weight of time-of-day vs. space in clustering |
| `PURITY_THRESHOLD` | 0.60 | Cluster is "cross-boundary" if < 60 % sits in one district |
| `MODEL_YEARS` | 5 | Most recent N years used for modelling |
| `MODEL_SAMPLE_FRAC` (`CC_MODEL_FRAC`) | 0.30 | Sample fraction for modelling |
| `RF_TREES` (`CC_RF_TREES`) | 60 | Random-forest trees |
| `GBT_ITERS` (`CC_GBT_ITERS`) | 40 | GBT iterations |
| `TOP_LOCATIONS` / `TOP_TYPES` | 40 / 10 | Category caps (rest → `OTHER`) |
| `DRIVER_MEMORY` (`CC_DRIVER_MEM`) | `8g` | Spark driver memory |
| `SEED` | 42 | Reproducibility |

---

## Data sources

| Source | Access | Size | Role |
|---|---|---|---|
| Crimes 2001–present | Kaggle `utkarshx27/crimes-2001-to-present` (via `kagglehub`) or `CC_CRIME_CSV` | 1.84 GB, 7,784,664 rows | Primary fact table |
| Census socio-economic indicators (`kn9c-c2s2`) | Chicago Data Portal (SODA CSV) | 77 community areas | Hardship, poverty, unemployment, income, crowding, education; also supplies community-area names |
| Daily weather | Open-Meteo archive API (ERA5), point 41.88 N, −87.63 E | 8,146 days | Temperature, precipitation, snow, wind |

---

## Task 1 — Distributed Data Engineering & Pipeline Construction

### 1.1 Ingestion
The CSV is read in `PERMISSIVE` mode (a malformed line never aborts the job), with every column as a string. Column names are normalised to `snake_case`.

### 1.2 Data contract
A 21-column contract declares the target type and whether each field is required (`id`, `date`, `primary_type`, `arrest` are required). Missing columns are added as null; unexpected columns are dropped (the source's redundant `location` column was dropped). Every string is trimmed and blanks become null before casting.

### 1.3 Typed casting with failure accounting
- Timestamps are parsed against six candidate formats using `try_to_timestamp` + `coalesce`.
- Booleans accept `TRUE/T/Y/YES/1` and `FALSE/F/N/NO/0`.
- Numerics use `try_cast`.
- A cast failure = raw value present but typed value null; all failure counts are computed in **one aggregation pass**.
- The typed "bronze" layer is materialised once as Parquet (161 s) so later passes don't re-parse 1.8 GB of text.

**Result:** 7,784,664 raw rows, **0 cast failures** in every column.

### 1.4 Quality rules

| Rule | What it does |
|---|---|
| (a) Duplicates | Keep the latest `updated_on` version of each `id` |
| (b) Required / range | Missing id/date/type/arrest or date outside 2001-01-01…now → quarantine with a `reject_reason` |
| (c) Year consistency | `year` recomputed from the timestamp (timestamp is authoritative) |
| (d) Text canonicalisation | Upper-case, whitespace collapse, merge known variants (e.g. `NON - CRIMINAL` → `NON-CRIMINAL`, `CRIM SEXUAL ASSAULT` → `CRIMINAL SEXUAL ASSAULT`); unknown location → `UNKNOWN` |
| (e) Administrative codes | Invalid community area (1–77), ward (1–50), beat (100–3200) → null. District repaired from beat (beat 1134 → district 11) |
| (f) Coordinates | Invalid if missing, (0,0) or outside the Chicago bounding box. Repaired with the beat's median centroid; community area imputed from the beat's modal area. Tagged `loc_quality` = `valid` / `imputed_beat` / `missing` |
| (g) Feature derivation | `date`, `month`, `hour`, `dow`, `is_weekend`, `season`, `tod_band`, cyclic `hour_sin`/`hour_cos`, `ingest_ts` |
| (h) Privacy by design | Drops `case_number`, `block`, `x_coordinate`, `y_coordinate` (data minimisation, UK GDPR Art. 5(1)(c)) |

### 1.5 Curated lake
Silver data is written as Parquet partitioned by `year` / `district` (531 files, 361 s). Quarantined rows go to a separate path.

**raw 7,784,664 → silver 7,784,664 | quarantined 0**

### 1.6 Data-quality report

| Rule | Rows affected | % of raw |
|---|---:|---:|
| community_area_imputed_from_beat | 613,547 | 7.881 |
| district_missing_or_inconsistent | 337,299 | 4.333 |
| invalid_or_missing_coordinates | 86,966 | 1.117 |
| primary_type_variants_merged | 3 | 0.000 |
| location_desc_variants_merged | 2 | 0.000 |
| duplicate_id_versions_removed | 0 | 0.000 |
| year_mismatch_fixed | 0 | 0.000 |


### 1.7 Storage & partition-pruning benchmark
Query: one year (`max year − 1`) and the busiest district in that year.

| Format | Size (GB) | Query (s) | Rows matched |
|---|---:|---:|---:|
| Raw CSV (full scan) | 1.838 | 29.29 | 14,768 |
| Parquet partitioned (pruned) | 0.000* | 8.65 | 14,768 |

Both return identical row counts; Parquet is **≈3.4× faster**. *See [Known issues](#known-issues--caveats) for the zero size / empty `PartitionFilters` reading.

### 1.8 Secondary source A — socio-economic indicators
Column names mapped to a stable schema, numeric fields cast with `try_cast`, the city-wide "CHICAGO" total row removed, deduplicated by community area. **77 areas, no nulls** in any indicator. The `community_area` / `community_area_name` pair doubles as the small lookup table for broadcast joins.

### 1.9 Secondary source B — daily weather
Downloaded once for the full date range of the crime data and cached as JSON. Quality steps: temperature/wind gaps filled with the mean of neighbouring days; precipitation/snow nulls → 0 and negatives clamped to 0 (no gaps were present). Weather "event" flags:

| Event | Threshold | Days |
|---|---|---:|
| heavy_rain | precip ≥ 10 mm | 753 |
| snow_day | snow ≥ 2 cm | 254 |
| hot_day | tmax ≥ 30 °C | 292 |
| freezing_day | tmax ≤ −5 °C | 374 |
| windy_day | wind ≥ 45 km/h | 69 |

---

## Task 2 — High-Dimensional Exploratory Data Analysis (Spark SQL)

Silver, weather, socio and community-name tables are registered as temp views. Data covers **2001–2023**; the last complete year is **2022** (used as the cut-off for annual comparisons).

### 2.1 Temporal structure
- Most crime types declined substantially from 2001 to 2020; theft and motor-vehicle theft rebounded in 2021–2022.
- Narcotics fell the most sharply (≈57 k in 2004 to ≈5 k in 2022).
- Weekday × hour heatmap: a trough at 04:00–06:00, a spike at 12:00 and midnight (default time stamps), and the heaviest activity on Friday/Saturday evenings.


### 2.2 Spatial structure & socio-economic correlation
Per-area aggregates (crimes/year, arrest rate, domestic share, violent share) joined to socio-economic indicators; Spearman ρ across 77 areas:

| Indicator | crimes/year | violent share | arrest rate |
|---|---:|---:|---:|
| hardship_index | 0.15 | 0.74 | 0.58 |
| pct_below_poverty | 0.37 | **0.82** | **0.73** |
| pct_unemployed | 0.20 | **0.87** | 0.52 |
| per_capita_income | −0.14 | −0.70 | −0.55 |
| pct_housing_crowded | 0.18 | 0.27 | 0.39 |
| pct_no_hs_diploma | 0.07 | 0.41 | 0.41 |

Deprivation is only weakly linked to **crime volume** but strongly linked to the **violent share** and to the **arrest rate**.


### 2.3 Rolling crime rates per community area (window functions)
A full (area × month) calendar grid is generated first so that `ROWS`-based windows don't silently span missing months; missing months become 0. Monthly counts are converted to incidents/day, then:

- `roll3_rate`, `roll12_rate` — 3- and 12-month rolling means
- `prev12_mean`, `prev12_sd` — trailing 12-month baseline (excluding current month)
- `z_score` — anomaly vs. baseline; flagged when |z| ≥ 3 and n ≥ 20

Top anomalies (all in 2001–2002):

| Community area | Month | n | Daily rate | Prev-12 mean | z |
|---|---|---:|---:|---:|---:|
| Oakland | 2002-04 | 45 | 1.50 | 0.04 | 38.2 |
| Grand Boulevard | 2001-03 | 666 | 21.48 | 19.27 | 32.7 |
| Fuller Park | 2002-04 | 51 | 1.70 | 0.04 | 31.6 |
| West Town | 2001-03 | 1,360 | 43.87 | 41.62 | 30.0 |
| Hermosa | 2002-04 | 83 | 2.77 | 0.16 | 26.7 |


### 2.4 Weather events vs crime frequency
Because crime has strong seasonal and weekly cycles, each day is compared with its **expected level** (mean of the same year-month-weekday); temperature is expressed as an anomaly from the monthly mean.

- Temperature vs daily crime: raw r = 0.23; **deseasonalised r = 0.37** — warmer-than-usual days carry more crime even after removing the season.
- Event days relative to expected (Welch t-test):

| Event | Days | % vs expected (event days) | % vs expected (other days) | p |
|---|---:|---:|---:|---:|
| heavy_rain | 744 | −2.84 | +0.29 | < 0.0001 |
| snow_day | 249 | −5.90 | +0.19 | < 0.0001 |
| hot_day | 292 | +1.36 | −0.05 | < 0.0001 |
| freezing_day | 371 | **−7.38** | +0.36 | < 0.0001 |
| windy_day | 66 | −3.59 | +0.03 | 0.0157 |

By type: freezing days cut criminal damage (−12 %) and burglary (−10 %); snow cuts narcotics (−10 %); hot days lift battery (+4 %).


### 2.5 Broadcast variables & broadcast joins
Joining 77 community names onto the crime table:

| Approach | Seconds |
|---|---:|
| DataFrame sort-merge join (auto-broadcast disabled) | 0.13 |
| DataFrame broadcast hash join (`F.broadcast`) | **0.05** |
| RDD `reduceByKey` + broadcast variable + accumulator | 35.83 |

The physical plans confirm `SortMergeJoin` vs `BroadcastHashJoin`. The RDD version shows an explicit broadcast variable and an accumulator (5 records had no community name) but is far slower because rows leave the JVM for Python.


---

## Task 3 — Unsupervised Learning: Spatio-Temporal Hotspot Clustering

**Input:** 233,124 incidents from 2022 with valid coordinates.
**Features:** `latitude`, `longitude`, `hour_sin`, `hour_cos` → `StandardScaler` → `ElementwiseProduct` (time weight). Cyclic encoding keeps 23:00 next to 00:00.

### 3.1 Choosing K
K-Means and Bisecting K-Means fitted for K = 2…15; WSSSE (elbow, located automatically with a Kneedle-style method) and silhouette evaluated.

**Elbow K = 6, best-silhouette K = 6 → K = 6 used.**

### 3.2 Final model & alignment with police districts
Silhouette **0.413**. For each cluster: size, peak hour (circular mean), time concentration, 1σ area, density, dominant district and its share ("purity"), arrest rate, top crime types. A cluster is a **high-activity zone (HAZ)** if its density ≥ median, and **not aligned** if its district purity < 0.60.

| Cluster | n | Peak hour | Area km² | Density /km² | Main district | Purity | Arrest rate | Top types | HAZ not aligned |
|---:|---:|---:|---:|---:|---:|---:|---:|---|:-:|
| 5 | 49,222 | 18.6 | 73.0 | 674.6 | 12 | 0.105 | 0.141 | Theft, Battery, Criminal damage | ✔ |
| 4 | 35,854 | 23.0 | 58.0 | 618.5 | 6 | 0.143 | 0.120 | Battery, Theft, Criminal damage | ✔ |
| 3 | 40,026 | 16.0 | 65.7 | 609.0 | 6 | 0.126 | 0.110 | Theft, Battery, Assault | ✔ |
| 0 | 41,057 | 11.0 | 79.6 | 515.5 | 11 | 0.107 | 0.117 | Theft, Battery, Deceptive practice | |
| 2 | 31,956 | 8.9 | 62.5 | 511.4 | 6 | 0.136 | 0.086 | Theft, Battery, Criminal damage | |
| 1 | 35,009 | 1.5 | 79.5 | 440.4 | 11 | 0.098 | 0.100 | Theft, Battery, Criminal damage | |

**3 of 6 high-activity zones cut across district boundaries.** With the time dimensions weighted equally to space, K = 6 separates mostly by **time of day** (peaks at 01:30, 09:00, 11:00, 16:00, 18:30, 23:00), so every cluster spans many districts.

### 3.3 Sensitivity to K & seed stability

| K | Silhouette | Weighted district purity | Median zone area km² | Cross-boundary clusters | HAZ not aligned |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.359 | 0.122 | 82.2 | 3 | 2 |
| 6 | **0.413** | 0.118 | 69.3 | 6 | 3 |
| 12 | 0.394 | 0.172 | 48.2 | 12 | 6 |
| 15 | 0.380 | 0.204 | 39.3 | 14 | 8 |

Higher K gives smaller, slightly more district-pure zones (more useful for patrol planning) at a small cost in silhouette. WSSSE varies by only **0.59 %** across five seeds, so the solution is stable.

---

## Task 4 — Distributed Predictive Modelling

**Data:** 2019–2023 (last 5 years), 30 % sample, rows with missing location excluded; weather and socio-economic features joined via broadcast.
**Temporal split:** train 2019–2022 (**276,106** rows), test 2023 (**22,176** rows). Arrest base rate **0.157**.

**Features:** `primary_type`, `loc_grp` (top 40 locations + OTHER), district, community area (all `StringIndexer`, `handleInvalid="keep"`), plus hour sin/cos, day of week, month, weekend, domestic, lat/lon, tmax, precipitation, snow, hardship index, % below poverty.

**Class weights:** balanced, `w_c = N / (n_classes · n_c)`.

### 4A. Arrest prediction (binary)

| Model | Train (s) |
|---|---:|
| RF (unweighted), 60 trees, depth 12 | 363.7 |
| RF (class-weighted) | 352.1 |
| GBT (class-weighted), 40 iters, depth 6 | 186.3 |

Evaluation at the default threshold (0.5) and at the F1-optimal threshold; ROC/PR curves built from a distributed 200-bin score histogram.

| Model | Threshold | AUC-ROC | AUC-PR | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF (unweighted) | 0.500 | 0.8725 | 0.6307 | 0.914 | 0.834 | 0.356 | 0.499 |
| RF (unweighted) | 0.280 | 0.8725 | 0.6307 | 0.912 | 0.686 | 0.500 | 0.578 |
| RF (class-weighted) | 0.500 | 0.8715 | 0.6171 | 0.831 | 0.387 | 0.693 | 0.497 |
| RF (class-weighted) | 0.625 | 0.8715 | 0.6171 | 0.910 | 0.672 | 0.489 | 0.566 |
| GBT (class-weighted) | 0.500 | **0.8805** | **0.6470** | 0.796 | 0.346 | 0.782 | 0.480 |
| GBT (class-weighted) | 0.760 | **0.8805** | **0.6470** | 0.912 | 0.670 | 0.520 | **0.586** |

**Best model: GBT (class-weighted)** — highest AUC-ROC and AUC-PR; F1 0.586 at threshold 0.76 (TP 1,386, FP 682, FN 1,280, TN 18,828). Class weighting mainly shifts the operating point; tuning the threshold recovers almost the same F1 as the unweighted model.

#### How crime-type skew drives precision/recall
Recall per crime type follows the type's own arrest base rate almost exactly:

| Type | Share of test | Arrest base rate | Recall | Precision |
|---|---:|---:|---:|---:|
| Theft | 20.4 % | 0.045 | 0.365 | 0.366 |
| Battery | 16.2 % | 0.151 | 0.011 | 0.500 |
| Motor vehicle theft | 12.2 % | 0.022 | 0.000 | 0.000 |
| Criminal damage | 11.5 % | 0.031 | 0.000 | 0.000 |
| Assault | 8.7 % | 0.105 | 0.005 | 0.250 |
| Other offense | 6.4 % | 0.143 | 0.676 | 0.493 |
| Weapons violation | 3.5 % | 0.599 | 0.920 | 0.676 |
| Narcotics | 2.3 % | 0.967 | **1.000** | **0.967** |
| Criminal trespass | 2.0 % | 0.301 | 0.788 | 0.435 |

The model effectively learns "which crime types lead to arrests" (narcotics, weapons, trespass) — `primary_type` dominates feature importance (≈0.60, then location group ≈0.14) — and barely identifies arrests within low-arrest, high-volume types.

### 4B. Primary-type prediction (multiclass, 11 classes)
Top 10 types + OTHER; `primary_type` itself is excluded from the features. Class weights balanced and capped at 10×.

| Model | Time (s) | Accuracy | Macro-F1 | Weighted-F1 | Macro OvR AUC |
|---|---:|---:|---:|---:|---:|
| RF multiclass (unweighted) | 753 | **0.320** | 0.173 | 0.240 | 0.759 |
| RF multiclass (class-weighted) | 816 | 0.299 | **0.239** | **0.276** | 0.760 |

Per class (recall):

| Class | Share | Recall (unweighted) | Recall (weighted) | OvR AUC (weighted) |
|---|---:|---:|---:|---:|
| Theft | 20.4 % | **0.813** | 0.259 | 0.709 |
| Battery | 16.2 % | 0.600 | 0.498 | 0.824 |
| Motor vehicle theft | 12.2 % | 0.024 | **0.505** | 0.840 |
| Criminal damage | 11.5 % | 0.108 | 0.010 | 0.652 |
| Assault | 8.7 % | 0.001 | 0.046 | 0.659 |
| OTHER | 7.9 % | 0.175 | 0.152 | 0.646 |
| Deceptive practice | 6.6 % | 0.362 | 0.589 | 0.860 |
| Other offense | 6.4 % | 0.001 | 0.062 | 0.704 |
| Robbery | 3.7 % | 0.004 | 0.255 | 0.762 |
| Weapons violation | 3.5 % | 0.096 | 0.488 | 0.825 |
| Burglary | 2.9 % | 0.019 | **0.594** | 0.873 |

Without weighting, the majority classes (theft, battery) absorb most of the recall; class weighting redistributes it to minority classes (burglary, MVT, weapons), raising macro-F1 by ~38 % at a small cost in accuracy.


---

## Key findings

1. **Clean source, meaningful repairs.** No cast failures, duplicates or quarantined rows, but 7.9 % of rows needed community-area imputation, 4.3 % district repair and 1.1 % coordinate repair.
2. **Parquet + partitioning** answered the benchmark query ≈3.4× faster than scanning CSV.
3. **Crime fell long-term**, with a post-2020 rebound in theft and motor-vehicle theft.
4. **Deprivation predicts the mix, not the volume:** poverty and unemployment correlate strongly with violent share (ρ 0.82–0.87) and arrest rate, weakly with total crime.
5. **Weather matters after deseasonalising:** freezing days −7 %, snow −6 %, heavy rain −3 %, hot days +1 %; all significant.
6. **Broadcast hash joins** are the right tool for small dimension tables; Python RDD pipelines are orders of magnitude slower.
7. **Hotspots don't respect district lines:** at K = 6, 3 of 6 high-activity zones span multiple districts; higher K gives patrol-scale zones.
8. **Arrest prediction** reaches AUC-ROC 0.88 / AUC-PR 0.65 (GBT), but mostly by recognising crime types that inherently lead to arrest.
9. **Class weighting** is essential for minority crime types in the multiclass model (macro-F1 0.17 → 0.24).

---

## Known issues & caveats

- **Storage benchmark (1.7):** the printed Parquet size (0.000 GB), file count (0) and empty `PartitionFilters: []` contradict the 531 files reported just after the silver write. The size / file-count glob and the plan inspection likely ran against a path or plan state that didn't reflect the written lake (e.g. the lake had been cleared/re-read). The timing and matching row counts are still valid, but the compression ratio and pruning evidence should be re-run before being quoted.
- **Rolling-rate anomalies (2.3):** the top z-scores are in 2001–2002 against near-zero baselines, which points to sparse community-area coding in the earliest years rather than real crime spikes. Consider starting the anomaly scan from 2003.
- **Midnight/noon spikes (2.1):** likely default timestamps for incidents with unknown times; they affect hour-based features.
- **Clustering is time-dominated:** with `TIME_WEIGHT = 1.0` the clusters split by hour first; lower the weight for purely spatial hotspots.
- **Modelling uses a 30 % sample** and 2 cores; results are indicative, and numbers will change with `CC_MODEL_FRAC`, `CC_RF_TREES`, `CC_GBT_ITERS`.
- **Leakage check:** `arrest` is recorded after the incident; features used are all available at report time, but `domestic` and `location_description` may be updated later in the record's life.

---

## Outputs

On completion the notebook zips everything to `report_assets.zip` (auto-downloaded on Colab): **12 figures** and **22 CSV tables**.

| Task | Figures | Tables |
|---|---|---|
| 1 | `t1_data_quality` | `t1_cast_failures`, `t1_dq_report`, `t1_quarantine_reasons`, `t1_location_quality`, `t1_storage_benchmark` |
| 2 | `t2_temporal`, `t2_spatial_socio`, `t2_rolling_rates`, `t2_weather`, `t2_broadcast` | `t2_yearly_by_type`, `t2_area_stats`, `t2_socio_spearman`, `t2_rolling_rates`, `t2_rate_anomalies`, `t2_daily_weather`, `t2_weather_event_effects`, `t2_weather_by_type`, `t2_broadcast_benchmark` |
| 3 | `t3_k_selection`, `t3_cluster_maps`, `t3_k_sensitivity` | `t3_k_selection`, `t3_cluster_profiles`, `t3_k_sensitivity`, `t3_seed_stability` |
| 4 | `t4_arrest_roc_pr_cm`, `t4_arrest_skew_importance`, `t4_type_skew` | `t4_arrest_metrics`, `t4_arrest_by_type`, `t4_arrest_feature_importance`, `t4_type_per_class` |

### How to run
1. Open `7002.ipynb` in Google Colab (or locally with Java 17 and PySpark 3.5).
2. Run all cells. The first cell installs PySpark, kagglehub and Java on Colab; the crime CSV downloads from Kaggle (Kaggle credentials may be needed) and weather/socio data download on first run.
3. If memory or time is tight, lower `CC_MODEL_FRAC`, `CC_CLUSTER_FRAC`, `CC_RF_TREES` or `CC_GBT_ITERS`.
