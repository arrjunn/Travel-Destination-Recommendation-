# BDA Lab Activity 6 — Travel Destination Recommendation System
## B.Tech CSE - 6th Semester | Big Data Analytics

### Objective
Build a collaborative filtering-based **travel destination recommendation system** using Apache Spark's MLlib ALS algorithm.

### Project Structure
```
bda-lab6/
├── recommendation_system.py   # Main script (all 8 steps)
├── generate_data.py           # Data generator (60 users, 35 destinations)
├── data/
│   ├── ratings.csv            # 1001 user-destination ratings
│   └── destinations.csv       # Destination metadata (35 places)
├── output_log.txt             # Sample execution output
└── README.md
```

### How to Run
```bash
# 1. Install PySpark
pip install pyspark

# 2. Generate sample data
python generate_data.py

# 3. Run the recommendation system
python recommendation_system.py
```

### Algorithm: ALS (Alternating Least Squares)
ALS is a matrix factorization technique for collaborative filtering:

- Decomposes the sparse user-destination rating matrix **R** into two low-rank matrices:
  - **U** (user factors) — captures traveler preferences
  - **V** (item factors) — captures destination characteristics
- Missing ratings are predicted as: **R̂ = U × Vᵀ**
- ALS alternates between fixing U to solve for V and vice versa

### Dataset
- **60 travelers** rating **35 global destinations** (Bali, Santorini, Kyoto, Machu Picchu, etc.)
- **1001 total ratings** (1.0 to 5.0 scale)
- Users have hidden preference patterns (beach, adventure, culture, scenic, remote)
- Sparsity: ~52%

### Steps Implemented
1. **Spark Session Initialization**
2. **Data Loading & Exploratory Analysis** — schema, distribution, top destinations
3. **Train/Test Split** — 80/20 with seed
4. **ALS Model Training** — with configurable hyperparameters
5. **Model Evaluation** — RMSE metric
6. **Hyperparameter Grid Search** — 6 configurations compared
7. **Top-N Recommendations** — for all users
8. **Personalized Recommendations** — per-user with travel history

### Key Results
| Metric | Value |
|--------|-------|
| Best RMSE | ~1.10 |
| Best Config | rank=5, regParam=0.1, maxIter=10 |
| Output | Top-5 destinations per traveler |
