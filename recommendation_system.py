"""
===================================================================
 BDA Lab Activity 6
 Build a Recommendation System using Apache Spark and MLlib
 Domain: Travel Destination Recommendations
 B.Tech CSE - 6th Semester | Big Data Analytics
===================================================================

 Objective:
   Implement a collaborative filtering-based travel destination
   recommendation system using the ALS (Alternating Least Squares)
   algorithm from Apache Spark's MLlib library.

 Algorithm:
   ALS (Alternating Least Squares) - a matrix factorization technique
   that decomposes the user-item rating matrix into two lower-rank
   matrices (user factors and item factors) to predict missing ratings.

   Given a sparse ratings matrix R (users x destinations):
       R ≈ U × V^T
   where U = user factor matrix, V = item factor matrix.

 Steps:
   1. Initialize Spark Session
   2. Load and explore the ratings dataset
   3. Split data into training and test sets
   4. Train ALS model
   5. Evaluate using RMSE (Root Mean Square Error)
   6. Hyperparameter tuning (grid search)
   7. Generate top-N destination recommendations for users
   8. Personalized recommendations for specific users

 Dependencies: pyspark (with MLlib)
===================================================================
"""

# ======================== IMPORTS ========================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, count, avg, desc, round as spark_round
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

# ======================== CONFIGURATION ========================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
DESTINATIONS_FILE = os.path.join(DATA_DIR, "destinations.csv")

SEED = 42
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


# ============================================================
#  STEP 1: Initialize Spark Session
# ============================================================
def create_spark_session():
    """
    SparkSession is the unified entry point to Spark SQL,
    DataFrame API, and MLlib. Configured for local execution.
    """
    print("=" * 65)
    print(" BDA Lab 6: Travel Destination Recommendation System")
    print(" Using Apache Spark MLlib (ALS Algorithm)")
    print("=" * 65)

    spark = SparkSession.builder \
        .appName("BDA_Lab6_TravelRecommendationSystem") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"\n[+] Spark Session initialized (version: {spark.version})")
    return spark


# ============================================================
#  STEP 2: Load and Explore Data
# ============================================================
def load_and_explore_data(spark):
    """
    Load the ratings and destination datasets into Spark DataFrames.
    Perform exploratory data analysis (EDA) to understand the data.
    """
    print("\n" + "=" * 65)
    print(" STEP 2: Loading and Exploring Data")
    print("=" * 65)

    # Load CSVs with automatic schema inference
    ratings_df = spark.read.csv(RATINGS_FILE, header=True, inferSchema=True)
    dest_df = spark.read.csv(DESTINATIONS_FILE, header=True, inferSchema=True)

    # --- Basic Statistics ---
    total_ratings = ratings_df.count()
    total_users = ratings_df.select("userId").distinct().count()
    total_dests = ratings_df.select("destId").distinct().count()
    sparsity = (1 - total_ratings / (total_users * total_dests)) * 100

    print(f"\n--- Dataset Overview ---")
    print(f"  Total ratings       : {total_ratings}")
    print(f"  Unique travelers    : {total_users}")
    print(f"  Unique destinations : {total_dests}")
    print(f"  Matrix sparsity     : {sparsity:.1f}%")

    print(f"\n--- Schema ---")
    ratings_df.printSchema()

    print("--- Sample Ratings ---")
    ratings_df.show(10, truncate=False)

    # --- Rating Distribution ---
    print("--- Rating Distribution (top 10 values) ---")
    ratings_df.groupBy("rating") \
        .agg(count("*").alias("count")) \
        .orderBy("rating") \
        .show(10)

    # --- Most Popular Destinations ---
    print("--- Top 10 Most-Rated Destinations ---")
    ratings_df.groupBy("destId") \
        .agg(count("*").alias("num_ratings")) \
        .join(dest_df, "destId") \
        .orderBy(desc("num_ratings")) \
        .select("destId", "name", "num_ratings") \
        .show(10, truncate=False)

    # --- Highest Rated Destinations (min 5 ratings) ---
    print("--- Top 10 Highest-Rated Destinations (min 5 ratings) ---")
    ratings_df.groupBy("destId") \
        .agg(
            spark_round(avg("rating"), 2).alias("avg_rating"),
            count("*").alias("num_ratings")
        ) \
        .filter(col("num_ratings") >= 5) \
        .join(dest_df, "destId") \
        .orderBy(desc("avg_rating")) \
        .select("destId", "name", "avg_rating", "num_ratings") \
        .show(10, truncate=False)

    return ratings_df, dest_df


# ============================================================
#  STEP 3: Train/Test Split
# ============================================================
def split_data(ratings_df):
    """
    Randomly split data into training (80%) and test (20%) sets.
    Seed ensures reproducibility across runs.
    """
    print("\n" + "=" * 65)
    print(" STEP 3: Splitting Data into Train and Test Sets")
    print("=" * 65)

    (training_df, test_df) = ratings_df.randomSplit(
        [TRAIN_RATIO, TEST_RATIO], seed=SEED
    )

    train_count = training_df.count()
    test_count = test_df.count()

    print(f"  Training set : {train_count} ratings ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Test set     : {test_count} ratings ({TEST_RATIO*100:.0f}%)")
    print(f"  Total        : {train_count + test_count}")

    return training_df, test_df


# ============================================================
#  STEP 4: Train ALS Model
# ============================================================
def train_als_model(training_df):
    """
    Train the ALS (Alternating Least Squares) model.

    Key ALS Parameters:
    - rank       : Number of latent factors (dimensionality of U and V).
                   Higher = more expressive but risk of overfitting.
    - maxIter    : Maximum number of optimization iterations.
    - regParam   : Regularization parameter (lambda). Prevents overfitting
                   by penalizing large factor values.
    - coldStartStrategy : 'drop' ignores users/items absent from training.
    """
    print("\n" + "=" * 65)
    print(" STEP 4: Training ALS Model")
    print("=" * 65)

    als = ALS(
        maxIter=10,
        regParam=0.1,
        rank=10,
        userCol="userId",
        itemCol="destId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=SEED
    )

    print("\n  ALS Configuration:")
    print(f"    rank (latent factors) = 10")
    print(f"    maxIter               = 10")
    print(f"    regParam (lambda)     = 0.1")
    print(f"    coldStartStrategy     = drop")

    print("\n  [..] Training ALS model...")
    model = als.fit(training_df)
    print("  [+] Model trained successfully!")

    return model


# ============================================================
#  STEP 5: Evaluate Model
# ============================================================
def evaluate_model(model, test_df):
    """
    Evaluate model accuracy using RMSE (Root Mean Square Error).

    RMSE = sqrt( (1/n) * Σ(actual - predicted)² )

    Lower RMSE indicates better prediction accuracy.
    """
    print("\n" + "=" * 65)
    print(" STEP 5: Model Evaluation (RMSE)")
    print("=" * 65)

    # Generate predictions on test set
    predictions = model.transform(test_df)

    print("\n--- Predicted vs Actual Ratings (sample) ---")
    predictions.select("userId", "destId", "rating", "prediction") \
        .orderBy("userId", "destId") \
        .show(15, truncate=False)

    # Calculate RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)

    print(f"  +-----------------------------------------+")
    print(f"  |  RMSE = {rmse:.4f}                          |")
    print(f"  +-----------------------------------------+")

    if rmse < 0.8:
        print("    -> Excellent (RMSE < 0.8)")
    elif rmse < 1.0:
        print("    -> Good (RMSE < 1.0)")
    elif rmse < 1.3:
        print("    -> Acceptable (RMSE < 1.3)")
    else:
        print("    -> Needs improvement (RMSE >= 1.3)")

    return predictions, rmse


# ============================================================
#  STEP 6: Hyperparameter Tuning (Grid Search)
# ============================================================
def hyperparameter_tuning(training_df, test_df):
    """
    Compare multiple ALS configurations to find optimal
    hyperparameters. Tests combinations of rank and regParam.
    """
    print("\n" + "=" * 65)
    print(" STEP 6: Hyperparameter Tuning (Grid Search)")
    print("=" * 65)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    # Grid of hyperparameter combinations
    param_grid = [
        {"rank": 5,  "regParam": 0.01, "maxIter": 10},
        {"rank": 5,  "regParam": 0.1,  "maxIter": 10},
        {"rank": 10, "regParam": 0.01, "maxIter": 10},
        {"rank": 10, "regParam": 0.1,  "maxIter": 10},
        {"rank": 15, "regParam": 0.05, "maxIter": 10},
        {"rank": 15, "regParam": 0.1,  "maxIter": 10},
    ]

    best_rmse = float("inf")
    best_model = None
    best_params = None

    print(f"\n  Testing {len(param_grid)} configurations...\n")
    print(f"  {'Rank':<8}{'RegParam':<12}{'MaxIter':<10}{'RMSE':<12}{'Status'}")
    print(f"  {'-'*52}")

    for params in param_grid:
        als = ALS(
            rank=params["rank"],
            regParam=params["regParam"],
            maxIter=params["maxIter"],
            userCol="userId",
            itemCol="destId",
            ratingCol="rating",
            coldStartStrategy="drop",
            seed=SEED
        )
        model = als.fit(training_df)
        preds = model.transform(test_df)
        rmse = evaluator.evaluate(preds)

        is_best = rmse < best_rmse
        status = "<-- BEST" if is_best else ""
        print(f"  {params['rank']:<8}{params['regParam']:<12}"
              f"{params['maxIter']:<10}{rmse:<12.4f}{status}")

        if is_best:
            best_rmse = rmse
            best_model = model
            best_params = params

    print(f"\n  [+] Best Configuration:")
    print(f"      rank={best_params['rank']}, "
          f"regParam={best_params['regParam']}, "
          f"maxIter={best_params['maxIter']}")
    print(f"      RMSE = {best_rmse:.4f}")

    return best_model, best_rmse, best_params


# ============================================================
#  STEP 7: Generate Top-N Recommendations (All Users)
# ============================================================
def generate_recommendations(model, dest_df, num_recs=5):
    """
    Generate top-N travel destination recommendations for all users.
    Uses the trained model's factor matrices to predict ratings for
    unseen destinations.
    """
    print("\n" + "=" * 65)
    print(f" STEP 7: Top-{num_recs} Recommendations for All Travelers")
    print("=" * 65)

    # Generate recommendations
    user_recs = model.recommendForAllUsers(num_recs)

    print("\n--- Raw Recommendation Structure ---")
    user_recs.show(3, truncate=False)

    # Flatten: explode array of (destId, rating) structs
    recs_flat = user_recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("userId"),
        col("rec.destId").alias("destId"),
        spark_round(col("rec.rating"), 2).alias("predicted_rating")
    )

    # Join with destination names
    recs_named = recs_flat.join(dest_df, "destId") \
        .select("userId", "destId", "name", "predicted_rating") \
        .orderBy("userId", desc("predicted_rating"))

    print("\n--- Recommendations with Destination Names ---")
    recs_named.show(30, truncate=False)

    return recs_named


# ============================================================
#  STEP 8: Personalized Recommendations for a Specific User
# ============================================================
def recommend_for_user(model, dest_df, ratings_df, user_id, num_recs=5):
    """
    Display a specific traveler's rating history alongside
    new personalized destination suggestions.
    """
    print("\n" + "=" * 65)
    print(f" STEP 8: Personalized Plan for Traveler #{user_id}")
    print("=" * 65)

    # Show destinations already rated/visited by this user
    user_history = ratings_df.filter(col("userId") == user_id) \
        .join(dest_df, "destId") \
        .select("destId", "name", "rating") \
        .orderBy(desc("rating"))

    print(f"\n  --- Places Already Rated by Traveler #{user_id} ---")
    user_history.show(truncate=False)

    # Generate new recommendations
    user_subset = ratings_df.select("userId").distinct() \
        .filter(col("userId") == user_id)
    recs = model.recommendForUserSubset(user_subset, num_recs)

    recs_flat = recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("rec.destId").alias("destId"),
        spark_round(col("rec.rating"), 2).alias("predicted_rating")
    )

    recs_named = recs_flat.join(dest_df, "destId") \
        .select("name", "predicted_rating") \
        .orderBy(desc("predicted_rating"))

    print(f"  --- Top-{num_recs} NEW Destinations for Traveler #{user_id} ---")
    recs_named.show(truncate=False)


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main():
    """Execute the full recommendation system pipeline."""

    # Step 1: Initialize Spark
    spark = create_spark_session()

    try:
        # Step 2: Load and explore data
        ratings_df, dest_df = load_and_explore_data(spark)

        # Step 3: Train/Test split
        training_df, test_df = split_data(ratings_df)

        # Step 4: Train default ALS model
        default_model = train_als_model(training_df)

        # Step 5: Evaluate default model
        _, default_rmse = evaluate_model(default_model, test_df)

        # Step 6: Hyperparameter tuning
        best_model, best_rmse, best_params = hyperparameter_tuning(
            training_df, test_df
        )

        # Select the best performing model
        if best_rmse < default_rmse:
            final_model = best_model
            final_rmse = best_rmse
            print(f"\n  [+] Tuned model selected (RMSE: {final_rmse:.4f})")
        else:
            final_model = default_model
            final_rmse = default_rmse
            print(f"\n  [+] Default model selected (RMSE: {final_rmse:.4f})")

        # Step 7: Generate recommendations for all users
        all_recs = generate_recommendations(final_model, dest_df, num_recs=5)

        # Step 8: Personalized recommendations
        for uid in [1, 10]:
            recommend_for_user(final_model, dest_df, ratings_df, uid, num_recs=5)

        # =================== SUMMARY ===================
        print("\n" + "=" * 65)
        print(" FINAL SUMMARY")
        print("=" * 65)
        print(f"  Domain          : Travel Destination Recommendations")
        print(f"  Dataset         : {ratings_df.count()} ratings | "
              f"{ratings_df.select('userId').distinct().count()} travelers | "
              f"{ratings_df.select('destId').distinct().count()} destinations")
        print(f"  Algorithm       : ALS (Alternating Least Squares)")
        print(f"  Best Parameters : rank={best_params['rank']}, "
              f"regParam={best_params['regParam']}")
        print(f"  Best RMSE       : {final_rmse:.4f}")
        print(f"  Output          : Top-5 destinations per traveler")
        print("=" * 65)
        print(" Lab Activity 6 - COMPLETE")
        print("=" * 65)

    finally:
        spark.stop()
        print("\n[+] Spark session stopped.")


if __name__ == "__main__":
    main()
