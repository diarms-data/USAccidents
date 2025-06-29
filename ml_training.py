import logging
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
 
def ml_training():
    # Configuration du logger Python
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
 
    spark = SparkSession.builder \
        .appName("ML_Spark_AccidentSeverity") \
        .config("spark.sql.warehouse.dir", "warehouse") \
        .config("spark.driver.memory", "4g") \
        .enableHiveSupport() \
        .getOrCreate()
 
    # Réduire le niveau de log Spark (ERROR, WARN, INFO)
    spark.sparkContext.setLogLevel("WARN")
 
    logger.info("Chargement des données silver_accidents")
    df_acc = spark.sql("SELECT ID, Severity, City, State, Country, Timezone FROM silver.silver_accidents")
    logger.info(f"silver_accidents count: {df_acc.count()}")
 
    logger.info("Chargement des données silver_weather")
    df_met = spark.sql("""
        SELECT ID, Temperature_F_, Humidity_pct_, Visibility_mi_, Wind_Speed_mph_
        FROM silver.silver_weather
    """)
    logger.info(f"silver_weather count: {df_met.count()}")
 
    # On considère que silver_weather est plus petite => broadcast pour optimiser join
    logger.info("Jointure avec broadcast sur la table météo")
    start_join = time.time()
    df = df_acc.join(broadcast(df_met), on="ID", how="inner") \
        .dropna(subset=["Severity", "Temperature_F_", "Humidity_pct_", "Visibility_mi_", "Wind_Speed_mph_", "City", "State", "Country", "Timezone"])
    join_duration = time.time() - start_join
    logger.info(f"Durée de la jointure : {join_duration:.2f}s")
 
    # Cache du DataFrame final pour réutilisation efficace
    logger.info("Mise en cache du DataFrame joint")
    df.cache()
    df.show(10)  # Action pour matérialiser le cache
 
    # Cast target column
    df = df.withColumn("Severity", df["Severity"].cast("int"))
 
    # Encodage des colonnes catégorielles
    cat_features = ["City", "State", "Country", "Timezone"]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="skip") for col in cat_features]
 
    # Features numériques
    num_features = ["Temperature_F_", "Humidity_pct_", "Visibility_mi_", "Wind_Speed_mph_"]
    all_features = [col+"_idx" for col in cat_features] + num_features
 
    assembler = VectorAssembler(inputCols=all_features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
 
    rf = RandomForestClassifier(labelCol="Severity", featuresCol="features", numTrees=100)
 
    pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])
 
    # Split train/test
    logger.info("Split des données en train/test")
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Train size: {train_data.count()}, Test size: {test_data.count()}")
 
    # Entraînement
    logger.info("Début de l'entraînement du modèle")
    start_train = time.time()
    model = pipeline.fit(train_data)
    train_duration = time.time() - start_train
    logger.info(f"Durée d'entraînement : {train_duration:.2f}s")
 
    # Prédictions & évaluation
    logger.info("Prédiction sur le test set")
    predictions = model.transform(test_data)
 
    evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions)
    logger.info(f"F1-score du modèle : {f1:.4f}")
 
    logger.info("Affichage de quelques prédictions")
    predictions.select("City", "Temperature_F_", "Humidity_pct_", "Severity", "prediction").show(10, truncate=False)
 
    # Nettoyage
    logger.info("Libération du cache")
    df.unpersist()
 
    spark.stop()
    logger.info("Spark session stopped")
 
if __name__ == "__main__":
    ml_training()