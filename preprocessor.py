import os
import time
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark import StorageLevel
 
# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("preprocessor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocessor():
    start_time = time.time()
    logger.info("Initialisation de la session Spark...")
    
    # Initialisation Spark avec support Hive
    spark = SparkSession.builder \
        .appName("SilverLayerUSAccidents") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.warehouse.dir", "warehouse")\
        .enableHiveSupport() \
        .getOrCreate()
    # Lecture des données brutes
    file_path = "Bronze/*/*/*"
    logger.info("Chargement des données brutes depuis : %s", file_path)
    
    #df_raw = spark.read.option("header", True).option("inferSchema", True).parquet(file_path)
    #df_raw = spark.read.parquet(file_path)
    df_raw = spark.read.option("mergeSchema", True).parquet(file_path)
    #df_raw.persist(StorageLevel.MEMORY_AND_DISK)
 
    # Assure l'existence de la base Hive
    spark.sql("CREATE DATABASE IF NOT EXISTS silver")
    # ------------------------------
    # Dataset 1 : silver_accidents
    # ------------------------------
    logger.info("Traitement du dataset silver_accidents...")
    
    df_accidents = df_raw.select(
        "ID", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng", 
        "City", "State", "Country", "Timezone", "Distance_mi_"
    ).dropna(subset=["Start_Time", "End_Time", "Start_Lat", "Start_Lng"])
    df_accidents = df_accidents.withColumn("event_date", to_date("Start_Time")) \
        .withColumn("Severity", col("Severity").cast("int")) \
        .withColumn("Distance_mi_", col("Distance_mi_").cast("double"))
    df_accidents.write.mode("overwrite").format("parquet").saveAsTable("silver.silver_accidents")
    # ------------------------------
    # Dataset 2 : silver_weather.
    # ------------------------------
    logger.info("Traitement du dataset silver_weather...")
    
    df_weather = df_raw.select(
        "ID", "Start_Time", "Temperature_F_", "Humidity_pct_", 
        "Visibility_mi_", "Wind_Speed_mph_", "Weather_Condition"
    ).dropna(subset=["Start_Time", "Temperature_F_", "Humidity_pct_"])
    df_weather = df_weather.withColumn("Start_Time", to_date("Start_Time")) \
        .withColumn("Temperature_F_", col("Temperature_F_").cast("double")) \
        .withColumn("Humidity_pct_", col("Humidity_pct_").cast("double")) \
        .withColumn("Visibility_mi_", col("Visibility_mi_").cast("double")) \
        .withColumn("Wind_Speed_mph_", col("Wind_Speed_mph_").cast("double"))
    df_weather.write.mode("overwrite").format("parquet").saveAsTable("silver.silver_weather")
    #print("Silver datasets written to Hive successfully.")
    #spark.stop()

    logger.info("silver_weather écrit dans Hive (parquet).")
 
    #df_raw.unpersist()
    spark.stop()
 
    end_time = time.time()
    logger.info("Pipeline silver terminé en %.2f secondes.", end_time - start_time)
    
if __name__ == "__main__":
    preprocessor()