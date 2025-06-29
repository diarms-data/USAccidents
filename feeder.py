import os
import time
import gc
import logging
from datetime import datetime
 
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark import StorageLevel
 
 
# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("feeder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
 
# Nettoyage des noms de colonnes
def clean_column_names(columns):
    return [
        col.replace("(", "_").replace(")", "_")
           .replace("%", "pct").replace(".", "_")
           .replace(" ", "_").replace("[", "").replace("]", "")
           .replace(";", "").replace(":", "").replace("{", "").replace("}", "")
        for col in columns
    ]
 
 
def feeder():
    global_start = time.time()
    logger.info("Initialisation de la session Spark...")
 
    spark = SparkSession.builder \
        .appName("CSV to Parquet Feeder Optimized with Logging") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
 
    base_path = "Bronze"
    input_dir = "source"
    cumulative_df = None
 
    logger.info("Lecture des fichiers CSV depuis : %s", input_dir)
 
    csv_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".csv") and len(f) >= 14],
        key=lambda x: datetime.strptime(x.replace(".csv", ""), "%d-%m-%Y")
    )
 
    logger.info("%d fichiers à traiter.", len(csv_files))
 
    for i, filename in enumerate(csv_files):
        file_start = time.time()
 
        file_path = os.path.join(input_dir, filename)
        date_str = filename.replace(".csv", "")
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
        output_path = os.path.join(base_path, year, month, day)
 
        is_last_file = i == len(csv_files) - 1
        data_already_exists = os.path.exists(os.path.join(output_path, "data_complete" if is_last_file else ""))
 
        logger.info("Traitement de : %s (%s)", filename, dt.strftime("%Y-%m-%d"))
 
        df = spark.read.option("header", "true").option("escape", "\"").csv(file_path)
        new_columns = clean_column_names(df.columns)
        df = df.toDF(*new_columns)
        df = df.withColumn("situation_date", lit(dt.strftime("%Y-%m-%d")).cast("date"))
        df.persist(StorageLevel.MEMORY_AND_DISK)
 
        if data_already_exists:
            logger.info("Données déjà présentes pour %s, lecture uniquement.", date_str)
        else:
            if is_last_file:
                final_path = os.path.join(output_path)
                logger.info("Écriture du fichier cumulé dans : %s", final_path)
                cumulative_df = df if cumulative_df is None else cumulative_df.unionByName(df)
                #cumulative_df = cumulative_df.coalesce(1)
                cumulative_df.write.mode("overwrite").option("mergeSchema", "true").parquet(final_path)
            else:
                df.write.mode("overwrite").parquet(output_path)
                logger.info("Données écrites dans : %s", output_path)
 
        cumulative_df = df if cumulative_df is None else cumulative_df.unionByName(df)
 
        df.unpersist()
        gc.collect()
 
        file_end = time.time()
        logger.info("Temps de traitement de %s : %.2f sec", filename, file_end - file_start)
 
    spark.stop()
    global_end = time.time()
    logger.info("Pipeline Spark terminé en %.2f secondes.", global_end - global_start)
 
if __name__ == "__main__":
    feeder()