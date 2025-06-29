import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, count, avg, sum, hour, year, month, dayofmonth, dayofweek, date_format,
    when, unix_timestamp
)
from pyspark import StorageLevel

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("datamart.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def datamart():
    start_time = time.time()
    logger.info("🚀 Initialisation de Spark pour la génération des datamarts...")

    spark = SparkSession.builder \
        .appName("GoldDatamartBuilder") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.warehouse.dir", "warehouse") \
        .config("spark.jars", "jars\postgresql-42.7.5.jar") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")\
        .enableHiveSupport() \
        .getOrCreate()

    postgres_url = "jdbc:postgresql://host.docker.internal:5432/postgres"
    connection_properties = {
        "user": "postgres",
        "password": "diarms",
        "driver": "org.postgresql.Driver"
    }

    try:
        df_accidents = spark.table("silver.silver_accidents")
        df_weather = spark.table("silver.silver_weather")
        logger.info("Chargement des données Silver depuis Hive terminé.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des tables Hive : {e}")
        spark.stop()
        return

    # ------------------------------
    # Datamart 1 : Gravité par ville et météo par jour
    # ------------------------------
    try:
        df_accidents_renamed = df_accidents.withColumnRenamed("Start_Time", "Start_Time_acc")
        df_weather_renamed = df_weather.withColumnRenamed("Start_Time", "Start_Time_weather")

        df_joined = df_accidents_renamed.join(df_weather_renamed, on="ID", how="inner") \
            .withColumn("event_date", to_date("Start_Time_acc"))

        dm_severity_by_city_day_weather = df_joined.groupBy("City", "event_date", "Weather_Condition") \
            .agg(
                avg("Severity").alias("avg_severity"),
                count("*").alias("nb_accidents"),
                avg("Temperature_F_").alias("avg_temp"),
                avg("Humidity_pct_").alias("avg_humidity")
            ).filter(col("avg_severity").isNotNull())

        dm_severity_by_city_day_weather.write.jdbc(postgres_url, "dmAcc_severity_by_city_day_weather", "overwrite", connection_properties)
        logger.info("dmAcc_severity_by_city_day_weather écrit dans PostgreSQL.")
    except Exception as e:
        logger.error(f"Erreur écriture dm_severity_by_city_day_weather : {e}")

    # ------------------------------
    # Datamart 2 : Accidents par heure et par état
    # ------------------------------
    try:
        df_hourly = df_accidents.withColumn("hour", hour("Start_Time"))
        dm_accidents_by_hour_and_state = df_hourly.groupBy("State", "hour") \
            .agg(count("*").alias("nb_accidents"))

        dm_accidents_by_hour_and_state.write.jdbc(postgres_url, "dmAcc_accidents_by_hour_and_state", "overwrite", connection_properties)
        logger.info("dmAcc_accidents_by_hour_and_state écrit dans PostgreSQL.")
    except Exception as e:
        logger.error(f"Erreur écriture dm_accidents_by_hour_and_state : {e}")

    # ------------------------------
    # Datamart 3 : Impact météo sur la gravité
    # ------------------------------
    try:
        df_weather_joined = df_accidents.join(df_weather, on="ID", how="inner")
        dm_weather_impact_on_severity = df_weather_joined.groupBy("Weather_Condition") \
            .agg(
                avg("Severity").alias("avg_severity"),
                count("*").alias("nb_accidents")
            ).filter(col("avg_severity").isNotNull())

        dm_weather_impact_on_severity.write.jdbc(postgres_url, "dmAcc_weather_impact_on_severity", "overwrite", connection_properties)
        logger.info("dmAcc_weather_impact_on_severity écrit dans PostgreSQL.")
    except Exception as e:
        logger.error(f"Erreur écriture dm_weather_impact_on_severity : {e}")

    # ------------------------------
    # Datamart 4 : Analyse temporelle détaillée
    # ------------------------------
    try:
        df_temporal = df_accidents.withColumn("event_date", to_date("Start_Time")) \
            .withColumn("year", year("event_date")) \
            .withColumn("month", month("event_date")) \
            .withColumn("day", dayofmonth("event_date")) \
            .withColumn("day_of_week", date_format("event_date", "EEEE")) \
            .withColumn("is_weekend", when(dayofweek("event_date").isin(1, 7), True).otherwise(False))\
            .withColumn("duration_min", 
                        (unix_timestamp("End_Time") - unix_timestamp("Start_Time")) / 60)

        dm_temporal_analysis_accidents = df_temporal.groupBy("event_date", "year", "month", "day", "day_of_week", "is_weekend") \
            .agg(
                count("*").alias("nb_accidents"),
                avg("Severity").alias("avg_severity"),
                sum("Distance_mi_").alias("total_distance"),
                avg("duration_min").alias("avg_duration_min")
            )

        dm_temporal_analysis_accidents.write.jdbc(postgres_url, "dmAcc_temporal_analysis_accidents", "overwrite", connection_properties)
        logger.info("dmAcc_temporal_analysis_accidents écrit dans PostgreSQL.")
    except Exception as e:
        logger.error(f"Erreur écriture dm_temporal_analysis_accidents : {e}")

    spark.stop()
    duration = time.time() - start_time
    logger.info("🏁 Processus terminé en %.2f secondes.", duration)

if __name__ == "__main__":
    datamart()
