from pyspark.sql import SparkSession
import feeder
import preprocessor
import datamarts
import ml_training
 
def main():
    spark = SparkSession.builder \
        .appName("Data Pipeline Bronze-Silver-Gold") \
        .enableHiveSupport() \
        .getOrCreate()
 
    print("🔶 STEP 1: Feeder - Source → Bronze")
    feeder.run(spark)
 
    print("🔷 STEP 2: Preprocessor - Bronze → Silver")
    preprocessor.run(spark)
 
    print("🟡 STEP 3: Datamart - Silver → Gold (analytique)")
    datamarts.run(spark)
 
    print("🧠 STEP 4: ML - Silver → Model Training")
    ml_training.run(spark)
 
    spark.stop()
 
if __name__ == "__main__":
    main()