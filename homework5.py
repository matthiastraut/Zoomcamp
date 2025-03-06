import datetime
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .master("local[*]") \
    .appName('test') \
    .getOrCreate()

df = spark.read \
    .option("header", "true") \
    .parquet('yellow_tripdata_2024-10.parquet')

df = df.repartition(4)

df.write.parquet('yellow_tripdata.parquet')

df.filter(df.tpep_pickup_datetime >= datetime.datetime(2024,10,15, 0, 0, 0)).filter(df.tpep_pickup_datetime <= datetime.datetime(2024,10,16, 0, 0, 0)).drop_duplicates().count()

df = df.withColumn('triplength', (F.col("tpep_dropoff_datetime") - F.col("tpep_pickup_datetime")))

df.sort('triplength', ascending=False).first()

x = df.groupBy('PULocationID').count().orderBy('count', ascending=True).head(5)

locations = spark.read \
    .option("header", "true") \
    .csv('taxi_zone_lookup.csv')

for i in range(5):
    print(locations.filter(locations.LocationID == x[i]['PULocationID']).first())
