import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, mean, round, when, count, expr

# Initialize Spark Session
spark = SparkSession.builder.appName("SASToPython").getOrCreate()

# Load parquet file
df = spark.read.parquet("./data.parquet")
df.show(5)
print('\n\n')

print('\n\nSort input data')
df_sorted = df.orderBy(["incur_yr", "product"])
df_sorted.show(5)
print('\n\n')


print('\n\nCreate column product_remove_year')
df = df.withColumn("product_remove_year", 
    when(col("product") == "ACC", 2026)
    .when(col("product") == "LTD", 2025)
    .when(col("product") == "VLF", 2030)
    .when(col("product") == "LIF", 2027)
)
df.show(5)
print('\n\n')

print('\n\nAdd column tot_type_product - counts by year and product')
windowSpec = Window.partitionBy("incur_yr", "product")
df = df.withColumn("tot_type_product", count("*").over(windowSpec))
df.show(5)
print('\n\n')

print('\n\nSummarize the dataset by volume and transpose it')
avg_vol_by_year_product = df.groupBy("feed", "product").agg(round(mean("volume"), 1).alias("avg_volume"))
transpose_vol_summary = avg_vol_by_year_product.groupBy("feed").pivot("product").sum("avg_volume")
transpose_vol_summary.show(5)
print('\n\n')


print('\n\nRun stratified analysis by computing statistics across product types and years where product are flagged for removal')
df.groupBy("product", "product_remove_year").count().show(5)
print('\n\n')


print('\n\nProvide summary and descriptive statistics for variable volume by incur_yr ')
summary_stats = df.groupBy("incur_yr").agg(
    mean("volume").alias("mean_volume"),
    expr("percentile_approx(volume, 0.5)").alias("median_volume"),
    count("*").alias("n")
)
summary_stats.show()


print('\n\nCreate data subset containing products VLF/LIF and calculate volume sums across pol and phname columns')
df_vlf = df.filter((col("incur_yr").isin([2013, 2014])) & (col("product") == "VLF")).select("pol", "phname", "volume")
df_lif = df.filter((col("incur_yr").isin([2013, 2014])) & (col("product") == "LIF")).select("pol", "phname", "volume")

products = df_vlf.union(df_lif).orderBy("pol")
products.show(5)


product_volumes = products.withColumn("volume_over_10000000", when(col("volume") > 10000000, "yes").otherwise("no"))
total_volumes = product_volumes.filter(col("volume_over_10000000") == "yes").groupBy("pol", "phname").sum("volume")
total_volumes.show()

print('\n\nProvide report for Volumes over 100M across products and years')
df.filter(col("volume") > 100000000).groupBy("product", "incur_yr").sum("volume").show()

print('\n\nFunction with input parameters year and product')
def show_result(year, product):
    df.filter((col("incur_yr") == year) & (col("product") == product)).show()

show_result(2014, "ACC")


print('\n\nMacro equivalent in Python')
def test(finish):
    i = 1
    while i < finish:
        print(f"The value of i is {i}")
        i += 1

test(5)


print('\n\nCreate a permanent dataset, add a new column product_remove_year and update it')
df_1 = df.drop("pol", "product_remove_year").filter(col("incur_yr") != 2013)
df_1 = df_1.withColumn("product_remove_year", 
    when(col("product") == "ACC", 2026).otherwise(None))
df_1.show()
