# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Gold: model features and fleet health
# MAGIC `gold_drive_features`: one row per drive-day with 7-day SMART deltas and the label
# MAGIC "fails within 7 days". `gold_fleet_daily`: active drives, failures and failure rate per day.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from lakehouse.transforms import gold_fleet_daily, silver_to_gold_features

silver = spark.table(SILVER)
silver_to_gold_features(silver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_FEATURES)
gold_fleet_daily(silver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_FLEET)

# COMMAND ----------

display(spark.sql(f"""SELECT count(*) AS feature_rows, sum(label_fail_7d) AS positive_rows
                      FROM {GOLD_FEATURES}"""))
display(spark.table(GOLD_FLEET).orderBy("date"))
