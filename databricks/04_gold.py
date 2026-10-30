# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Gold: model features and fleet health
# MAGIC `gold_drive_features`: one row per drive-day with 7-day SMART deltas and the label
# MAGIC "fails within 7 days". `gold_fleet_daily`: active drives, failures and failure rate per day.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

silver = spark.table(SILVER)
silver_to_gold_features(silver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_FEATURES)
gold_fleet_daily(silver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_FLEET)

# COMMAND ----------

r = spark.sql(f"SELECT count(*) AS n, sum(label_fail_7d) AS pos FROM {GOLD_FEATURES}").first()
f = spark.sql(f"SELECT count(*) AS days, sum(failures) AS failures FROM {GOLD_FLEET}").first()
dbutils.notebook.exit(f"gold_feature_rows={r.n} positive_rows={r.pos} fleet_days={f.days} fleet_failures={f.failures}")
