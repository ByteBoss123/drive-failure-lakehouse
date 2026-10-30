# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Governance: Unity Catalog metadata, access control, lineage
# MAGIC * Table and column comments become searchable metadata in Catalog Explorer.
# MAGIC * Gold tables are readable by all workspace users; bronze, silver and the quarantine
# MAGIC   table stay with the owner (least privilege).
# MAGIC * Lineage from the Volume through bronze, silver and gold is captured by Unity Catalog
# MAGIC   automatically: open any gold table in Catalog Explorer and select the **Lineage** tab.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

comments = {
    BRONZE: "Raw daily Backblaze SMART rows, all strings, with source file and load time. Append-only.",
    SILVER: "Typed, validated, deduplicated drive-day SMART readings. Key: (serial_number, date).",
    QUARANTINE: "Rows rejected by silver data-quality rules, with dq_reason. Owner access only.",
    GOLD_FEATURES: "Drive-day model features and label_fail_7d (drive fails within 7 days).",
    GOLD_FLEET: "Daily fleet health: active drives, failures, daily failure rate, mean temperature.",
}
for table, text in comments.items():
    spark.sql(f"COMMENT ON TABLE {table} IS '{text}'")

column_comments = {
    "smart_5_raw": "Reallocated sectors count",
    "smart_9_raw": "Power-on hours",
    "smart_187_raw": "Reported uncorrectable errors",
    "smart_194_raw": "Drive temperature, degrees C",
    "smart_197_raw": "Current pending sector count",
    "smart_198_raw": "Offline uncorrectable sector count",
}
for table in (SILVER, GOLD_FEATURES):
    for col, text in column_comments.items():
        spark.sql(f"ALTER TABLE {table} ALTER COLUMN {col} COMMENT '{text}'")
spark.sql(f"ALTER TABLE {GOLD_FEATURES} ALTER COLUMN label_fail_7d COMMENT '1 if the drive fails within the next 7 days'")
spark.sql(f"ALTER TABLE {SILVER} SET TAGS ('layer' = 'silver', 'domain' = 'hardware_telemetry')")
spark.sql(f"ALTER TABLE {GOLD_FEATURES} SET TAGS ('layer' = 'gold', 'domain' = 'hardware_telemetry')")

# COMMAND ----------

for table in (GOLD_FEATURES, GOLD_FLEET):
    spark.sql(f"GRANT SELECT ON TABLE {table} TO `account users`")
spark.sql(f"GRANT USE SCHEMA ON SCHEMA {NS} TO `account users`")

# COMMAND ----------

for table in (GOLD_FLEET, QUARANTINE):
    print(table)
    display(spark.sql(f"SHOW GRANTS ON TABLE {table}"))
