"""Rebuild the local gold tables from the existing silver Delta table (after a feature change)."""
import shutil

from deltalake import DeltaTable, write_deltalake

from lakehouse.transforms import gold_fleet_daily, silver_to_gold_features
from run_local import LAKE, TMP, read_delta, spark_session, stream, to_arrow

spark = spark_session()
silver = read_delta(spark, LAKE / "silver_smart")
write_deltalake(str(LAKE / "gold_drive_features"), stream(to_arrow(silver_to_gold_features(silver), "gold_features")),
                mode="overwrite", schema_mode="overwrite")
write_deltalake(str(LAKE / "gold_fleet_daily"), stream(to_arrow(gold_fleet_daily(silver), "gold_fleet")),
                mode="overwrite", schema_mode="overwrite")
g = DeltaTable(str(LAKE / "gold_drive_features")).to_pyarrow_dataset()
print("gold rows", g.count_rows(), "columns", len(g.schema.names))
spark.stop()
shutil.rmtree(TMP, ignore_errors=True)
