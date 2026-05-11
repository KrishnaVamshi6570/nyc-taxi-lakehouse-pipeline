from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("check").getOrCreate()
model = PipelineModel.load("gs://gcp-pyspark-lakehouse-cdebatch52-486907/models/tip_prediction_gbt")
# Get the VectorAssembler input columns
assembler = model.stages[0]
print("Model expects these features:")
for f in assembler.getInputCols():
    print(f" - {f}")
spark.stop()
