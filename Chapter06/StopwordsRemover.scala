import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

StopWordsRemover remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered");

List<Row> data = Arrays.asList(
  RowFactory.create(Arrays.asList("I", "am", "removing", "the", "stop", "words")),
  RowFactory.create(Arrays.asList("from", "a", "large", "volume", "of","data"))
);

StructType schema = new StructType(new StructField[]{
  new StructField(
    "raw", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
});

Dataset<Row> dataset = spark.createDataFrame(data, schema);
remover.transform(dataset).show(false);