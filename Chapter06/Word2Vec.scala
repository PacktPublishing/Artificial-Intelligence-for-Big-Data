import org.apache.spark.ml.feature.{Word2Vec,Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

val exampleData = spark.createDataFrame(Seq(
  (0.0, "extraction of the feature using word2Vec extraction method")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val tokensData = tokenizer.transform(exampleData)

val word2Vec = new Word2Vec()
  .setInputCol("words")
  .setOutputCol("features")
  .setVectorSize(3)
  .setMinCount(0)
val model = word2Vec.fit(tokensData)
val result = model.transform(tokensData)
result.select("words","features").show(false)