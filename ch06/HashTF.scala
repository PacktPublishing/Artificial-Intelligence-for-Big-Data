import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

val exampleData = spark.createDataFrame(Seq(
  (0.0, "extraction of the feature using HashingTF extraction method")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val tokensData = tokenizer.transform(exampleData)

val hashingTF = new HashingTF()
  .setInputCol("words").setOutputCol("TF").setNumFeatures(10)
val features = hashingTF.transform(tokensData)
val idf = new IDF().setInputCol("TF").setOutputCol("IDF")
val idfModel = idf.fit(features)
val rescaledData = idfModel.transform(features)
rescaledData.select("label", "TF","IDF").show(truncate=false)