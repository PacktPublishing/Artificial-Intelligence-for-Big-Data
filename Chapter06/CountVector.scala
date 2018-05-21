import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,Tokenizer}


val exampleData = spark.createDataFrame(Seq(
  (0.0, "extraction of the feature using countvectorizer extraction method")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val tokensData = tokenizer.transform(exampleData)

val cvModel: CountVectorizerModel = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("features")
  .setVocabSize(3)
  .setMinDF(1)
  .fit(tokensData)

cvModel.transform(tokensData).select("words","features").show(false)