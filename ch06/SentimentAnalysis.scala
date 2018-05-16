import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.feature.{StringIndexer, StopWordsRemover, HashingTF, Tokenizer, IDF, NGram}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

//Sample Data
val exampleDF = spark.createDataFrame(Seq(
(1,"Samsung 80 cm 32 inches FH4003 HD Ready LED TV"),
(2,"Polaroid LEDP040A Full HD 99 cm LED TV Black"),
(3,"Samsung UA24K4100ARLXL 59 cm 24 inches HD Ready LED TV Black")
)).toDF("id","description")

exampleDF.show(false)

//Add labels to dataset
val indexer = new StringIndexer()
  .setInputCol("description")
  .setOutputCol("label")

val tokenizer = new Tokenizer()
  .setInputCol("description")
  .setOutputCol("words")

val remover = new StopWordsRemover()
  .setCaseSensitive(false)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("filtered")

val bigram = new NGram().setN(2).setInputCol(remover.getOutputCol).setOutputCol("ngrams")


val hashingTF = new HashingTF()
  .setNumFeatures(1000)
  .setInputCol(bigram.getOutputCol)
  .setOutputCol("features")

val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("IDF")

val nb = new NaiveBayes().setModelType("multinomial")
val pipeline = new Pipeline().setStages(Array(indexer,tokenizer,remover,bigram, hashingTF,idf,nb))
val nbmodel = pipeline.fit(exampleDF)
nbmodel.write.overwrite().save("/tmp/spark-logistic-regression-model")

val evaluationDF = spark.createDataFrame(Seq(
(1,"Samsung 80 cm 32 inches FH4003 HD Ready LED TV")
)).toDF("id","description")

val results = nbmodel.transform(evaluationDF)
results.show(false)