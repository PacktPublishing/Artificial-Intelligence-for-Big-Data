import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans

val kmeansSampleData = sc.textFile("aibd/k-means-sample.txt")

val labeledData = kmeansSampleData.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).toDouble, parts(2).toDouble))
}.cache().toDF


val kmeans = new KMeans()
.setK(2) // default value is 2
.setFeaturesCol("features")
.setMaxIter(3) // default Max Iteration is 20
.setPredictionCol("prediction")
.setSeed(1L)

val model = kmeans.fit(labeledData)

summary.predictions.show
model.clusterCenters.foreach(println)

