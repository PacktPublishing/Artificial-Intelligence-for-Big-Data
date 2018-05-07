import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression

val linearRegrsssionSampleData = sc.textFile("aibd/linear_regression_sample.txt")

val labeledData = linearRegrsssionSampleData.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).toDouble))
}.cache().toDF

val lr = new LinearRegression()

val model = lr.fit(labeledData)

val summary = model.summary
println("R-squared = "+ summary.r2)





