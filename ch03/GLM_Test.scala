import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GeneralizedLinearRegression

val mpg_data = sc.textFile( "aibd/mpg_data.txt" )

val labeledData = mpg_data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).toDouble, 
  parts(2).toDouble,parts(3).toDouble, parts(4).toDouble, 
  parts(5).toDouble, parts(6).toDouble, parts(7).toDouble))
}.cache().toDF

val glr = new GeneralizedLinearRegression()
.setMaxIter(1000)
.setRegParam(0.03) //the value ranges from 0.0 to 1.0. Experimentation required to identify the right value.
.setFamily("gaussian")
.setLink( "identity" )

val glrModel = glr.fit(labeledData)

val summary = glrModel.summary

summary.residuals().show()
println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom)
println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull)
println("AIC: " + summary.aic)
println("Dispersion: " + summary.dispersion)
println("Null Deviance: " + summary.nullDeviance)
println("Deviance: " +summary.deviance)
println("p-values: " + summary.pValues.mkString(","))
println("t-values: " + summary.tValues.mkString(","))
println("Coefficient Standard Error: " + summary.coefficientStandardErrors.mkString(","))