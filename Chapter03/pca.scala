import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val data = Array(Vectors.dense(2.0, 1.0, 75.0, 18.0, 1.0,2),
Vectors.dense(0.0, 1.0, 21.0, 28.0, 2.0,4),
Vectors.dense(0.0, 1.0, 32.0, 61.0, 5.0,10),
Vectors.dense(0.0, 1.0, 56.0, 39.0, 2.0,4),
Vectors.dense(1.0, 1.0, 73.0, 81.0, 3.0,6),
Vectors.dense(0.0, 1.0, 97.0, 59.0, 7.0,14))

val rows = sc.parallelize(data)

val mat: RowMatrix = new RowMatrix(rows)

// Principal components are stored in a local dense matrix.
val pc: Matrix = mat.computePrincipalComponents(2)

// Project the rows to the linear space spanned by the top 2 principal components.
val projected: RowMatrix = mat.multiply(pc)

projected.rows.foreach(println)