import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition

val data = Array(Vectors.dense(2.0, 1.0, 75.0, 18.0, 1.0,2),
Vectors.dense(0.0, 1.0, 21.0, 28.0, 2.0,4),
Vectors.dense(0.0, 1.0, 32.0, 61.0, 5.0,10),
Vectors.dense(0.0, 1.0, 56.0, 39.0, 2.0,4),
Vectors.dense(1.0, 1.0, 73.0, 81.0, 3.0,6),
Vectors.dense(0.0, 1.0, 97.0, 59.0, 7.0,14))

val rows = sc.parallelize(data)

val mat: RowMatrix = new RowMatrix(rows)

val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(3, computeU = true)

val U: RowMatrix = svd.U // The U factor is stored as a row matrix
val s: Vector = svd.s	 // The sigma factor is stored as a singular vector
val V: Matrix = svd.V	 // The V factor is stored as a local dense matrix



