import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.util.Random
import org.apache.spark.mllib.clustering._
import org.apache.spark.ml.clustering._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.FuzzyCMeans
import org.apache.spark.mllib.clustering.FuzzyCMeans._
import org.apache.spark.mllib.clustering.FuzzyCMeansModel


val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),
      Vectors.dense(9.0, 0.0),
      Vectors.dense(9.0, 0.2),
      Vectors.dense(9.2, 0.0)
    )
    val rdd = sc.parallelize(points, 3).cache()

    for (initMode <- Seq(KMeans.RANDOM, KMeans.K_MEANS_PARALLEL)) {

      (1 to 10).map(_ * 2) foreach { fuzzifier =>

        val model = org.apache.spark.mllib.clustering.FuzzyCMeans.train(rdd, k = 2, maxIterations = 10, runs = 10, initMode,
          seed = 26031979L, m = fuzzifier)

        val fuzzyPredicts = model.fuzzyPredict(rdd).collect()
        
        rdd.collect() zip fuzzyPredicts foreach { fuzzyPredict =>
          println(s" Point ${fuzzyPredict._1}")
          fuzzyPredict._2 foreach{clusterAndProbability =>
            println(s"Probability to belong to cluster ${clusterAndProbability._1} " +
              s"is ${"%.2f".format(clusterAndProbability._2)}")
          }
        }
      }
    }
	
