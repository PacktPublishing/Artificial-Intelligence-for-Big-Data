object FeedForwardNetworkWithSpark {

	def main(args:Array[String]): Unit ={
	val recordReader:RecordReader = new CSVRecordReader(0,",")
	val conf = new SparkConf()
	.setMaster("spark://master:7077")
	.setAppName("FeedForwardNetwork-Iris")
	val sc = new SparkContext(conf)

	val numInputs:Int = 4
	val outputNum = 3
	val iterations =1
	val multiLayerConfig:MultiLayerConfiguration = new
	NeuralNetConfiguration.Builder()
		.seed(12345)
		.iterations(iterations)
		.optimizationAlgo(OptimizationAlgorithm
		.STOCHASTIC_GRADIENT_DESCENT)
		.learningRate(1e-1)
		.l1(0.01).regularization(true).l2(1e-3)
		.list(3)
		.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
		.activation("tanh")
		.weightInit(WeightInit.XAVIER)
		.build())
		.layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
		.activation("tanh")
		.weightInit(WeightInit.XAVIER)
		.build())
		.layer(2, new
		OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
		.weightInit(WeightInit.XAVIER)
		.activation("softmax")
		.nIn(2).nOut(outputNum).build())
		.backprop(true).pretrain(false)
		.build

		val network:MultiLayerNetwork = new	MultiLayerNetwork(multiLayerConfig)
		network.init
		network.setUpdater(null)
		val sparkNetwork:SparkDl4jMultiLayer = new
		SparkDl4jMultiLayer(sc,network)
		val nEpochs:Int = 6
		val listBuffer = new ListBuffer[Array[Float]]()
		(0 until nEpochs).foreach{i =>
		val net:MultiLayerNetwork =
		sparkNetwork.fit("file:///<path>/
		iris_shuffled_normalized_csv.txt",4,recordReader)
		listBuffer +=(net.params.data.asFloat().clone())
		}
		println("Parameters vs. iteration Output: ")
		(0 until listBuffer.size).foreach{i =>
		println(i+"\t"+listBuffer(i).mkString)}
		}
	}
