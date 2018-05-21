package com.aibd.dnn;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NNSumCalculator {

    //Random number generator seed
    public static final int SEED = 12345;

    //Number of data points
    public static final int nSamples = 10000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 2000;
    //Network learning rate
    public static final double learningRate = 0.07;
    // The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges and see how it effects the results
    // also try changing the range along with changing the activation function
    public static int MIN_RANGE = 0;
    public static int MAX_RANGE = 10;

    public static final Random randomNumberGenerator = new Random(SEED);

    //Initialize the user interface backend
    static UIServer uiServer = UIServer.getInstance();

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    static StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later


    //Then add the StatsListener to collect this information from the network, as it trains

    public static void main(String[] args){

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Generate the training data
        DataSetIterator iterator = generateTrainingData(batchSize,randomNumberGenerator);
        System.out.println("Hidden Layer Count, Iterations, Learning Rate, Epoch Count, Time Taken, Error");
        // Test 1: -------------------------------------------------------------------------------------------
        //Create the network
        int nHidden                 = 10;
        int iterations              = 1;
        double learningRate         = 0.01;
        int nEpochs                 = 200;
        double startTime = System.currentTimeMillis();
        MultiLayerNetwork net = generateModel(nHidden,iterations,learningRate,nEpochs,iterator);

        double endTime = System.currentTimeMillis();
        double trainingTime = (endTime - startTime);

        // Test the addition of 2 numbers
        INDArray input = Nd4j.create(new double[] { 0.6754345, 0.3333333333333 }, new int[] { 1, 2 });
        INDArray out = net.output(input, false);
        double actualSum = 0.6754345 + 0.3333333333333;
        double error = actualSum - out.getDouble(0);

        System.out.println(""+nHidden + "," + iterations + "," + learningRate + "," + nEpochs + "," + trainingTime + "," + error );
        // ----------------------------------------------------------------------------------------------------

        // Test 2: -------------------------------------------------------------------------------------------
        //Create the network
        nHidden                 = 10;
        iterations              = 1;
        learningRate            = 0.02;
        nEpochs                 = 200;
        startTime = System.currentTimeMillis();
        net = generateModel(nHidden,iterations,learningRate,nEpochs,iterator);
        endTime = System.currentTimeMillis();
        trainingTime = (endTime - startTime);

        // Test the addition of 2 numbers
        input = Nd4j.create(new double[] { 0.6754345, 0.3333333333333 }, new int[] { 1, 2 });
        out = net.output(input, false);
        actualSum = 0.6754345 + 0.3333333333333;
        error = actualSum - out.getDouble(0);
        System.out.println(""+nHidden + "," + iterations + "," + learningRate + "," + nEpochs + "," + iterations + "," + trainingTime + "," + error );
        // ----------------------------------------------------------------------------------------------------

        // Test 3: -------------------------------------------------------------------------------------------
        //Create the network
        nHidden                 = 10;
        iterations              = 1;
        learningRate            = 0.04;
        nEpochs                 = 200;
        startTime = System.currentTimeMillis();
        net = generateModel(nHidden,iterations,learningRate,nEpochs,iterator);
        endTime = System.currentTimeMillis();
        trainingTime = (endTime - startTime);

        // Test the addition of 2 numbers
        input = Nd4j.create(new double[] { 0.6754345, 0.3333333333333 }, new int[] { 1, 2 });
        out = net.output(input, false);
        actualSum = 0.6754345 + 0.3333333333333;
        error = actualSum - out.getDouble(0);
        System.out.println(""+nHidden + "," + iterations + "," + learningRate + "," + nEpochs + "," + iterations + "," + trainingTime + "," + error );
        // ----------------------------------------------------------------------------------------------------

        // Test 3: -------------------------------------------------------------------------------------------
        //Create the network
        nHidden                 = 10;
        iterations              = 1;
        learningRate            = 0.08;
        nEpochs                 = 200;
        startTime = System.currentTimeMillis();
        net = generateModel(nHidden,iterations,learningRate,nEpochs,iterator);
        endTime = System.currentTimeMillis();
        trainingTime = (endTime - startTime);

        // Test the addition of 2 numbers
        input = Nd4j.create(new double[] { 0.6754345, 0.3333333333333 }, new int[] { 1, 2 });
        out = net.output(input, false);
        actualSum = 0.6754345 + 0.3333333333333;
        error = actualSum - out.getDouble(0);
        System.out.println(""+nHidden + "," + iterations + "," + learningRate + "," + nEpochs + "," + iterations + "," + trainingTime + "," + error );


        // ----------------------------------------------------------------------------------------------------


        nHidden                 = 5;
        iterations              = 1;
        learningRate         = 0.01;
        nEpochs                 = 200;
        startTime = System.currentTimeMillis();
        net = generateModel(nHidden,iterations,learningRate,nEpochs,iterator);

        endTime = System.currentTimeMillis();
        trainingTime = (endTime - startTime);

        // Test the addition of 2 numbers
        input = Nd4j.create(new double[] { 0.6754345, 0.3333333333333 }, new int[] { 1, 2 });
        out = net.output(input, false);
        actualSum = 0.6754345 + 0.3333333333333;
        error = actualSum - out.getDouble(0);

        System.out.println(""+nHidden + "," + iterations + "," + learningRate + "," + nEpochs + "," + trainingTime + "," + error );
    }



    /** Method for generating a multi-layer network
     * @param numHidden - the int value denoting number of nodes in the hidden unit
     * @param iterations - number of iterations per mini-batch
     * @param learningRate - The step size of the gradient descent algorithm
     * @param numEpochs - number of full passes through the data
     * @param trainingDataIterator - the iterator through the randomly generated training data
     * @return the model object (MultiLayerNetwork)
     * */
    private static MultiLayerNetwork generateModel(int numHidden, int iterations, double learningRate, int numEpochs, DataSetIterator trainingDataIterator ) {

        int numInput = 2;   // using two nodes in the input layer
        int numOutput = 1;  // using one node in the output layer
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(numHidden)
                .activation(Activation.TANH)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(numHidden).nOut(numOutput).build())
            .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new StatsListener(statsStorage));

        //Train the network on the full data set, and evaluate in periodically

        for( int i=0; i<numEpochs; i++ ){
            trainingDataIterator.reset();

            net.fit(trainingDataIterator);
        }

        return net;
    }
    // Method to generate the training data based on batch size passed as parameter
    private static DataSetIterator generateTrainingData(int batchSize, Random rand){

        // container for the sum (output variable)
        double [] sum = new double[nSamples];
        // container for the first input variable x1
        double [] input1 = new double[nSamples];
        //container for the second input variable x2
        double [] input2 = new double[nSamples];

        // for set size of the sample in configuration, generate random
        // numbers and fill the containers
        for (int i= 0; i< nSamples; i++) {
            input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            input2[i] =  MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            // fill the dependent variable y
            sum[i] = input1[i] + input2[i];
        }
        // Format in the deeplearning4j data structure
        INDArray inputNDArray1 = Nd4j.create(input1, new int[]{nSamples,1});
        INDArray inputNDArray2 = Nd4j.create(input2, new int[]{nSamples,1});
        INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
        INDArray outPut = Nd4j.create(sum, new int[]{nSamples, 1});
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rand);
        return new ListDataSetIterator(listDs,batchSize);

    }
}
