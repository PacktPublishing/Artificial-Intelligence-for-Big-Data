package com.aibd.dnn;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public class Normalizer {


    public static void main(String[] args) throws  Exception {
        int numLinesToSkip = 0;
        char delimiter = ',';
        System.out.println("Starting the normalization process");
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);

        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;

        DataSetIterator fulliterator = new RecordReaderDataSetIterator(recordReader,150,labelIndex,numClasses);

        DataSet dataset = fulliterator.next();

        // Original data set
        System.out.println("\n{}\n" + dataset.getRange(0,9));

        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
        System.out.println("Fitting with a dataset...............");
        preProcessor.fit(dataset);
        System.out.println("Calculated metrics");
        System.out.println("Min: {} - "  + preProcessor.getMin());
        System.out.println("Max: {} - " + preProcessor.getMax());

        preProcessor.transform(dataset);
        // Normalized data set
        System.out.println("\n{}\n" + dataset.getRange(0,9));
    }
}
