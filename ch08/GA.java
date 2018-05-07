package com.aibd;

import java.util.ArrayList;
import java.util.Random;

import org.encog.examples.ml.tsp.genetic.TSPScore;
import org.encog.ml.CalculateScore;

public class GA {
	
	public int solutionSpace;
	public int populationSize;
	public int[] population = null; 
	public int generation;
	public int targetValue; 
	public int maxGenerations;
	public int mutationPercent;
	
	public GA(int solutionSpace, int populationSize,int targetValue, int maxGenerations, int mutationPercent) {
		
		this.solutionSpace 				= solutionSpace;		// Entire solution space in which the algorithm needs to search
		this.populationSize 			= populationSize;		// Size of the random sample from the solution space
		this.targetValue 				= targetValue;			// Value of the target solution
		this.maxGenerations 			= maxGenerations;		// Maximum number of generations (iterations) of the GA
		this.mutationPercent 			= mutationPercent;		// This field defines the percentage of new generation members to be mutated
		
		System.out.println("Solution Space " + this.solutionSpace + " -- Population Size " + this.populationSize + " -- Target Value " + this.targetValue);
		
		population = new int[this.populationSize];				// Initialize the first generation
		for(int i=0; i< this.populationSize; i++) {
			population[i] = new Random().nextInt(this.solutionSpace);
		}
	}
	
	private int getFitness(int chromosome) {
		int distance =  Math.abs(targetValue - chromosome);
		double fitness = solutionSpace / new Double(distance);
		return (int)fitness;
	}
	
	private int[] getInitialPopulation() {
		return population;
	}
	
	private ArrayList <Integer> getSelectionPool() {
		
		ArrayList <Integer> selectionPool = new ArrayList <Integer>();
		
		for(int i=0; i<this.populationSize; i++ ) {
			int memberFitnessScore = getFitness(this.population[i]);
			//System.out.println("Member fitness score = " + memberFitnessScore);
			Integer value = new Integer(this.population[i]);
			for(int j=0; j<memberFitnessScore; j++) {
				selectionPool.add(value);
			}
		}
		return selectionPool;
	}

	public static void main(String[] args) {
		
		GA algorithm = new GA(500000,10000,1234,20,10);
		algorithm.population = algorithm.getInitialPopulation();	// Population initialization
		
		for (int g=0; g<algorithm.maxGenerations; g++) {
		
		System.out.println("********** Generation " + g + " ************");
		ArrayList <Integer> pool = algorithm.getSelectionPool();
		
		Random randomGenerator = new Random();
		
		int[] nextGeneration = new int[algorithm.populationSize];
		
		for(int i=0; i<algorithm.populationSize; i++) {
			
			if(pool.size() == 0)
				break;
			int parent1RandomIndex = randomGenerator.nextInt(pool.size());
			int parent2RandomIndex = randomGenerator.nextInt(pool.size());
			
			int parent1 = pool.get(parent1RandomIndex).intValue();
			int parent2 = pool.get(parent2RandomIndex).intValue();
			
			if(parent1 == algorithm.targetValue || parent2 == algorithm.targetValue) {
				System.out.println("Found a match !!! ");
				System.exit(1);
			}
			
			int child1 = (parent1 + parent2) >  algorithm.solutionSpace ? algorithm.solutionSpace - (parent1 + parent2) : (parent1 + parent2);
			int child2 = Math.abs(parent1 - parent2);
			if (child1 == algorithm.targetValue || child2 == algorithm.targetValue) {
				System.out.println("Found a match !!! ");
				System.exit(1);
			}
			
			double mutatioRate = 0.001; 
			float randomizer = new Random().nextFloat();
			if(randomizer < mutatioRate) {
				System.out.println("Mutating....");
				child1 += new Random().nextInt(1);
				child2 -= new Random().nextInt(1);
			}
				
			if(algorithm.getFitness(child1) > algorithm.getFitness(child2))
				nextGeneration[i] = child1;
			else
				nextGeneration[i] = child2;
		
		}	
			algorithm.population = nextGeneration;
		}
	}
	
	/*public static void main(String[] args) {
		
		GA algorithm = new GA(500000,10000,1234,20,10);
		algorithm.population = algorithm.getInitialPopulation();	// Population initialization
	
		CalculateScore score =  new TSPScore(algorithm.population);
	} */
}
