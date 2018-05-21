package com.aibd.rl;

import java.util.Random;

public class QLearner {

	 	private static final int STATE_COUNT 		= 6;
	    private static final double GAMMA 			= 0.8;
	    private static final int MAX_ITERATIONS 	= 10;
	    private static final int INITIAL_STATES[] 	= new int[] {1, 3, 5, 2, 4, 0};

	    // initialize the R matrix with the state transition combinations
	    private static final int R[][] = 
	    		
	    		new int[][] {{-1, -1, -1, -1, 0, -1}, 
	            			 {-1, -1, -1, 0, -1, 100}, 
	                         {-1, -1, -1, 0, -1, -1}, 
	                         {-1, 0, 0, -1, 0, -1}, 
	                         {0, -1, -1, 0, -1, 100}, 
	                         {-1, 0, -1, -1, 0, 100}};

	    
	    
	    private static int q[][] = new int[STATE_COUNT][STATE_COUNT];
	    private static int currentState = 0;
	    
	    public static void main(String[] args) {
	        train();
	        test();
	        return;
	    }
	    
	    private static void train() {
	    	// initialize the Q matrix to zero values
	    	initialize(); 

	        // Perform training, starting at all initial states.
	        for(int j = 0; j < MAX_ITERATIONS; j++){
	            for(int i = 0; i < STATE_COUNT; i++) {
	                episode(INITIAL_STATES[i]);
	            } 
	        } 

	        System.out.println("Q Matrix:");
	        for(int i = 0; i < STATE_COUNT; i++) {
	            for(int j = 0; j < STATE_COUNT; j++){
	                System.out.print(q[i][j] + ",\t");
	            } 
	            System.out.print("\n");
	        }
	        System.out.print("\n");

	        return;
	    }
	    
	    private static void test() {
	        // Perform tests, starting at all initial states.
	        System.out.println("Shortest routes from initial states:");
	        for(int i = 0; i < STATE_COUNT; i++) {
	            currentState = INITIAL_STATES[i];
	            int newState = 0;
	            do {
	                newState = maximum(currentState, true);
	                System.out.print(currentState + " --> ");
	                currentState = newState;
	            }while(currentState < 5);
	            System.out.print("5\n");
	        }

	        return;
	    }
	    
	    private static void episode(final int initialState) {
	        currentState = initialState;

	        do {
	            chooseAnAction();
	        }while(currentState == 5);

	        for(int i = 0; i < STATE_COUNT; i++){
	            chooseAnAction();
	        }
	        return;
	    }
	    
	    private static void chooseAnAction() {
	        int possibleAction = 0;

	        // Randomly choose a possible action connected to the current state.
	        possibleAction = getRandomAction(STATE_COUNT);

	        if(R[currentState][possibleAction] >= 0){
	            q[currentState][possibleAction] = reward(possibleAction);
	            currentState = possibleAction;
	        }
	        return;
	    }
	    
	    private static int getRandomAction(final int upperBound) {
	        int action = 0;
	        boolean choiceIsValid = false;

	        // Randomly choose a possible action connected to the current state.
	        while(choiceIsValid == false) {
	            // Get a random value between 0(inclusive) and 6(exclusive).
	            action = new Random().nextInt(upperBound);
	            if(R[currentState][action] > -1){
	                choiceIsValid = true;
	            }
	        }

	        return action;
	    }
	    
	    private static void initialize() {
	        for(int i = 0; i < STATE_COUNT; i++)
	        {
	            for(int j = 0; j < STATE_COUNT; j++)
	            {
	                q[i][j] = 0;
	            } // j
	        } // i
	        return;
	    }
	    
	    private static int maximum(final int State, final boolean ReturnIndexOnly) {
	        // If ReturnIndexOnly = True, the Q matrix index is returned.
	        // If ReturnIndexOnly = False, the Q matrix value is returned.
	        int winner = 0;
	        boolean foundNewWinner = false;
	        boolean done = false;

	        while(!done) {
	            foundNewWinner = false;
	            for(int i = 0; i < STATE_COUNT; i++)
	            {
	                if(i != winner){             // Avoid self-comparison.
	                    if(q[State][i] > q[State][winner]){
	                        winner = i;
	                        foundNewWinner = true;
	                    }
	                }
	            }

	            if(foundNewWinner == false){
	                done = true;
	            }
	        }

	        if(ReturnIndexOnly == true){
	            return winner;
	        }else{
	            return q[State][winner];
	        }
	    }
	    
	    private static int reward(final int Action) {
	        return (int)(R[currentState][Action] + (GAMMA * maximum(Action, false)));
	    }

}
