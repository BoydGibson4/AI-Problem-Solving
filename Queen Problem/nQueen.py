import aips.local.search as search
import random

"""Model the state of an N-Queen problem.
"""
class NqState(search.State):
    """This int array stores the columns of all queens.
    We don't need to store the rows as these are implicit in the array index.
    Create a NqState from an int array.
    :param size: The board size.
    :type size: int
    :param positions: Positions of the N queens.
    :type positions: An int list/array.
    """
    def __init__(self,size,positions=None):
        if (positions!=None):
            self.columns=positions.copy()   #create a copy of the array/list
        else:
            self.columns=[]
            for i in range(0,size):
                self.columns.append(random.randint(0,size-1))

    """Return the N-Queen problem state as a str (which can be printed).
    :returns: The current N-Queen problem state as a str.
    :rtype: str
    """
    def __str__(self):
        result=""
        for column in self.columns:                 #go through all column number of each queen
            for i in range(0,len(self.columns)):    #i go from 0 to width of board
                if i==column:       #if there is a queen at this column
                    result+="Q"     #append a "Q"
                else:
                    result+="."     #otherwise append a "."
            result+="\n"            #at the of end row, add a newline
        return result

    """Generate all neighbouring states of current state.
    :returns: All neighbouring states.
    :rtype: A list of State.
    """
    def neighbours(self):
        result=[]
        
        temp=self.columns.copy()                #create a copy of the array
        
        for qIndex in range(0,len(temp)):       #queen index to scan through the array
            originalColumn=temp[qIndex]         #remember column number of this queen
            for column in range(0,len(temp)):   #loop through all possible column values
                if column!=originalColumn:      #except when it is the same as the original column
                    temp[qIndex]=column;        #move this queen to a new column
                    result.append(NqState(0,temp))        #create new state and add into result
            temp[qIndex]=originalColumn;                #restore original column number at this row
        return result


"""Model the n-Queen problem as hill-climbing.
"""
class NQueenHillClimbing(search.LocalSearchProblem):
    """This method overrides the {@link LocalSearchProblem.objective(State) objective(State)} method
    to define the objective function of the N-Queen problem.
    In this implementation, we count the number of conflicts.
    @return The number of conflicts.
    """
    def objective(self,state):
        result=0    #conflict count

        for row in range(0,len(state.columns)): #go through the rows
            for rest in range(row+1,len(state.columns)):         #rest will go from row+1 to N
                if state.columns[row]==state.columns[rest]:    #queens on row "row" and row "rest" are on the same column
                    result+=1                                   #1 more conflict
                else:
                    rowDiff=abs(rest-row)                       #calculate row difference
                    colDiff=abs(state.columns[row]-state.columns[rest]) #calculate column difference
                    if rowDiff==colDiff:                    #if both differences are the same, the 2 queens are on a diagonal
                        result+=1                           #1 more conflict
        return -1*result    #negate result as we want to maximise the negation of conflict count

    """This method performs a domain-independent hill climbing local search.
    :param state: The current state.
    :type state: A State.
    :returns: The final state which cannot be improve further by hill climbing.
    :rtype: A State.
    """
    def search(self,state):
        currentState=state;
        currentScore=self.objective(currentState)
        while True:
            neighbours=currentState.neighbours()    #get all neighbours
            if len(neighbours)==0:                  #no neighbour
                return currentState                 #the current state is the solution
            #
            #We have neighbours. Let us find the best one.
            #
            bestNeighbour=neighbours[0]                        #get 1st neighbour
            bestNeighbourScore=self.objective(bestNeighbour);  #initial best neighbour score
            #
            #find neighbour with the highest score
            #
            for neighbour in neighbours[1:]:                    #scan through the rest
                neighbourScore=self.objective(neighbour)        #get this neighbour's score
                if neighbourScore>bestNeighbourScore:           #this neighbour has a better score
                    bestNeighbour=neighbour                     #now this is the best neighbour
                    bestNeighbourScore=neighbourScore           #remember it

            if currentScore>=bestNeighbourScore:    #current state is better than any neighbour
                return currentState                 #return it
            
            currentState=bestNeighbour              #otherwise move to best neighbour of current state
            currentScore=bestNeighbourScore         #remember current score
            print("Moving to neighbour with score: {}".format(currentScore))
            print(currentState)


"""Model the n-Queen problem as hill-climbing with sideway move.
It is similar to NQueen-hill-climbing except we are changing the search(...) function.
"""
class NQueenSidewayMove(NQueenHillClimbing):
    MAX_SIDEWAY_MOVE=100
    
    """This method performs a domain-independent hill climbing with sideway move.
    :param state: The current state.
    :type state: A State.
    :returns: The final state which cannot be improve further by hill climbing.
    :rtype: A State.
    """
    def search(self,state):
        currentState=state                          #current state
        currentScore=self.objective(currentState)   #score of current state
        sidewayMoveCounter=0                        #initialise sideway move counter
        #
        #infinite loop until we hit some return inside
        #
        while True:
            neighbours=currentState.neighbours()    #get all neighbours
            if len(neighbours)==0:                  #no neighbour
                return currentState                 #the current state is the solution
            #
            #We have neighbours. Let us find the best one.
            #
            bestNeighbour=neighbours[0]                         #to start, best neighbour is 1st neighbour
            bestNeighbourScore=self.objective(bestNeighbour);   #initialise best neighbour score
            #
            #find neighbour with the highest score
            #
            for neighbour in neighbours[1:]:         #scan through the rest to find the best
                neighbourScore=self.objective(neighbour)
                #
                #update best neighbour and its score if needed
                #
                if neighbourScore>bestNeighbourScore:       #this neighbour has a better score
                    bestNeighbour=neighbour
                    bestNeighbourScore=neighbourScore
            #
            #now we know the highest scored neighbour and its score
            #
    
            #reach local maximum at current state
            if currentScore>bestNeighbourScore:     #no better neighbour than current state
                    return currentState             #return current state
        
            #flat landscape
            if currentScore==bestNeighbourScore:    #best neighbour has the same score as current state
                print("Sideway move...")
                sidewayMoveCounter+=1               #increase sideway move counter
                #
                #check if there are too man consecutive sideway moves
                #
                if sidewayMoveCounter>self.MAX_SIDEWAY_MOVE:    #too many consecutive sideway move?
                        return currentState             #no more sideway move, return current state

            #
            #reset counter if we can move uphill again
            #do not reset if we are moving sideway
            #
            if currentScore<bestNeighbourScore:
                print("Moving uphill...")
                sidewayMoveCounter=0
            
            currentState=bestNeighbour                  #otherwise move to best neighbour
            currentScore=bestNeighbourScore             #Note that this will happen if neighbour is better, or flat landscape within sideway move limit
            print("Moving to neighbour with score: {}".format(currentScore))
            print(currentState)

"""Simulated annealing.
"""
class NQueenSimulatedAnnealing(NQueenHillClimbing):
    THRESHOLD=0.0001            #temperature threshold for stopping the search
    MAX_TEMPERATURE=1000.0     #the maximum temperature at time 0
    MAX_OBJECTIVE=0            #this value is needed n calculating probability

    """The main purpose of this constructor is to initialise the MAX_OBJECTIVE attribute based on the size of the problem.
    Our objective function counts the number of conflicting pairs.
    So the maximum possible objective function is the number of combinations in choosing 2 out of n queens.
    @param n
    """
    def __init__(self,n):
        NQueenSimulatedAnnealing.MAX_OBJECTIVE=n*(n-1)/2

    """This method overrides the one defined in NQueenHillClimbing to implement Simulated Annealing.
    :param state: The starting state of the search.
    :returns: The final solution state.
    """
    def search(self,state):
        currentState=state                              #start with state
        currentScore=self.objective(currentState)       #current score

        #initialise time and temperature
        time=0
        temperature=self.MAX_TEMPERATURE
        
        #infinite loop until hitting some return inside
        while True:
            temperature=self.getTemperature(time,temperature)   #get temperature based on time
            print("Time: {} Temperature: {} Score: {}".format(time,temperature,self.objective(currentState)))
            
            #temperature is cold enough to stop
            if temperature<NQueenSimulatedAnnealing.THRESHOLD:
                return currentState                             #return current state as solution
            
            #still warm, continue local search
            neighbours=currentState.neighbours()        #get all neighbours
            
            if len(neighbours)==0:                      #if there is no neighbour
                return currentState                     #the current state is the solution
            
            #
            #pick a random neighbour
            #
            randomIndex=random.randint(0,len(neighbours)-1)         #get a random number in 0..-length-1
            print("Picking neighbour# {} out of {} neighbours.".format(randomIndex,len(neighbours)))
            randomNeighbour=neighbours[randomIndex]                 #pick the neighbour using random index
            randomNeighbourScore=self.objective(randomNeighbour)    #get score of this random neighbour

            #neighbour is better than current state, always move            
            if randomNeighbourScore>currentScore:                   #neighbour is better
                print("Moving to better neighbour with score: {}\n{}".format(randomNeighbourScore,randomNeighbour))
                currentState=randomNeighbour                        #move to this neighbour
                currentScore=randomNeighbourScore                   #update current score
            #neighbour is worse than current state, move based on probability
            else:
                probability=self.probability(currentScore,randomNeighbourScore,temperature)     #get probability to move to worse neighbour
                if random.random()<probability:            #not better but probability allows
                    print("Probability {} allows moving to worse neighbour with score: {}\n{}".format(probability,randomNeighbourScore,randomNeighbour))
                    currentState=randomNeighbour        #move
                    currentScore=randomNeighbourScore;  #update current score
                else:
                    print("Probability to move: {}. Not moving to worse neighbour.".format(probability))
            time+=1             #increment time

    """Find temperature from time.
    Temperate should decrease when time increases.
    In the current implementation temperature drops 5% in each call.
    :param time: Time in the search. Start from 0.
    :param temperature: Current temperature.
    :returns:    The new temperature.
    """
    def getTemperature(self,time,temperature):
        if time==0:
            return NQueenSimulatedAnnealing.MAX_TEMPERATURE
        return temperature*0.95         #every time reduces by 0.05 of current temperature

    """Based on the current state score, neighbour state score and temperature
    find the probability of moving to a "worse" neighbour.
    :param e1: The current state score/objective function value.
    :param e2: The random neighbour score.
    :param temperature: The temperature.
    :returns: The probability of moving to a "worse" neighbour.
    """
    def probability(self,e1,e2,temperature):
        difference=abs(e1-e2)              #find difference between scores
        return temperature/NQueenSimulatedAnnealing.MAX_TEMPERATURE* \
                (1.0-difference/NQueenSimulatedAnnealing.MAX_OBJECTIVE)


"""Main function.
"""
def run():
    n=8
    #nqueenProblem=NQueenHillClimbing()          #create an N-Queen problem as Hill Climbing search
    nqueenProblem=NQueenSidewayMove()          #create an N-Queen problem as Hill Climbing with sideway move
    #nqueenProblem=NQueenSimulatedAnnealing(n)  #create an N-Queen problem as Simulated Annealing
    startState=NqState(n)                                            #create a random starting state
    print(startState)                                                       #print starting state
    solution=nqueenProblem.search(startState)                               #search
    print("Solution found:\n{}\n".format(solution))                         #print result
    print("Objective value: {}".format(nqueenProblem.objective(solution)))  #print objective value of final result

if __name__=="__main__":
    run()
    