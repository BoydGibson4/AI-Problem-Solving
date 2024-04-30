import search as search

"""Model an action in the 8-puzzle.
This is a subclass of Action.
An action cost of 1.0 is inherited.
"""
class Sq8Action(search.Action):
    """Create an Sq8Action object.
    :param tile: A tile 1-9.
    :type tile: An int.
    :param fromRow: The row from where the tile is moved.
    :type fromRow: An int 0-2.
    :param fromCol: The column from where the tile is moved.
    :type fromCol: An int 0-2.
    :param toRow: The row to where the tile is moved.
    :type toRow: An int 0-2.
    :param toCol: The column to where the tile is moved.
    :type toCol: An int 0-2.
    """
    def __init__(self,tile,fromRow,fromCol,toRow,toCol):
        super().__init__()
        self.tile=tile
        self.fromRow=fromRow
        self.fromCol=fromCol
        self.toRow=toRow
        self.toCol=toCol

    """Return the action as str for printing.
    """
    def __str__(self):
        return "slide tile {} from ({},{}) to ({},{})\n". \
                format(self.tile,self.fromCol,self.fromRow,self.toCol,self.toRow)

"""Model a state in the 8-puzzle.
This is a subclass of State.
"""
class Sq8State(search.State):
    """Create a Sq8State.
    :param initialPos: Position of the tiles as a 2D int array. If this is None, it will fill the board with 0. (i.e. spaces)
    :type initialPos: None for an empty board, or a 2D int array.
    """
    def __init__(self,initialPos=None):
        super().__init__()
        if initialPos==None:        #if None is given, fill the array with 0 for spaces
            self.tiles=[[0]*3]*3
            print(self.tiles)
        else:
            self.tiles=[]
            for row in initialPos:              #go through the 2D array
                self.tiles.append(row.copy())   #clone the 1D arrays inside the 2D array

    """Return the state as a str for printing.
    """        
    def __str__(self):
        result=""
        for row in self.tiles:          #go through the rows
            for cell in row:            #go through the cells in each row
                if cell==0:             #a space
                    result+="-"         #appears as "-"
                else:
                    result+=str(cell)   #otherwise just append the tile number
            result+="\n"                #ad a newline as the end of each row
        return result

    """Compare if the current state equals to another object.
    :param other: The other object to compare with.
    :rtype: True or False.
    """
    def __eq__(self,other):
        if not isinstance(other,Sq8State):  #if the other object is not a Sq8State
            return False                    #false
        return self.tiles==other.tiles      #otherwise it depends on the equality of the 2D arrays

    """Compute a hash value from the state.
    :rtype: An int.
    """
    def __hash__(self):
        result=0
        for row in self.tiles:
            for cell in row:
                result=result*10+cell
        return result

    """Apply an action to the current state.
    :param action: The action to apply to the current state.
    :type action: A Sq8Action.
    :returns: The next state after the action is applied.
    :rtype: A Sq8State.
    """    
    def applyAction(self,action):
        nextState=Sq8State(self.tiles)  #create next state as copy of current state
        nextState.tiles[action.toRow][action.toCol]=action.tile
        nextState.tiles[action.fromRow][action.fromCol]=0
        return nextState
    
    """Find all valid action and next state pairs from he current state.
    :returns: All children as a list of ActionStatePair.
    :rtype: A list of ActionStatePair.
    """
    def successor(self):
        result=list()       #result starts as an empty list
        
        for blankRow in range(0,len(self.tiles)):       #loop through row index
            try:
                blankCol=self.tiles[blankRow].index(0)  #find 0 in this row
                break                                   #stop loop
            except ValueError:                          #if 0 is not found
                pass                                    #just continue to next row

        if blankCol>0:  #blank not on column 0, can move tile W of blank to E
            tileRow=blankRow        #tile on same row
            tileCol=blankCol-1      #tile column is to the W of blank
            tileToSlide=self.tiles[tileRow][tileCol]    #tile number
            action=Sq8Action(tileToSlide,tileRow,tileCol,blankRow,blankCol)
            nextState=self.applyAction(action)
            actionStatePair=search.ActionStatePair(action,nextState)
            result.append(actionStatePair)
            
        if blankCol<2:  #blank not on column 2, can move tile E of blank to W
            tileRow=blankRow        #tile on same row
            tileCol=blankCol+1      #tile column is to the E of blank
            tileToSlide=self.tiles[tileRow][tileCol]    #tile number
            action=Sq8Action(tileToSlide,tileRow,tileCol,blankRow,blankCol)
            nextState=self.applyAction(action)
            actionStatePair=search.ActionStatePair(action,nextState)
            result.append(actionStatePair)

        if blankRow>0:  #blank not on row 0, can move tile N of blank to S
            tileRow=blankRow-1    #tile row is N of blank
            tileCol=blankCol      #tile is on the same row of blank
            tileToSlide=self.tiles[tileRow][tileCol]    #tile number
            action=Sq8Action(tileToSlide,tileRow,tileCol,blankRow,blankCol)
            nextState=self.applyAction(action)
            actionStatePair=search.ActionStatePair(action,nextState)
            result.append(actionStatePair)
            
        if blankRow<2:  #blank not on row 2, can move tile S of blank to N
            tileRow=blankRow+1    #tile row is S of blank
            tileCol=blankCol      #tile is on the same row of blank
            tileToSlide=self.tiles[tileRow][tileCol]    #tile number
            action=Sq8Action(tileToSlide,tileRow,tileCol,blankRow,blankCol)
            nextState=self.applyAction(action)
            actionStatePair=search.ActionStatePair(action,nextState)
            result.append(actionStatePair)
        
        return result

"""Model the 8-puzzle as a BFS problem.
"""
class Sq8Problem(search.SearchProblem):
    """Create a Sq8Problem.
    :param start: The initial state.
    :type start: A Sq8State.
    :param goal: The goal state.
    :type goal: A Sq8State.
    """
    def __init__(self,start,goal):
        super().__init__(start)
        self.goalState=goal

    """Check if a state is a goal state.
    :param state: The state to test.
    :type state: A Sq8State.
    :returns: True if the given state is a goal. False otherwise.
    :rtype: A boolean True/False.
    """
    def isGoal(self,state):
        return state==self.goalState

"""This version of the Sq8Problem uses DFS.
To do this we simply override the inherited addChild(...) method
which inserts a child node to the beginning of the fringe instead of
appending to the end.
Everything else is the same as in Sq8Problem.
"""
class Sq8ProblemDFS(Sq8Problem):
    def addChild(self, fringe, childNode):
        fringe.insert(0,childNode)


"""The main function.
"""    
def run():
    
    initial=Sq8State([[8,7,6],[5,4,3],[2,1,0]])    #this one takes a long time to solve in Python
#    initial=Sq8State([[6,3,1],[7,4,2],[8,5,0]])     #an easier problem
    goal=Sq8State([[0,1,2],[3,4,5],[6,7,8]])
    
    problem=Sq8Problem(initial,goal)        #problem using BFS
#    problem=Sq8ProblemDFS(initial,goal)    #problem using DFS
    
    print("Searching...\n")
    path=problem.search()
    if path==None:
        print("No solution\n")
    else:
        print(path)
        print("Nodes visited: {}\n".format(problem.nodeVisited))
        print("Cost: {}\n".format(path.cost))

if __name__=="__main__":
    run()
