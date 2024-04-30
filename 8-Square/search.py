#aips library Python version
#By K. Hui

"""This class models a generic action in a search problem.
All domain-specific action classes must extend this class.
This class has a cost attribute defined as every action has a cost.
That means you DO NOT need to re-declare the attribute in your domain-specific action class.
"""
class Action:
    """Constructor of an Action.
    The cost attribute is initialised to a default value of 1.0.
    Do not re-declare this attribute in your subclasses.
    """
    def __init__(self):
        self.cost=1.0       #default action cost is 1.0

    """Return the Action as a printable str.
    In your action subclass, you should tailor it to return a str
    containing details of the action.
    :returns: The action as a str for printing.
    :rtype: A str.
    """     
    def __str__(self):
        return super().__str__()           #default string representation

"""Model a state that represents the configuration of the world.
All your domain-specific state classes must be extend this superclass.
"""
class State:
    """Return the state as a str.
    You are expected to override this method in your domain-specfic state subclasses
    to customise how a state will be printed.
    :returns: The state as a str for printing.
    :rtype: A str.
    """
    def __str__(self):      #to be defined
        pass
    
    """Check if the current state is the same as another object.
    You are expected to override this method in your domain-specific state subclasses.
    Usually when 2 state objects have the same attribute values, they are equal.
    This method is used in checking the history, and the goal state.
    :param other: The other object to check against.
    :type other: Any object.
    :returns: True if the current state is the same as the other object.
    :rtype: A True or False.
    """
    def __eq__(self,other):       #to be defined
        pass

    """Return all valid actions and corresponding new states from the current state.
    You must override this method to return all successors in your domain-specific state subclasses.
    :returns: All valid action-state pairs as a list of ActionStatePair objects.
    :rtype: a list of ActionStatePair objects. 
    """    
    def successor(self):    #to be defined
        pass

"""This class models an action-state pair which represents an action and the next state.
Note: We don't really need this in Python as we can use a tuple.
But for simplicity I am porting this over from the Java version.

During the search process, we pick up a Node from the fringe, which contains a State.
To expand the Node, we invoke the successor method of this State to get all
valid actions and next states. (i.e. its children/successors)

The State#successor) method of the State generates and returns a list of ActionStatePair objects.
Each pair represents the action which can be applied to the current state,
and the new state after the application.
"""
class ActionStatePair:
    """Create an ActionStatePair object.
    :param action: An Action object. You are expected to give a domain-specific Action subclass.
    :type action: An Action, or any of its subclass.
    :param state: A State object. you are expected to give a domain-specific State subclass.
    :type state: A State, or any of is subclass.
    """
    def __init__(self,action,state):
        self.action=action
        self.state=state

    """Return the ActionStatePair as a str.
    :returns: The str representation of the ActionStatePair as a str for printing.
    :rtype: A str.
    """        
    def __str__(self):
        return self.action.__str__()+",\n"+self.state.__str__()

"""This class models a node in the search tree.
A Node is a book-keeping structure.
Apart from the state of the problem it has other attributes.
e.g. The parent attribute points back to the parent node so that we can trace the path when a goal is found.
The action attribute keeps track of the action which brings us to this state/node.
The getCost method tells us the cost from the root to this node.
The getDepth method tells us the level/depth of this node in the search tree.
This is helpful if we want to limit the depth of the search (e.g. in a depth-limited search).
"""
class Node:
    """Create a Node object in the search tree.
    :param state: The state of the domain.
    :type state: A State. In most cases it should be a domain-specific State subclass defined by you. 
    :param parent: The parent node in the search tree.
    :type parent: A Node.
    :param action: The Action that leads the parent node to this node.
    :type action: An Action. In most cases it should be a domain-specific Action subclass defined by you.
    """
    def __init__(self,state,parent,action):
        self.state=state
        self.parent=parent
        self.action=action

    """Return path cost from the root node to this node.
    """
    def getCost(self):
        '''recursive version may crash on very deep path
        if self.parent==None:
            return 0.0
        return self.action.cost+self.parent.getCost()
        '''
        #iterative version to remove tail recursion
        result=0.0
        current=self
        while current.parent!=None:
            result+=current.action.cost
            current=current.parent
        return result

    """Return the path depth from the root node to this node.
    """
    def getDepth(self):
        '''recursive version may crash on very deep path
        if self.parent==None:
            return 0
        return self.parent.getDepth()+1
        '''
        #iterative version to remove tail recursion
        result=0
        current=self
        while current.parent!=None:
            result+=1
            current=current.parent
        return result

"""Model a path which is the result of a successful search.
A path is simply a head Node followed by a list of action-state pairs.
"""
class Path:
    """Create an empty Path.
    """
    def __init__(self):
        self.head=None
        self.cost=0.0
        self.list=[]
    
    """Return a Path as a str for printing.
    :returns: A str representation of the path.
    :rtype: A str.
    """
    def __str__(self):
        if self.head==None:
            return ""
        
        result=self.head.__str__()+"\n"
        for x in self.list:
            result+=x.action.__str__()+"\n"
            result+=x.state.__str__()+"\n\n"
        return result

    #to insert new data into a position        
    """Insert a new data into a position of the list/path.
    :param index: The index position to insert.
    :type index: An int.
    :param data: The ActionStatePair to be inserted into the path.
    :type data: An ActionStatePair.
    """
    def insert(self,index,data):
        self.list.insert(index,data)
    
"""Model an uninformed search.
"""
class SearchProblem:
    """Create a SearchProblem.
    :param start: The initial state.
    :type start: A State. You are expected to use a domain-specific State subclass.
    """
    def __init__(self,start):
        self.startState=start
        self.nodeVisited=0

    """To search for a solution.
    :returns: The solution of the search as a Path. Or None if no solution is found.
    :rtype: A Path.
    """        
    def search(self):
        visitedState=set()  #empty set of visited states
        fringe=[]       #empty list of fringe
        
        newNode=Node(self.startState,None,None)   #create node from initial state
        fringe.append(newNode)                          #add into fringe
        self.nodeVisited+=1
        
        
        while True:
            if fringe==[]:   #no more node in fringe
                return None         #no solution
            
            node=fringe.pop(0)      #remove 1st node from fringe

            if self.isGoal(node.state): #goal is found
                return self.constructPath(node)
            
            if not node.state in visitedState:  #state of node not in history
                childrenNodes=node.state.successor()    #expand node to get children
                visitedState.add(node.state)            #add state into history
                self.addChildrenNodes(fringe,node,childrenNodes)  #add children into fringe

    """To add a list of nodes into the fringe.
    :param fringe: The fringe of unexplored nodes.
    :type fringe: A list of Node.
    :param parentNode: The parent node of these children nodes.
    :type parentNode: A Node.
    :param childrenNodes: A list of ActionStatePair on expanding the parent node.
    :type childrenNodes: A list of ActionStatePair.
    """    
    def addChildrenNodes(self,fringe,parentNode,childrenNodes):
        for actionState in childrenNodes:               #go through all children
            action=actionState.action                   #get action from ActionStatePair
            childState=actionState.state                #get state
            childNode=Node(childState,parentNode,action)    #create a new Node
            self.addChild(fringe,childNode)             #add new node into fringe
            self.nodeVisited+=1                         #increment node count

    """Add a child node into the fringe.
    This method is used by addChildrenNodes(...) to add a single child into the fringe.
    :param fringe: The list of Node waiting to be explored.
    :type fringe: A list of Node.
    :param childNode: An ActionStatePair representing a child after expanding a node.
    :type childNode: An ActionStatePair.
    """    
    def addChild(self,fringe,childNode):
        fringe.append(childNode)    #default is append to the end of the fringe. i.e. BFS

    """Build a Path by reverse traversing a tree from a node to the root.
    The node contains a goal state. By reversely following the parent node all the
    way up to the root, we can build the path from the root to the goal.
    :param node: The node where a goal state is found.
    :type node: A Node. 
    """        
    def constructPath(self,node):
        if node==None:
            return None
        result=Path()
        result.cost=node.getCost()
        while (node.parent!=None):
            actionStatePair=ActionStatePair(node.action,node.state) #create action-state pair
            result.insert(0,actionStatePair)                        #insert into beginning of result
            node=node.parent                                        #move up to parent node and continue
        result.head=node.state  #head of path is the root node's state
        return result
    
    """Test if a goal state is reached.
    In general this is goal test which needs to be defined by the search problem.
    You are expected to override this method in your domain-specific search problem class.
    For most cases you will have an attribute in your problem class and check if the given state equals to the attribute.
    :param state: The state to check if it is a goal.
    :type state: A boolean True or False.
    """
    def isGoal(self,state):
        pass
