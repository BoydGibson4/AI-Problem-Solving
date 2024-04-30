#Boyd Gibson 2310319

# Import necessary modules
from search import SearchProblem, State, Action as BaseAction, Node # Import the required library
import heapq  # For priority queue implementation
from itertools import combinations  # For generating combinations of dogs

# Define the Action class for representing actions
class Action:
    def __init__(self, cost=1.0):
        self.cost = cost

    def __str__(self):
        return f"Action (cost={self.cost})"

# Define the DogWalkingState class inheriting from State
class DogWalkingState(State):

    def __init__(self, site_a, site_b, walker_at_a=True, walker=None):
        """
        Initializes a state representing the configuration of dogs and walker at different sites.

        Args:
            site_a (list): List of Dog objects representing dogs at site A.
            site_b (list): List of Dog objects representing dogs at site B.
            walker_at_a (bool, optional): Boolean indicating whether the walker is at site A. Defaults to True.
            walker (Walker, optional): Walker object representing the person walking the dogs. Defaults to None.
        """
        super().__init__()  # Call the parent class constructor
        self.site_a = list(site_a)  # List of dogs at site A
        self.site_b = list(site_b)  # List of dogs at site B
        self.walker_at_a = walker_at_a  # Boolean indicating walker's location
        self.walker = walker  # Walker object

    def __str__(self):
        """
        Returns a string representation of the state, including the configuration of dogs at both sites and the walker's location.

        Returns:
            str: String representation of the state.
        """
        # Convert site A and site B dogs to string for printing
        site_a_str = "\n".join(str(dog) for dog in self.site_a)
        site_b_str = "\n".join(str(dog) for dog in self.site_b)
        walker_str = "Walker is at Site A" if self.walker_at_a else "Walker is at Site B"
        return f"Site A:\n{site_a_str}\nSite B:\n{site_b_str}\n{walker_str}"

    def __eq__(self, other):
        """
        Checks if two states are equal by comparing their attributes.

        Args:
            other (DogWalkingState): Another DogWalkingState object to compare with.

        Returns:
            bool: True if the states are equal, False otherwise.
        """
        # Check if two states are equal
        return (
            isinstance(other, DogWalkingState) and
            self.site_a == other.site_a and
            self.site_b == other.site_b and
            self.walker == other.walker
        )

    def __hash__(self):
        """
        Generates a hash value for the state based on its attributes.

        Returns:
            int: Hash value for the state.
        """
        # Generate a hash value for the state
        if self.walker:
            return hash((tuple(self.site_a), tuple(self.site_b), hash(self.walker), self.walker.time))
        else:
            return hash((tuple(self.site_a), tuple(self.site_b)))

    def generate_combinations(self, walker_site):
        """
        Generate all valid combinations of dogs to move from a given site within the walker's capacity.

        Args:
            walker_site (list): List of Dog objects representing the site where the walker is currently present.

        Returns:
            list: List of tuples, where each tuple represents a valid combination of dogs to move.
        """
        # Generate all combinations of dogs to move from a site
        combinations_list = []
        for i in range(1, len(walker_site) + 1):
            for dogs_combination in combinations(walker_site, i):
                # Calculate the total attention of dogs in the combination
                total_attention = sum(dog.attention for dog in dogs_combination)
                
                # Check if the total attention of dogs in the combination is within walker's capacity
                if total_attention <= self.walker.capacity:
                    combinations_list.append(dogs_combination)
        return combinations_list


    def successor(self):
        """
        Generate successor states based on possible moves from the current state.

        Returns:
            list: List of ActionStatePair objects representing possible successor states.
        """
        # Generate successor states based on possible moves
        successors = []

        # Determine the site where the walker is currently present
        walker_site = self.site_a if self.walker_at_a else self.site_b

        # Generate combinations of dogs to move that meet the walker's attention capacity
        combinations_list = self.generate_combinations(walker_site)

        # Iterate over each combination of dogs to move
        for dogs_combination in combinations_list:
            # Create new states for each possible move combination
            new_site_a = self.site_a[:]
            new_site_b = self.site_b[:]
            new_walker_at_a = not self.walker_at_a

            # Update the new site and walker's location based on the move
            for dog in dogs_combination:
                if self.walker_at_a:
                    new_site_a.remove(dog)
                    new_site_b.append(dog)
                else:
                    new_site_b.remove(dog)
                    new_site_a.append(dog)

            # Create a new state representing the move
            new_state = DogWalkingState(new_site_a, new_site_b, new_walker_at_a, self.walker)

            # Calculate the cost of the move
            old_walking_time = sum(dog.walking_time for dog in walker_site)
            new_walking_time = sum(dog.walking_time for dog in new_site_a + new_site_b)
            time_difference = old_walking_time - new_walking_time

            old_attention = sum(dog.attention for dog in walker_site)
            new_attention = sum(dog.attention for dog in new_site_a + new_site_b)
            attention_difference = old_attention - new_attention

            max_walk_time_dogs_moved = max(dog.walking_time for dog in dogs_combination)

            cost = max(max_walk_time_dogs_moved, time_difference, attention_difference)

            # Calculate the total attention required for this move
            total_attention_required = sum(dog.attention for dog in dogs_combination)

            # Create the Move instance with the correct parameters
            dog_names = ', '.join(dog.name for dog in dogs_combination)
            move = Move(move_type="MOVE", dog_name=dog_names, attention_required=total_attention_required, walker=None, cost=cost)
            action_state_pair = ActionStatePair(move, new_state)
            successors.append(action_state_pair)

        return successors

    def heuristic(self, state):
        """
        Calculate the heuristic value for a given state.

        Args:
            state (DogWalkingState): The state for which to calculate the heuristic.

        Returns:
            float: The heuristic value for the state.
        """
        # Calculate the heuristic value for the state
        remaining_walking_time_a = sum(dog.walking_time for dog in state.site_a)
        remaining_walking_time_b = sum(dog.walking_time for dog in state.site_b)
        if state.walker_at_a:
            remaining_walking_time_a -= state.walker.time
        else:
            remaining_walking_time_b -= state.walker.time
        remaining_attention_a = max(0, state.walker.capacity - sum(dog.attention for dog in state.site_a))
        remaining_attention_b = max(0, state.walker.capacity - sum(dog.attention for dog in state.site_b))
        remaining_cost_a = max(remaining_walking_time_a, remaining_attention_a)
        remaining_cost_b = max(remaining_walking_time_b, remaining_attention_b)
        num_dogs_a = len(state.site_a)
        num_dogs_b = len(state.site_b)
        if state.site_a:
            remaining_cost_a += max(0, num_dogs_b - num_dogs_a) * max(dog.walking_time for dog in state.site_a)
        if state.site_b:
            remaining_cost_b += max(0, num_dogs_a - num_dogs_b) * max(dog.walking_time for dog in state.site_b)
        return max(remaining_cost_a, remaining_cost_b)


    def calculate_cost(self, action):
        """
        Calculate the cost of a given action.

        Args:
            action (ActionStatePair): The action for which to calculate the cost.

        Returns:
            float: The cost of the action.
        """
        # Calculate the cost of an action
        total_attention_required = action.attention_required + self.startState.walker.time
        if total_attention_required > self.startState.walker.capacity:
            return float('inf')  # Penalize actions that exceed the walker's attention capacity heavily
        else:
            max_walk_time_dogs_moved = max(dog.walking_time for dog in action.state.site_a + action.state.site_b)
            time_difference = self.startState.walker.time - max_walk_time_dogs_moved
            num_dogs_left_alone = len(self.startState.site_a) + len(self.startState.site_b) - len(action.state.site_a) - len(action.state.site_b)
            return max(max_walk_time_dogs_moved, time_difference) + num_dogs_left_alone





# Define the Walker class inheriting from State
class Walker(State):
    def __init__(self, capacity, time, site_a=None, site_b=None):
        """
        Initialize a Walker object.

        Args:
            capacity (int): Attention capacity of the walker.
            time (int): Time spent by the walker.
            site_a (list, optional): List of dogs at site A. Defaults to None.
            site_b (list, optional): List of dogs at site B. Defaults to None.
        """
        super().__init__()  # Call the parent class constructor
        self.capacity = capacity  # Attention capacity of the walker
        self.time = time  # Time spent by the walker
        self.site_a = site_a or []  # List of dogs at site A
        self.site_b = site_b or []  # List of dogs at site B

    def __str__(self):
        """
        Return a string representation of the Walker object.

        Returns:
            str: String representation of the Walker object.
        """
        # Convert site A and site B dogs to string for printing
        site_a_str = "\n".join(str(dog) for dog in self.site_a)
        site_b_str = "\n".join(str(dog) for dog in self.site_b)
        return f"Walker (attention capacity:{self.capacity} time:{self.time} site_a:\n{site_a_str} site_b:\n{site_b_str})"

    def __eq__(self, other):
        """
        Check if two Walker objects are equal.

        Args:
            other (Walker): Another Walker object to compare with.

        Returns:
            bool: True if the two Walker objects are equal, False otherwise.
        """
        # Check if two walker objects are equal
        return (
            isinstance(other, Walker) and
            self.capacity == other.capacity and
            self.time == other.time and
            self.site_a == other.site_a and
            self.site_b == other.site_b
        )

    def __hash__(self):
        """
        Generate a hash value for the Walker object.

        Returns:
            int: Hash value for the Walker object.
        """
        # Generate a hash value for the walker object
        return hash((self.capacity, self.time, tuple(self.site_a), tuple(self.site_b)))




# Define the ActionStatePair class to represent an action and its resulting state
class ActionStatePair:
    def __init__(self, action, state):
        """
        Initialize an ActionStatePair object.

        Args:
            action (Action): Action object.
            state (State): State object.
        """        
        self.action = action  # Action object
        self.state = state  # State object




# Define the AStarNode class to represent a node in A* search
class AStarNode:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        """
        Initialize an AStarNode object.

        Args:
            state (State): State object.
            parent (AStarNode, optional): Parent node. Defaults to None.
            action (Action, optional): Action object. Defaults to None.
            cost (float, optional): Cost. Defaults to 0.
            heuristic (float, optional): Heuristic value. Defaults to 0.
        """
        self.state = state  # State object
        self.parent = parent  # Parent node
        self.action = action  # Action object
        self.cost = cost  # Cost
        self.heuristic = heuristic  # Heuristic value

    def __lt__(self, other):
        """
        Compare two nodes based on their total cost (cost + heuristic).

        Args:
            other (AStarNode): Another AStarNode object to compare with.

        Returns:
            bool: True if the total cost of self is less than other, False otherwise.
        """
        # Compare two nodes based on their total cost (cost + heuristic)
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def getCost(self):
        """
        Get the total cost of the node.

        Returns:
            float: Total cost of the node.
        """
        # Get the total cost of the node
        return self.cost + self.heuristic




# Define the Node class to represent a node in search algorithms
class Node:
    def __init__(self, state, parent, action, cost=0, heuristic=0):
        """
        Initialize a Node object.

        Args:
            state (State): State object.
            parent (Node): Parent node.
            action (Action): Action object.
            cost (float, optional): Cost. Defaults to 0.
            heuristic (float, optional): Heuristic value. Defaults to 0.
        """
        self.state = state  # State object
        self.parent = parent  # Parent node
        self.action = action  # Action object
        self.cost = cost  # Cost
        self.heuristic = heuristic  # Heuristic value

    def getCost(self):
        """
        Get the total cost of the node.

        Returns:
            float: Total cost of the node.
        """
        # Get the total cost of the node
        return self.cost + self.heuristic

    def __lt__(self, other):
        """
        Compare two nodes based on their total cost (cost + heuristic).

        Args:
            other (Node): Another Node object to compare with.

        Returns:
            bool: True if the total cost of self is less than other, False otherwise.
        """
        # Compare two nodes based on their total cost (cost + heuristic)
        return (self.cost + self.heuristic) < (other.cost + other.heuristic) if other else False

    def getDepth(self):
        """
        Get the depth of the node in the search tree.

        Returns:
            int: Depth of the node in the search tree.
        """
        # Get the depth of the node in the search tree
        result = 0
        current = self
        while current.parent is not None:
            result += 1
            current = current.parent
        return result




# Define the Path class to represent a path in the search tree
class Path:
    def __init__(self):
        """
        Initialize a Path object.
        """
        self.head = None  # Head of the path
        self.cost = 0.0  # Cost of the path
        self.list = []  # List of action-state pairs

    def insert(self, index, data):
        """
        Insert an action-state pair at a specified index in the list.

        Args:
            index (int): Index at which to insert the data.
            data (ActionStatePair): ActionStatePair object to insert.
        """
        # Insert an action-state pair at a specified index in the list
        self.list.insert(index, data)




# Define the SearchProblem class for general search problems
class SearchProblem:
    def __init__(self, start):
        """
        Initialize a SearchProblem object.

        Args:
            start (State): Start state of the problem.
        """
        self.startState = start  # Start state of the problem
        self.nodeVisited = 0  # Number of nodes visited during search

    def search(self):
        """
        Implement the search algorithm.

        Returns:
            tuple: Tuple containing the solution path, number of nodes visited, and cost.
        """
        # Implement search algorithm
        visitedState = set()  # Set to keep track of visited states
        fringe = []  # Priority queue for nodes
        newNode = Node(self.startState, None, None, 0, self.heuristic(self.startState))  # Create a new node
        heapq.heappush(fringe, newNode)  # Push the start node to the fringe
        self.nodeVisited += 1  # Increment node visited count

        while True:
            if not fringe:
                print("No solution found.")
                return None, self.nodeVisited, 0  # Return if fringe is empty

            node = heapq.heappop(fringe)  # Pop the node with the lowest total cost from the fringe

            if self.isGoal(node.state):  # Check if the goal state is reached
                return self.constructPath(node), self.nodeVisited, node.getCost()  # Return solution path, nodes visited, and cost

            if node.state not in visitedState:
                visitedState.add(node.state)  # Mark the current state as visited
                self.nodeVisited += 1  # Increment node visited count
                childrenNodes = self.successor(node.state)  # Generate successor nodes
                for action_state_pair in childrenNodes:
                    action, state = action_state_pair.action, action_state_pair.state
                    cost = node.cost + self.calculate_cost(action)  # Calculate the cost of the action
                    heuristic = self.heuristic(state)  # Calculate heuristic value for the state
                    child_node = Node(state, node, action, cost, heuristic)  # Create a new node
                    heapq.heappush(fringe, child_node)  # Push the child node to the fringe

    # Helper method to add children nodes to the fringe
    def addChildrenNodes(self, fringe, parentNode, childrenNodes):
        """
        Add children nodes to the fringe.

        Args:
            fringe (list): Priority queue for nodes.
            parentNode (Node): Parent node.
            childrenNodes (list): List of children nodes.
        """
        for actionState in childrenNodes:
            action, childState = actionState.action, actionState.state
            cost = parentNode.cost + self.calculate_cost(action)  # Calculate the cost of the action
            heuristic = self.heuristic(childState)  # Calculate heuristic value for the child state
            childNode = Node(childState, parentNode, action, cost, heuristic)  # Create a new node
            self.addChild(fringe, childNode)  # Add the child node to the fringe

    def addChild(self, fringe, childNode):
        """
        Add a child node to the fringe with total cost as priority.

        Args:
            fringe (list): Priority queue for nodes.
            childNode (Node): Child node to add to the fringe.
        """
        # Add a child node to the fringe with total cost as priority
        total_cost = childNode.cost + childNode.heuristic
        heapq.heappush(fringe, (total_cost, childNode))

    def heuristic(self, state):
        """
        Calculate the heuristic value for the state.

        Args:
            state (State): State object.

        Returns:
            float: Heuristic value for the state.
        """
        # Calculate the heuristic value for the state
        # Heuristic is the sum of the absolute differences between walker's attention and each dog's attention in both sites
        walker_position = state.walker.site_a + state.walker.site_b
        goal_position = [dog for dog_list in [state.site_a, state.site_b] for dog in dog_list]
        return sum(abs(walker.attention - goal.attention) for walker in walker_position for goal in goal_position)

    def constructPath(self, node):
        """
        Construct the solution path given the final node.

        Args:
            node (Node): Final node.

        Returns:
            Path: Solution path.
        """
        # Construct the solution path given the final node
        if node is None:
            return None
        result = Path()
        result.cost = node.cost
        # Traverse back from the goal node to the start node
        while node.parent is not None:
            action = Action(move_type="MOVE", dog_name=None, attention_required=0, walker=None, cost=node.cost - node.parent.cost)
            action_state_pair = ActionStatePair(action, node.state)
            result.insert(0, action_state_pair)
            node = node.parent
        result.head = node.state
        return result




# Dog class definition
class Dog:
    def __init__(self, name, attention, walking_time, dislikes=None):
        """
        Initialize a Dog object.

        Args:
            name (str): Name of the dog.
            attention (int): Attention level of the dog.
            walking_time (int): Time required for walking the dog.
            dislikes (list, optional): List of other dogs that this dog dislikes. Defaults to None.
        """
        # Initialize dog attributes
        self.name = name
        self.attention = attention
        self.walking_time = walking_time
        self.dislikes = list(dislikes) if dislikes else []

    def __str__(self):
        """
        Return a string representation of the Dog object.

        Returns:
            str: String representation of the Dog object.
        """
        # String representation of a dog
        dislikes_str = ", ".join(dog.name for dog in self.dislikes)
        return f"{self.name} (Attention: {self.attention}, Walking Time: {self.walking_time}, Dislikes: {dislikes_str})"

    def __eq__(self, other):
        """
        Check if two Dog objects are equal.

        Args:
            other (Dog): Another Dog object to compare with.

        Returns:
            bool: True if the two Dog objects are equal, False otherwise.
        """
        # Check if two dog objects are equal
        return (
            isinstance(other, Dog) and
            self.name == other.name and
            self.attention == other.attention and
            self.walking_time == other.walking_time and
            self.dislikes == other.dislikes
        )

    def __hash__(self):
        """
        Generate a hash value for the Dog object.

        Returns:
            int: Hash value for the Dog object.
        """
        # Generate a hash value for the dog object
        return hash((self.name, self.attention, self.walking_time, tuple(self.dislikes)))




# Move class definition
class Move(BaseAction):
    def __init__(self, move_type=None, dog_name=None, attention_required=0, walker=None, cost=0):
        """
        Initialize a Move object.

        Args:
            move_type (str, optional): Type of move. Defaults to None.
            dog_name (str, optional): Name of the dog involved in the move. Defaults to None.
            attention_required (int, optional): Attention required for the move. Defaults to 0.
            walker (Walker, optional): Walker object involved in the move. Defaults to None.
            cost (int, optional): Cost of the move. Defaults to 0.
        """
        # Initialize move attributes
        super().__init__()
        self.move_type = move_type
        self.dog_name = dog_name
        self.attention_required = attention_required
        self.walker = walker
        self.cost = cost

    def __str__(self):
        """
        Return a string representation of the Move object.

        Returns:
            str: String representation of the Move object.
        """
        # String representation of a move
        if self.dog_name is not None:
            return f"Move type: {self.move_type}, Dog name: {self.dog_name}, Attention required: {self.attention_required}, Cost: {self.cost}"
        else:
            return f"Move type: {self.move_type}, Attention required: {self.attention_required}, Cost: {self.cost}"




# DogWalkingProblem class definition
class DogWalkingProblem(SearchProblem):
    def __init__(self, start_state, goal_state, walker):
        """
        Initialize a DogWalkingProblem object.

        Args:
            start_state (State): Start state of the problem.
            goal_state (State): Goal state of the problem.
            walker (Walker): Walker object associated with the problem.
        """
        # Initialize DogWalkingProblem attributes
        super().__init__(start_state)
        self.goal_state = goal_state
        self.walker = walker
        self.visited_states = set()

    def heuristic(self, state):
        """
        Calculate the heuristic value for a given state.

        Args:
            state (State): State for which to calculate the heuristic value.

        Returns:
            float: Heuristic value for the state.
        """
        # Calculate the heuristic value for the state
        # Heuristic is the remaining walking time and attention capacity at each site
        remaining_walking_time_a = sum(dog.walking_time for dog in state.site_a)
        remaining_walking_time_b = sum(dog.walking_time for dog in state.site_b)
        if state.walker_at_a:
            remaining_walking_time_a -= state.walker.time
        else:
            remaining_walking_time_b -= state.walker.time
        remaining_attention_a = max(0, state.walker.capacity - sum(dog.attention for dog in state.site_a))
        remaining_attention_b = max(0, state.walker.capacity - sum(dog.attention for dog in state.site_b))
        remaining_cost_a = max(remaining_walking_time_a, remaining_attention_a)
        remaining_cost_b = max(remaining_walking_time_b, remaining_attention_b)
        num_dogs_a = len(state.site_a)
        num_dogs_b = len(state.site_b)
        if state.site_a:
            remaining_cost_a += max(0, num_dogs_b - num_dogs_a) * max(dog.walking_time for dog in state.site_a)
        if state.site_b:
            remaining_cost_b += max(0, num_dogs_a - num_dogs_b) * max(dog.walking_time for dog in state.site_b)
        return max(remaining_cost_a, remaining_cost_b)

    def calculate_cost(self, action):
        """
        Calculate the cost of an action.

        Args:
            action (Action): Action for which to calculate the cost.

        Returns:
            int: Cost of the action.
        """
        # Calculate the cost of an action
        # Check if the cost of the dog is less that the walker
        # If so then change cost to that of the walker
        if action.cost < self.startState.walker.time:
            action.cost = self.startState.walker.time
        return action.cost

    def isGoal(self, state):
        """
        Check if a given state is a goal state.

        Args:
            state (State): State to check.

        Returns:
            bool: True if the state is a goal state, False otherwise.
        """
        # Placeholder method for checking if a state is a goal state
        return state == self.goal_state

    def actions(self, state):
        """
        Get possible actions for a given state.

        Args:
            state (State): State for which to get possible actions.

        Returns:
            list: List of possible actions for the state.
        """
        # Get possible actions for a state
        return state.successor()

    def successor(self, state):
        """
        Generate successor states for a given state.

        Args:
            state (State): State for which to generate successor states.

        Returns:
            list: List of ActionStatePair objects representing the successor states.
        """
        # Generate successor states for a given state
        return [ActionStatePair(action, successor_state) for action_state_pair in state.successor() for action, successor_state in [(action_state_pair.action, action_state_pair.state)]]

    def constructPath(self, node):
        """
        Construct the solution path given the final node.

        Args:
            node (AStarNode): Final node of the search.

        Returns:
            Path: Solution path.
        """
        # Construct the solution path given the final node
        if node is None:
            return None
        result = Path()
        result.cost = node.getCost()
        # Traverse back from the goal node to the start node
        while node.parent is not None:
            actionStatePair = ActionStatePair(node.action, node.state)
            result.insert(0, actionStatePair)
            node = node.parent
        result.head = node.state
        return result

    def addChild(self, fringe, childNode):
        """
        Add a child node to the fringe with total cost as priority.

        Args:
            fringe (list): Priority queue for A* search.
            childNode (AStarNode): Child node to be added to the fringe.
        """
        # Add child node to the fringe with total cost as priority
        total_cost = childNode.cost + childNode.heuristic
        heapq.heappush(fringe, (total_cost, childNode))

    def search(self):
        """
        Search for a solution to the problem using A* search.

        Returns:
            tuple: Solution path, number of nodes visited, and cost.
        """
        # Search for a solution to the problem using A* search
        fringe = []  # Priority queue for A* search
        heapq.heappush(fringe, (self.startState.heuristic, AStarNode(self.startState)))  # Push start state with heuristic as priority
        visited_states = set()

        while fringe:
            _, current_node = heapq.heappop(fringe)  # Pop the node with the lowest priority (based on cost + heuristic)

            if current_node.state in visited_states:
                continue

            visited_states.add(current_node.state)

            if self.isGoal(current_node.state):
                return self.constructPath(current_node), len(visited_states), current_node.cost

            for action_state_pair in self.actions(current_node.state):
                action, successor_state = action_state_pair.action, action_state_pair.state
                cost = current_node.cost + self.calculate_cost(action)
                heuristic = self.heuristic(successor_state)
                child_node = AStarNode(successor_state, current_node, action, cost, heuristic)
                heapq.heappush(fringe, (cost + heuristic, child_node))

        print("No solution found.")
        return None, len(visited_states), 0




def print_solution(problem, goal_state, solution_path, nodes_explored, cost):
    """
    Print the solution to the problem.

    Args:
        problem (DogWalkingProblem): The DogWalkingProblem instance.
        goal_state (State): The goal state of the problem.
        solution_path (Path): The solution path.
        nodes_explored (int): Number of nodes explored during the search.
        cost (int): Cost of the solution.
    """
    #Prints the solution
    print("===Problem===")
    print("Start:")
    print("------")
    print(problem.startState)
    print("Finish:")
    print("------")
    print(goal_state)

    if solution_path:
        print("===Solution===")
        print(f"Node explored: {nodes_explored}, Cost: {cost}")

        current_state = problem.startState
        print(current_state)

        for action_state_pair in solution_path.list:
            action, successor_state = action_state_pair.action, action_state_pair.state
            
            # Identify the dogs moving
            dogs_moving = []
            for dog in current_state.site_a + current_state.site_b:
                if dog not in successor_state.site_a + successor_state.site_b:
                    dogs_moving.append(dog)

            # Print the action first
            print(action)

            # Print the successor state
            print(successor_state)

            # Print the walker's movement information
            print(f"->Walker (attention capacity: {successor_state.walker.capacity}, time: {successor_state.walker.time})")

            current_state = successor_state

        print("Final State")
        print("------")
        print(current_state)

        #seperation to improve the presentation of the output
        print("------")
        print("")
        print("")
        print("")
        print("------")
    else:
        print("===Solution===")
        print("No solution found.")

        #seperation to improve the presentation of the output
        print("------")
        print("")
        print("")
        print("")
        print("------")



    
    

"""
General Problem:
This part of the code defines a general problem of dog walking where dogs are initially placed
in two different sites (site_a and site_b) along with a walker. It then initializes a DogWalkingProblem
instance and solves it to find a solution path.
Define dogs and initialize walker
Define dogs at site_a and site_b with their respective attributes
Initialize the walker with its capacity and time attributes
"""      
dog4 = Dog(name="Dog4", attention=4, walking_time=11)
dog5 = Dog(name="Dog5", attention=2, walking_time=7)
dog6 = Dog(name="Dog6", attention=3, walking_time=6)
dog1 = Dog(name="Dog1", attention=1, walking_time=12, dislikes=[dog4])
dog2 = Dog(name="Dog2", attention=2, walking_time=8, dislikes=[dog6])
dog3 = Dog(name="Dog3", attention=3, walking_time=5)
initial_walker = Walker(capacity=6, time=8)

# Initialize the initial and goal states
# Define the initial state with dogs placed at site_a and site_b, walker position, and walker attributes
# Define the goal state with dogs rearranged compared to the initial state and walker position
initial_state = DogWalkingState(site_a=[dog3, dog2, dog1], site_b=[dog6, dog5, dog4], walker_at_a=False, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog6, dog5, dog4], site_b=[dog3, dog2, dog1], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===General Problem===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)









# Extreme solutions:
"""
Example 2:
This example involves a scenario with fewer dogs and a relatively simple setup compared to other examples.
It initializes a new problem instance and solves it to find a solution path.
Define dogs and initialize initial state
Define dogs at site_a and site_b with their respective attributes
Initialize the initial state with dogs placed at site_a and site_b, walker position, and walker attributes
Define the goal state with dogs rearranged compared to the initial state and walker position
"""
dog7 = Dog(name="Dog7", attention=5, walking_time=10)
dog8 = Dog(name="Dog8", attention=4, walking_time=8)
dog9 = Dog(name="Dog9", attention=2, walking_time=6)
initial_state = DogWalkingState(site_a=[dog7, dog8], site_b=[dog9], walker_at_a=True, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog9], site_b=[dog7, dog8], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===Example 2===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)






"""
Example 3:
This example presents a challenging scenario where there are more dogs with higher walking times,
and one dog dislikes all other dogs. It initializes a new problem instance and solves it to find a solution path.
Define dogs with varying attention and walking times
Initialize the initial state with dogs placed at site_a and site_b, walker position, and walker attributes
Define the goal state with dogs rearranged compared to the initial state and walker position
"""
dog11 = Dog(name="Dog11", attention=4, walking_time=9)
dog12 = Dog(name="Dog12", attention=5, walking_time=8)
dog13 = Dog(name="Dog13", attention=2, walking_time=15)
dog14 = Dog(name="Dog14", attention=3, walking_time=10)
dog15 = Dog(name="Dog15", attention=4, walking_time=7)
dog16 = Dog(name="Dog16", attention=2, walking_time=11)
dog10 = Dog(name="Dog10", attention=3, walking_time=12, dislikes=[dog11, dog12, dog13, dog14, dog15, dog16])
initial_state = DogWalkingState(site_a=[dog10], site_b=[dog11, dog12, dog13, dog14, dog15, dog16], walker_at_a=True, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog11, dog12, dog13, dog14, dog15, dog16], site_b=[dog10], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===Example 3===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)






"""
Example 4:
This example adds complexity by including dislikes among the dogs, making the problem more challenging to solve.
It initializes a new problem instance and solves it to find a solution path.
Define dogs with dislikes
Initialize the initial state with dogs placed at site_a and site_b, walker position, and walker attributes
Define the goal state with dogs rearranged compared to the initial state and walker position
"""
dog20 = Dog(name="Dog20", attention=2, walking_time=15)
dog18 = Dog(name="Dog18", attention=4, walking_time=9, dislikes=[dog20])
dog21 = Dog(name="Dog21", attention=3, walking_time=10, dislikes=[dog20, dog18])
dog17 = Dog(name="Dog17", attention=3, walking_time=12, dislikes=[dog21, dog20])
dog19 = Dog(name="Dog19", attention=5, walking_time=8, dislikes=[dog21, dog20])
initial_state = DogWalkingState(site_a=[dog17, dog18, dog19], site_b=[dog20, dog21], walker_at_a=True, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog20, dog21], site_b=[dog17, dog18, dog19], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===Example 4===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)






"""
Example 5:
This example involves rearranging dogs between two sites with careful consideration of their dislikes and interactions.
It initializes a new problem instance and solves it to find a solution path.
Define dogs with dislikes
Initialize the initial state with a more complex setup, including dislikes among the dogs
Define the goal state with dogs rearranged compared to the initial state and walker position
"""
dog22 = Dog(name="Dog22", attention=3, walking_time=13)
dog23 = Dog(name="Dog23", attention=4, walking_time=11, dislikes=[dog22])
dog24 = Dog(name="Dog24", attention=2, walking_time=14, dislikes=[dog23, dog22])
dog25 = Dog(name="Dog25", attention=5, walking_time=9, dislikes=[dog23, dog24])
dog26 = Dog(name="Dog26", attention=3, walking_time=12, dislikes=[dog24, dog25])
initial_state = DogWalkingState(site_a=[dog22, dog23, dog24, dog25], site_b=[dog26], walker_at_a=False, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog26], site_b=[dog22, dog23, dog24, dog25], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===Example 5===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)





"""
Example 6:
This example demonstrates a scenario where dogs have different preferences and dislikes towards each other.
It initializes a new problem instance and solves it to find a solution path.
Define dogs with dislikes
Initialize the initial state with dogs placed at site_a and site_b, walker position, and walker attributes
Define the goal state with dogs rearranged compared to the initial state and walker position
"""
dog27 = Dog(name="Dog27", attention=5, walking_time=10)
dog28 = Dog(name="Dog28", attention=4, walking_time=8, dislikes=[dog27])
dog29 = Dog(name="Dog29", attention=3, walking_time=12, dislikes=[dog27])
dog30 = Dog(name="Dog30", attention=3, walking_time=11, dislikes=[dog27, dog28, dog29])
dog31 = Dog(name="Dog31", attention=2, walking_time=13, dislikes=[dog27, dog29, dog30])
initial_state = DogWalkingState(site_a=[dog27, dog28, dog29], site_b=[dog30, dog31], walker_at_a=True, walker=initial_walker)
goal_state = DogWalkingState(site_a=[dog30, dog31], site_b=[dog27, dog28, dog29], walker=initial_walker)

# Create the problem instance and solve it
problem = DogWalkingProblem(initial_state, goal_state, initial_walker)
solution_path, nodes_explored, cost = problem.search()

# Print the problem and solution
print("===Example 6===")
print_solution(problem, goal_state, solution_path, nodes_explored, cost)
