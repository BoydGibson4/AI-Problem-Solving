"""
This interface defines the methods required on a problem state.
Any class that represents a problem state must implement this interface.
"""
class State:
    """Return all neighbouring states of the current one.
    These neighbours are obtained by changing each variable in the current state.
    Any problem state should have this method.
    :returns: All neighbouring states.
    :rtype: A list of State.
    """
    def neighbours(self):
        pass


"""Model a local search problem.
"""
class LocalSearchProblem:
    
    """This method performs a local search on the problem.
    It should be overridden in different sub-classes which implement specific local search algorithms.
    :param state: The starting state.
    :type state: A State.
    :returns: The solution state if found.
    """
    def search(self,state):
        pass
    
    """The objective function on a state.
    This method should be overridden in different sub-classes which defined problems in specific domains.
    :returns: The objective function value that measures how good a state is.:
    :rtype: A float value.
    """
    def objective(self,state):
        pass
