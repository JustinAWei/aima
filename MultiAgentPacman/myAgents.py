# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    def findPathToClosestDot(self, gameState, sector=None):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """

        startPosition = gameState.getPacmanPosition(self.index)
        problem = AnyFoodSearchProblem(gameState, self.index)
        foodList = gameState.getFood().asList()

        # Find closest food in sector
        searchSpace = [food for food in foodList if food[0] in sector] if sector else foodList

        if searchSpace:
            _, closestDot = min([(util.manhattanDistance(startPosition, food), food) for food in searchSpace])
        else:
            return False

        # A* Search for path to specified food using Manhattan Heuristic
        heuristic = util.manhattanDistance
        explored = []
        frontier = util.PriorityQueue()
        frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), closestDot))
        while(not frontier.isEmpty()):
            currentNode, actions = frontier.pop()
            if(currentNode in explored):
                continue
            explored.append(currentNode)
            if(currentNode == closestDot):
                return actions
            for successor, action, successorCost in problem.getSuccessors(currentNode):
                newActions = actions + [action]
                frontier.push((successor, newActions), problem.getCostOfActions(newActions) + heuristic(successor, closestDot))
        return []

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """
        width = state.getWidth()
        numPacmen = state.getNumPacmanAgents()

        # Give each agent a sector of the map
        sectorStart = int(width*(self.index) / numPacmen)
        sectorEnd = int(width*(self.index + 1) / numPacmen)
        sector = range(sectorStart, sectorEnd)

        # Generate path
        if not self.path:
            self.path = self.findPathToClosestDot(state) if self.sectorClean else self.findPathToClosestDot(state, sector)
        # Agent has cached path, execute it
        if self.path:
            return self.path.pop(0)
        else:
            self.sectorClean = True
            return self.getAction(state)
        return 'Stop'


    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """
        self.path = []
        self.sectorClean = False

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        explored = []
        actions = []
        initial = problem.getStartState()
        frontier = util.Queue()

        frontier.push((initial, actions))

        while not frontier.isEmpty():
            node, actions = frontier.pop()
            if node in explored:
                continue
            explored.append(node)
            if problem.isGoalState(node):
                return actions
            for successor, action, cost in problem.getSuccessors(node):
                frontier.push((successor, actions + [action]))

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state sector and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        return state in self.food.asList()
