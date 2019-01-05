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
        numPacmen = gameState.getNumPacmanAgents()

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

    def findPathToClosestDotInSpace(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)
        width = gameState.getWidth()
        numPacmen = gameState.getNumPacmanAgents()

        spaceStart = int(width*(self.index) / numPacmen)
        spaceEnd = int(width*(self.index+1) / numPacmen)
        space = range(spaceStart, spaceEnd)

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
            if problem.isGoalState(node) and node[0] in space:
                return actions
            for successor, action, cost in problem.getSuccessors(node):
                frontier.push((successor, actions + [action]))
        return []

    def findPathToSpace(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        initial = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)
        width = gameState.getWidth()
        numPacmen = gameState.getNumPacmanAgents()

        spaceStart = int(width*(self.index) / numPacmen)
        spaceEnd = int(width*(self.index+1) / numPacmen)
        space = range(spaceStart, spaceEnd)

        explored = []
        actions = []
        frontier = util.Queue()

        frontier.push((initial, actions))
        x,y = initial
        if abs(x-spaceStart) > abs(x-spaceEnd):
            closestSpaceNode = (spaceEnd, y)
        else:
            closestSpaceNode = (spaceStart, y)

        dist = util.manhattanDistance(initial, closestSpaceNode)

        while not frontier.isEmpty():
            node, actions = frontier.pop()
            # print(dist, util.manhattanDistance(initial, node), util.manhattanDistance(node, closestSpaceNode))
            # print(node)
            if node in explored:
                continue
            explored.append(node)
            # if problem.isGoalState(node) and dist >= util.manhattanDistance(initial, node) and dist >= util.manhattanDistance(node, closestSpaceNode):
            if node[0] in space:
                print(actions)
                return actions
            for successor, action, cost in problem.getSuccessors(node):
                frontier.push((successor, actions + [action]))
        return []

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """
        startPosition = state.getPacmanPosition(self.index)
        food = state.getFood()
        walls = state.getWalls()
        problem = AnyFoodSearchProblem(state, self.index)
        width = state.getWidth()
        numPacmen = state.getNumPacmanAgents()
        space = range(int(width*(self.index) / numPacmen), int(width*(self.index+1) / numPacmen))

        if self.finished:
            print(self.index, 'finished')
            return self.findPathToClosestDot(state)[0]

        elif startPosition[0] in space:
            print(self.index, True)
            if self.pathToSpaceDot and self.pathToSpaceDotIndex < len(self.pathToSpaceDot)-1:
                self.pathToSpaceDotIndex+=1
                return self.pathToSpaceDot[self.pathToSpaceDotIndex]
            else:
                self.pathToSpaceDot = self.findPathToClosestDotInSpace(state)
                if(self.pathToSpaceDot):
                    self.pathToSpaceDotIndex = 0
                    return self.pathToSpaceDot[self.pathToSpaceDotIndex]
                else:
                    self.finished = True
                    return self.findPathToClosestDot(state)[0]
        else:
            # print(self.index, False)
            if self.pathToSpace:
                self.pathToSpaceIndex+=1
                return self.pathToSpace[self.pathToSpaceIndex]
            else:
                self.pathToSpaceIndex = 0
                self.pathToSpace = self.findPathToSpace(state)
                return self.pathToSpace[self.pathToSpaceIndex]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """
        self.pathToSpace = []
        self.pathToSpaceDot = []
        self.pathToDot = []

        self.finished = False

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
    different goal test, which you need to fill in below.  The state space and
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
