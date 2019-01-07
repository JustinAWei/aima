# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (Food) and Pacman position after moving (Pos).
        ScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        Pos = successorGameState.getPacmanPosition()
        Food = successorGameState.getFood().asList()
        Capsules = successorGameState.getCapsules()
        GhostStates = successorGameState.getGhostStates()
        GhostPositions = successorGameState.getGhostPositions()
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

        distanceFromNearestFood = min([util.manhattanDistance(Pos, food) for food in Food]) if Food else 0
        # distanceFromNearestCapsule = min([util.manhattanDistance(Pos, capsule) for capsule in Capsules]) if Capsules else 0
        distanceFromGhosts = min([util.manhattanDistance(Pos, ghost) for ghost in GhostPositions]) if GhostPositions else 5

        # Eating reward - try to eat food
        eatingReward = 1/(len(Food + Capsules)+1)

        # Food proximity reward - try get closer to food
        foodproximityReward = 1/(distanceFromNearestFood+1)

        # Ghost Avoidance - try to keep at least 2 distance from ghost
        ghostAvoidanceReward = distanceFromGhosts > 2
        scared = ScaredTimes[0]

        return 100*ghostAvoidanceReward + 1000*eatingReward + foodproximityReward + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        startingIndex = 0
        actions = gameState.getLegalActions(startingIndex)
        results = [(gameState.generateSuccessor(0, action), action) for action in actions]
        v, action = max([(self.value(state, startingIndex + 1, 0), action) for state, action in results])
        return action

    def value(self, state, index, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(state, index, depth)
        else:
            return self.minValue(state, index, depth)

    def maxValue(self, state, index, depth):
        v = -float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for successor in [state.generateSuccessor(index, action) for action in state.getLegalActions(index)]:
            v = max(v, self.value(successor, nextIndex, depth))
        return v

    def minValue(self, state, index, depth):
        v = float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for successor in [state.generateSuccessor(index, action) for action in state.getLegalActions(index)]:
            v = min(v, self.value(successor, nextIndex, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # return self.maxValue(gameState, startingIndex, 0, -float('inf'), float('inf'))
        state = gameState
        index = 0
        depth = 0
        a = -float('inf')
        b = float('inf')
        v = -float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for action in state.getLegalActions(index):
            successor = state.generateSuccessor(index, action)
            V = self.value(successor, nextIndex, depth, a, b)
            if v < V:
                v = V
                bestAction = action
            if v > b:
                return v
            a = max(a,v)
        return bestAction

    def value(self, state, index, depth, a, b):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(state, index, depth, a, b)
        else:
            return self.minValue(state, index, depth, a, b)

    def maxValue(self, state, index, depth, a, b):
        v = -float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for action in state.getLegalActions(index):
            successor = state.generateSuccessor(index, action)
            v = max(v, self.value(successor, nextIndex, depth, a, b))
            if v > b:
                return v
            a = max(a,v)
        return v

    def minValue(self, state, index, depth, a, b):
        v = float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for action in state.getLegalActions(index):
            successor = state.generateSuccessor(index, action)
            v = min(v, self.value(successor, nextIndex, depth, a, b))
            if v < a:
                return v
            b = min(b,v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        startingIndex = 0
        actions = gameState.getLegalActions(startingIndex)
        results = [(gameState.generateSuccessor(0, action), action) for action in actions]
        v, action = max([(self.value(state, startingIndex + 1, 0), action) for state, action in results])
        return action

    def value(self, state, index, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(state, index, depth)
        else:
            return self.expValue(state, index, depth)

    def maxValue(self, state, index, depth):
        v = -float('inf')
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for successor in [state.generateSuccessor(index, action) for action in state.getLegalActions(index)]:
            v = max(v, self.value(successor, nextIndex, depth))
        return v

    def expValue(self, state, index, depth):
        v = 0
        p = 1/len(state.getLegalActions(index))
        nextIndex = (index+1) % state.getNumAgents()
        if nextIndex == 0:
            depth+=1
        for successor in [state.generateSuccessor(index, action) for action in state.getLegalActions(index)]:
            v += p * self.value(successor, nextIndex, depth)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood().asList()
    Capsules = currentGameState.getCapsules()
    GhostStates = currentGameState.getGhostStates()
    GhostPositions = currentGameState.getGhostPositions()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    distanceFromNearestFood = min([util.manhattanDistance(Pos, food) for food in Food+Capsules]) if Food else 0
    # distanceFromNearestCapsule = min([util.manhattanDistance(Pos, capsule) for capsule in Capsules]) if Capsules else 0
    distanceFromGhosts = min([util.manhattanDistance(Pos, ghost) for ghost in GhostPositions]) if GhostPositions else 5

    # Eating Score - try to eat food
    eatingScore = 1/(len(Food + Capsules)+1)

    # Food proximity Score - try get closer to food
    foodproximityScore = 1/(distanceFromNearestFood+1)

    # Ghost Avoidance - try to keep at least 2 distance from ghost
    scared = ScaredTimes[0]
    ghostScore = distanceFromGhosts if not scared else 1/distanceFromGhosts

    score = 100*currentGameState.getScore() + scared*1000*ghostScore + 1000*eatingScore + 10*foodproximityScore
    # print(score)
    return score
# Abbreviation
better = betterEvaluationFunction
