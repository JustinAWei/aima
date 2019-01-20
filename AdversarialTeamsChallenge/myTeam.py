# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.depth = 1
        self.numStartingFood = len(self.getFood(gameState).asList())

    def chooseAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        startingIndex = self.index
        actions = gameState.getLegalActions(startingIndex)
        results = [(self.getSuccessor(gameState,action), action) for action in actions]
        v, action = max([(self.value(state, startingIndex + 1, 0), action) for state, action in results])
        return action

    def value(self, state, index, depth):
        # print(state, index, depth)
        if depth == self.depth:
            return self.evaluate(state)
        elif index in self.getTeam(state):
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

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def evaluate(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        features['successorScore'] = self.getScore(gameState)
        return features

    def getWeights(self, gameState):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState):
    features = util.Counter()
    successor = gameState
    foodList = self.getFood(successor).asList()
    pos = successor.getAgentPosition(self.index)
    capsules = self.getCapsules(successor)
    opponentIndices = self.getOpponents(successor)
    opponentPositions = [gameState.getAgentPosition(i) for i in opponentIndices]
    opponentStates = [gameState.getAgentState(i) for i in opponentIndices]
    scaredTimes = [state.scaredTimer for state in opponentStates]

    numCurrentFood = len(foodList)
    _, nearestOwnFood = min([(self.getMazeDistance(pos, food),food) for food in self.getFoodYouAreDefending(gameState).asList()])
    eatenFood = self.numStartingFood - numCurrentFood
    distanceToHome = self.getMazeDistance(pos, nearestOwnFood)
    features['goHomeScore'] = eatenFood * distanceToHome
    features['nearestOpponent'] = min([self.getMazeDistance(pos, opponentPos) for opponentPos in opponentPositions]) > 2
    features['eatingScore'] = len(foodList + capsules)
    features['capsuleScore'] = (len(capsules) + 1)
    features['foodScore'] = len(foodList)
    features['distanceToFood'] = min([self.getMazeDistance(pos, food) for food in foodList])
    features['nearestVictim'] = (features['nearestOpponent'] + 1) if scaredTimes[0] else 0
    # print(features)
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'goHomeScore':-100, 'successorScore': 100, 'nearestOpponent': 10, 'eatingScore': -1000, 'distanceToFood': -10, 'nearestVictim': -100, 'capsuleScore':-100, 'foodScore':-1}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState):
    features = util.Counter()
    successor = gameState

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies]# if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    # if action == Directions.STOP: features['stop'] = 1
    # rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    #if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
