# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for _ in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if state == 'TERMINAL_STATE':
                    newValues['TERMINAL_STATE'] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                newValues[state] = max([self.computeQValueFromValues(state, action) for action in actions])
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        Q = 0
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in statesAndProbs:
            reward = self.mdp.getReward(state, action, nextState)
            Q += prob*(reward + self.discount * self.values[nextState])
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if state == 'TERMINAL_STATE':
            return None
        actions = self.mdp.getPossibleActions(state)
        _, bestAction = max([(self.computeQValueFromValues(state, action), action) for action in actions])
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        i = 0
        while i < self.iterations:
            for state in self.mdp.getStates():
                if i >= self.iterations:
                    break
                i+=1
                if state == 'TERMINAL_STATE':
                    self.values['TERMINAL_STATE'] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in actions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors of all states.
        predecessors = {}
        states = [s for s in self.mdp.getStates() if s != 'TERMINAL_STATE']
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitionStates = [s for s,_ in self.mdp.getTransitionStatesAndProbs(state, action)]
                for nextState in transitionStates:
                    if nextState not in predecessors:
                        predecessors[nextState] = set()
                    predecessors[nextState].add(state)

        # Initialize an empty priority queue.
        pq = util.PriorityQueue()

        # For each non-terminal state s, do:
        for s in states:
            # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s
            actions = self.mdp.getPossibleActions(s)
            maxQ = max([self.computeQValueFromValues(s, action) for action in actions])
            diff = abs(self.values[s] - maxQ)

            # Push s into the priority queue with priority -diff
            pq.push(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for _ in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break

            # Pop a state s off the priority queue.
            s = pq.pop()

            # Update s's value (if it is not a terminal state) in self.values.
            actions = self.mdp.getPossibleActions(s)
            self.values[s] = max([self.computeQValueFromValues(s, action) for action in actions])

            # For each predecessor p of s, do:
            for p in predecessors[s]:
                # Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff.
                actions = self.mdp.getPossibleActions(p)
                maxQ = max([self.computeQValueFromValues(p, action) for action in actions])
                diff = abs(self.values[p] - maxQ)

                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority.
                if diff > self.theta:
                    pq.update(p, -diff)
