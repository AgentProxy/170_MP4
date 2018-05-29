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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print (newScaredTimes)
        val = successorGameState.getScore()

        # The ghost's threat value drops significantly if they're scared, but they remain a threat nonetheless.
        # Thus, when the ghosts are scared, food should be gathered more.
        scared = min(newScaredTimes) < 1
        ghost_weight = (10 * scared) + 5
        food_weight = 10

        distanceGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        try:
            if min(distanceGhost) > 0:  # We're only concerned with the nearest ghost as it is the immediate danger to pacman.
                val -= ghost_weight / min(distanceGhost)
        except ValueError:
            print("VALUE ERROR")

        distanceFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        try:
            if min(distanceFood) > 0:  # As long as food remains, pacman must seek them while avoiding the ghosts.
                val += food_weight / min(distanceFood)
        except ValueError:
            print("VALUE ERROR")

        return val

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
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action, score = self.value(gameState, currentAgentIndex, currentDepth)
        # print action
        return action

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):

      if currentAgentIndex >= gameState.getNumAgents():# when current Agent index is more than num of agents is becomes pacman's index and the depth increases
          currentAgentIndex = 0
          currentDepth += 1

      if  currentDepth == self.depth or gameState.isWin() or gameState.isLose(): #when we are starting to hit terminal states like reaching max depth.
          return self.evaluationFunction(gameState)

      elif currentAgentIndex == 0:          #get max value when index of agent is pacman
          return self.max_value(gameState, currentAgentIndex, currentDepth)

      else:                                 #get min value when index of agent is ghosts
          return self.min_value(gameState, currentAgentIndex, currentDepth)
      # check whether currentAgentIndex is our pacman agent or ghost agent
      # if our agent: return max_value(....)
      # otherwise: return min_value(....)

    def max_value(self, gameState, currentAgentIndex, currentDepth):

        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_value = float('-inf')
        current_action = "None"
        ret_pair = [current_action, current_value] #pair to be returned

        for action in legalActions:#loops through all legal actions available to current agent

            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth) #gets the value of the succesor of the other agent

            if type(val) is list: # since i have to return pairs and this is in recursion it is needed to help get the value of the action
                check_val = val[1]
            else:
                check_val = val

            if check_val > current_value: #if the checking value is larger then that will be the updated current value and the value of the action's score
                ret_pair = [action, check_val]
                current_value = check_val

        return ret_pair     #retruns the list pair

      # loop over each action available to current agent:
      # (hint: use gameState.getLegalActions(...) for this)
      #     use gameState.generateSuccessor to get nextGameState from action
      #     compute value of nextGameState by calling self.value
      #     compare value of nextGameState and current_value
      #     keep whichever value is smaller, and take note of the action too

    def min_value(self, gameState, currentAgentIndex, currentDepth):

        current_value = float('inf')
        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_action = "None"
        ret_pair = [current_action, current_value]#pair to be returned

        for action in legalActions: #loops through all legal actions available to current agent

            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth) #gets the value of the succesor of the other agent

            if type(val) is list: # since i have to return pairs and this is in recursion it is needed to help get the value of the action
                check_val = val[1]
            else:
                check_val = val

            if check_val < current_value:  #if the checking value is smaller then that will be the updated current value and the value of the action's score
                ret_pair = [action, check_val]
                current_value = check_val

        return ret_pair # returns the list pair

        # loop over each action available to current agent:
        # (hint: use gameState.getLegalActions(...) for this)
        #      use gameState.generateSuccessor to get nextGameState from action
        #     compute value of nextGameState by calling self.value
        #     compare value of nextGameState and current_value
        #     keep whichever value is smaller, and take note of the action too


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        alpha = float('inf') * -1
        beta = float('inf')
        action,score = self.value(gameState, currentAgentIndex, currentDepth, alpha, beta)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        if currentAgentIndex >= gameState.getNumAgents():# when current Agent index is more than num of agents is becomes pacman's index and the depth increases
            currentAgentIndex = 0
            currentDepth += 1

        if  currentDepth == self.depth or gameState.isWin() or gameState.isLose(): #when we are starting to hit terminal states like reaching max depth.
            return self.evaluationFunction(gameState)

        elif currentAgentIndex == 0:          #get max value when index of agent is pacman
            return self.max_value(gameState, currentAgentIndex, currentDepth, alpha, beta)

        else:                                 #get min value when index of agent is ghosts
            return self.min_value(gameState, currentAgentIndex, currentDepth, alpha, beta)

    def max_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_value = float('inf') * -1
        current_action = "None"
        ret_pair = [current_action, current_value]  #pair to be returned
        for action in legalActions:                 #loops through all legal actions available to current agent
            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth, alpha, beta)
            if type(val) is list:                   #Return the value. If it is a list where it returns a direction and value, return the value only hence [1].
                check_val = val[1]              
            else:
                check_val = val

            if check_val > current_value:           #if the checking value is larger then that will be the updated current value and the value of the action's score
                ret_pair = [action, check_val]
                current_value = check_val

            if check_val > beta:                    #if check_val is greater than beta return action and value. This is where pruning happens because it disregards other branches
                return [action, check_val]
            alpha = max(alpha, check_val)           #if not, update the alpha to the maximum of the current beta and the new value
        return ret_pair

    # Similar to MinimaxAgent's max_value() method
    # Include checking if current_value is worse than beta
    #   if so, immediately return current (action,current_value) tuple
    # Include updating of alpha
    # Note: always returns (action,score) pair

    def min_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_value = float('inf')
        current_action = "None"
        ret_pair = [current_action, current_value]  #pair to be returned
        for action in legalActions:                 #loops through all legal actions available to current agent
            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth, alpha, beta)
            if type(val) is list:                   #Return the value. If it is a list where it returns a direction and value, return the value only hence [1].
                check_val = val[1]
            else:
                check_val = val

            if check_val < current_value:           #if the checking value is smaller then that will be the updated current value and the value of the action's score
                ret_pair = [action, check_val]
                current_value = check_val

            if check_val < alpha:                   #if check_val is less than alpha return action and value. This is where pruning happens because it disregards other branches
                return [action, check_val]
            beta = min(beta, check_val)             #if not, update the beta to the minimum of the current beta and the new value
        return ret_pair

      # Similar to MinimaxAgent's min_value() method
      # Include checking if current_value is worse than alpha
      #   if so, immediately return current (action,current_value) tuple
      # Include updating of beta

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
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action,score = self.value(gameState, currentAgentIndex, currentDepth)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):
      if currentAgentIndex >= gameState.getNumAgents():# when current Agent index is more than num of agents is becomes pacman's index and the depth increases
          currentAgentIndex = 0
          currentDepth += 1

      if  currentDepth == self.depth or gameState.isWin() or gameState.isLose(): #when we are starting to hit terminal states like reaching max depth.
          return self.evaluationFunction(gameState)

      elif currentAgentIndex == 0:          #get max value when index of agent is pacman
          return self.max_value(gameState, currentAgentIndex, currentDepth)

      else:                                 #get min value when index of agent is ghosts
          return self.exp_value(gameState, currentAgentIndex, currentDepth)

      # More or less the same with MinimaxAgent's value() method
      # Only difference: use exp_value instead of min_value

    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth):
        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_value = float('-inf')
        current_action = "None"
        ret_pair = [current_action, current_value] #pair to be returned

        for action in legalActions:     #loops through all legal actions available to current agent
            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth)   #gets the value of the succesor of the other agent

            if type(val) is list:       #Return the value. If it is a list where it returns a direction and value, return the value only hence [1].
                check_val = val[1]
            else:
                check_val = val
            if check_val > current_value:    #if the checking value is larger then that will be the updated current value and the value of the action's score
                ret_pair = [action, check_val]
                current_value = check_val
        return ret_pair     #returns the list pair
      # Exactly like MinimaxAgent's max_value() method

    # Note: always returns (action,score) pair
    def exp_value(self, gameState, currentAgentIndex, currentDepth):
        legalActions = gameState.getLegalActions(currentAgentIndex)
        current_value = 0
        current_action = "None"
        prob = 1.0              #Assigned a default value
        if len(legalActions)!=0:                   #To prevent division by 0
                prob = 1.0/float(len(legalActions))     #Get the probability by dividing the probability of an action over the number of actions
    
        for action in legalActions:      #loops through all legal actions available to current agent
            val = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth) #gets the value of the succesor of the other agent
            if type(val) is list:       #Return the value. If it is a list where it returns a direction and value, return the value only hence [1].
                check_val = val[1]
            else:
                check_val = val
            current_value += prob * check_val   #The expected valus is the average, weighted by the probability distribution over outcomes
        return [current_action,current_value]    #returns the list pair

      # use gameState.getLegalActions(...) to get list of actions
      # assume uniform probability of possible actions
      # compute probabilities of each action
      # be careful with division by zero
      # Compute the total expected value by:
      #   checking all actions
      #   for each action, compute the score the nextGameState will get
      #   multiply score by probability
      # Return (None,total_expected_value) 
      # None action --> we only need to compute exp_value but since the 
      # signature return values of these functions are (action,score), we will return an empty action
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
	  # Food is more important if Ghosts are scared, otherwise it is more important to avoid Ghosts first.
    """
    "*** YOUR CODE HERE ***"
    val = currentGameState.getScore()
    # Similar to Q1, only this time there's only one state (no nextGameState to compare it to)
    # Use similar features here: position, food, ghosts, scared ghosts, distances, etc.
    # Can use manhattanDistance() function
    # You can add weights to these features
    # Update the score variable (add / subtract), depending on the features and their weights
    # Note: Edit the Description in the string above to describe what you did here

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
	
    scared = min(currentScaredTimes) < 1
    ghost_weight = (10 * scared) + 4
    food_weight = 5

    distanceGhost = [manhattanDistance(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
    try:
		if min(distanceGhost) > 0:  # We're only concerned with the nearest ghost as it is the immediate danger to pacman.
			val -= ghost_weight / min(distanceGhost)
    except ValueError:
		print("VALUE ERROR: GHOST EMPTY")

    distanceFood = [manhattanDistance(currentPos, food) for food in currentFood.asList()]
    try:
		if min(distanceFood) > 0:  # As long as food remains, pacman must seek them while avoiding the ghosts.
			val += food_weight / min(distanceFood)
    except ValueError:
		print("VALUE ERROR: FOOD EMPTY")
	
    return val

# Abbreviation
better = betterEvaluationFunction

