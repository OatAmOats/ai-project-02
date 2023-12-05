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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #gameState containing all of the following:
        newPos = successorGameState.getPacmanPosition() #(x,y)
        newFood = successorGameState.getFood() #big grid of true and false as always
        newGhostStates = successorGameState.getGhostStates() #list(?) of ghosts, each including: (x,y) and direction they're moving in
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #list containing the times the ghost(s) will be scared for
        "*** YOUR CODE HERE ***"
        ghostPositions = successorGameState.getGhostPositions() #list of positions of ghosts
        defaultScore = successorGameState.getScore() #natural game score at the next state

        #distance to closest food:
        newFoodList = newFood.asList() #get food positions as a list
        foodDistance = float('inf') #initilialize food distance 
        for food in newFood.asList(): #find closest manhattan distance to a food
            tempFoodDistance = abs(newPos[0]- food[0]) + abs(newPos[1] - food[1])
            foodDistance = min(foodDistance, tempFoodDistance)

        ghostDistances = [] #initialize array to hold distances to ghosts
        for pos in ghostPositions: #for each ghost position
            ghostDistances.append(abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])) #add manhattan distance to distance array
        stopStopper = 1 #initialize stopStopper
        if action == "STOP" or action=="stop" or action=="Stop": #if the action is stop, punish pacman
            stopStopper = -100
        mmmFood = 0 #initialize mmmFood
        if newPos in currentGameState.getFood().asList(): #if the next location contains a food pellet, reward pacman
            mmmFood = 100
        if newPos in currentGameState.getCapsules(): #same happens if it's a power pellet
            mmmFood = 100
        #yucky combination of all the values calculated. (default score, distance to closeset ghost, distance to closest food, 
        #stop stopper and whether or not there's a food pellet in the next position)
        return ((defaultScore+10*min(ghostDistances))/(5*(foodDistance))) + stopStopper + mmmFood
        return successorGameState.getScore()

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
        "*** YOUR CODE HERE ***"
        
        best = '' #initilialize best action 
        highest = -float('inf') #and highest value
        for action in gameState.getLegalActions(0): #for each action
            blehh = minimax(gameState.generateSuccessor(0,action), 1, self.depth, self.evaluationFunction) #get the minimax value
            highest = max(highest,blehh) #update highest
            if highest == blehh: #if highest was updated, update best action
                best = action
        return best #return the best action according to minimax
        util.raiseNotDefined()

def minimax(gameState, agentIndex, depth, eval):
    #minimax returns utility, and i need to pick the action with the highest minimax utility?
    #depth check
    if depth == 0:
        return eval(gameState)
    #game over checks
    if gameState.isWin():
        return eval(gameState)
    if gameState.isLose():
        return eval(gameState)

    #player == MAX
    if agentIndex == 0:
        highest = -float('inf') #initialize highest as - infinity
        for action in gameState.getLegalActions(agentIndex): #for each action MAX can take:
            nextState = gameState.generateSuccessor(agentIndex, action) #generate new state for that action
            actionCost = minimax(nextState, agentIndex + 1, depth, eval) #calculate action cost with minimax
            highest = max(actionCost, highest) #update highest, to be the highest value out of all the actions
        return highest
    
    #player == MIN
    if agentIndex != 0:
        lowest = float('inf') #initialize lowerst as infinity
        if agentIndex != gameState.getNumAgents() - 1: #if the agent isn't last in order:
            for action in gameState.getLegalActions(agentIndex): #for each action
                nextState = gameState.generateSuccessor(agentIndex, action) #generate the next state
                actionCost = minimax(nextState, agentIndex + 1, depth, eval) #and get minimax of the next agent's actions
                lowest = min(actionCost, lowest) #and update lowest
        else: #if the agent is last in order:
            for action in gameState.getLegalActions(agentIndex): #for each action
                nextState = gameState.generateSuccessor(agentIndex, action) #get the successor state
                actionCost = minimax(nextState, 0, depth - 1, eval) #and run minimax on that state for pacman, updating the depth
                lowest = min(actionCost, lowest) #then update lowest
        return lowest
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, agentIndex, depth, alpha, beta): #max value portion
        #check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        #initialize v and best action
        v = -float('inf')
        yayAction = None
        #for each action pacman can take:
        for action in gameState.getLegalActions(agentIndex):
            #new state is the resulting state from pacman taking that action
            newState = gameState.generateSuccessor(agentIndex, action)
            #and tempVal is the minValue of that newState for the next agent
            tempVal = self.minValue(newState, 1 ,depth, alpha, beta)[0]
            #if tempVal is greater than v, update v and yayAction
            if tempVal > v:
                v, yayAction = tempVal, action
            # if v is greater than beta, prune the rest and return the current (v, yayAction)
            if v > beta:
                return (v, yayAction)
            #update Alpha
            alpha = max(v, alpha)
        #return value
        return (v, yayAction)
        
    def minValue(self, gameState, agentIndex, depth, alpha, beta): #min Value portion
        #check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        #initialize v and yayAction
        v = float('inf')
        yayAction = None
        #for each action the agent can take:
        for action in gameState.getLegalActions(agentIndex):
            newState = gameState.generateSuccessor(agentIndex, action) #new state after agent takes said action
            #if the agent is last in order, get maxValue for pacman's actions, and increment the depth
            if agentIndex == gameState.getNumAgents() - 1:
                tempVal = self.maxValue(newState, 0, depth + 1, alpha, beta)[0]
            #if the agent isn't last, get the min value for the next ghost/adversary's actions
            else:
                tempVal = self.minValue(newState, agentIndex + 1, depth, alpha, beta)[0]
            #if the value is the new min, update v and yayAction
            if tempVal < v:
                v, yayAction = tempVal, action
            #if v is less than alpha, prune and return current (v, yayAction)
            if v < alpha:
                return (v, yayAction)
            #update beta
            beta = min(v, beta)
        return (v, yayAction)
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        
        #return the action from pacman's maxValue at gameState with alpha = -(infinity), beta = infinity
        return self.maxValue(gameState, 0,0, -float('inf'), float('inf'))[1]
        util.raiseNotDefined()

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
        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose(): #check for terminal state
                return (self.evaluationFunction(gameState), None)
            if depth == self.depth: #check for depth reached
                return (self.evaluationFunction(gameState), None)
            if agentIndex == 0: #if agent is pacman:
                yayAction = None #initialize action
                v = -float('inf') #initialize value
                for action in gameState.getLegalActions(agentIndex): #for each action he can take
                    uhh = expectimax(gameState.generateSuccessor(agentIndex, action), depth, 1)[0] #expectimax value for that action, 0th index is value
                    if uhh > v: #if uhh is the current best value:
                        v, yayAction = uhh, action #update v and yayAction
                return (v, yayAction) #return v and yayAction as a tuple
            else: #if agent is a ghost
                yayAction = None #initialize action
                avg = 0 #initialize average

                for action in gameState.getLegalActions(agentIndex): #for each action the agent can take
                    if agentIndex != gameState.getNumAgents() - 1 : #if it isn't the last agent:
                        #add expectimax value for next agent to avg
                        avg += expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0]
                        yayAction = action #and update yayAction (just a filler, we don't care about the action here)
                    else: #if it is the last agent:
                        #add expectimax value for pacman, incrementing the depth, to avg
                        avg += expectimax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)[0] 
                        yayAction = action #again, we don't care about the action here
                avg = (float(avg))/float(len(gameState.getLegalActions(agentIndex))) #divide avg by number of actions to get actual average
                return (avg, yayAction) #return average and dummy action as a tuple
            
        return expectimax(gameState, 0, 0)[1] #return expectimax for pacman with a starting depth of zero. the [1] index returns the best action.
        util.raiseNotDefined()
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Takes the actual game score at that state (realGameScore), and adds 5 times the sum of the ghosts' manhattan distances to pacman.
    Then it divides that by (whatever the highest scary timer is) + 6 * (the sum of the manhattan distances to all the foods) + 2* (how many food pellets remain)
    + 30 (calibration number)
    any constant is sort of just there for calibration purposes. i got them by messing around with numbers until it worked.
    """
    "*** YOUR CODE HERE ***"
    realGameScore = currentGameState.getScore() #actual score in game
    foodList = currentGameState.getFood().asList()
    ghostPositions = currentGameState.getGhostPositions()
    #ok so now i'm working with the state, not the action. so i can just ball out, ya know
    sumGhosts = 0 #sum of ghosts distances
    sumFood = 0 #sum of distance to all foods
    GhostStates = currentGameState.getGhostStates() #list(?) of ghosts, each including: (x,y) and direction they're moving in

    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates] #list of how long each ghost is scared for
    oooohScary = max(newScaredTimes) #ooooohScary is the longest time a ghost is scared for
    for food in foodList: #for each food
        distance = manhattanDistance(currentGameState.getPacmanPosition(), food) #find distance to food
        sumFood += distance #and add the distance to that food to sumFood
    for ghost in ghostPositions: #for each ghost
        distance = manhattanDistance(currentGameState.getPacmanPosition(), ghost) #find distance
        sumGhosts += distance  
    return ((realGameScore + 5*sumGhosts))/(oooohScary + (6*(sumFood) + 2*len(foodList) + 30)) #return evaluation function
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
