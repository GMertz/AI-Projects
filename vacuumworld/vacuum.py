import matplotlib
matplotlib.use("TkAgg")  # This seems necessary for the animation to work within PyCharm
import matplotlib.pyplot as plt
import numpy as np
import random

WORLD_WIDTH = 25  # Width, in squares, of the world

# These are used by apply to update the agent's location
OFFSETS = {'north': (-1, 0),
           'south': (1, 0),
           'east': (0, 1),
           'west': (0, -1)}

# These are used by animate and draw_world to draw the world in color
EMPTY = 0.0
DIRT = 0.33
OBSTACLE = 0.67
AGENT = 1.0
cmap = matplotlib.colors.ListedColormap(['white', 'orange', 'black', 'blue'])
norm = matplotlib.colors.BoundaryNorm([0.0, 0.25, 0.5, 0.75, 1.0], cmap.N)


def random_world():
    """Creates and returns a random world."""
    # Create empty world
    grid = np.zeros((WORLD_WIDTH, WORLD_WIDTH))
    # Add dirt and obstacles
    for r in range(WORLD_WIDTH):
        for c in range(WORLD_WIDTH):
            if random.random() < 0.5:
                grid[r, c] = DIRT
            elif random.random() < 0.1:
                grid[r, c] = OBSTACLE
    # Place agent
    while True:
        r = random.randrange(WORLD_WIDTH)
        c = random.randrange(WORLD_WIDTH)
        if grid[r, c] == EMPTY:
            return grid, r, c


def get_percept(grid, r, c):
    """Returns the percept for an agent at position r, c on grid: 'dirty' or 'clean'."""
    if grid[r, c] == DIRT:
        return 'dirty'
    else:
        return 'clean'


def draw_world(grid, r, c, image):
    """Updates image, showing grid with the agent at r, c."""
    under = grid[r, c]
    grid[r, c] = AGENT
    image.set_data(grid)
    grid[r, c] = under


def apply(grid, r, c, action):
    """Applies action ('suck', 'north', etc.) for an agent at position r, c on grid."""
    if action == 'suck':
        grid[r, c] = EMPTY
    else:
        new_r = r + OFFSETS[action][0]
        new_c = c + OFFSETS[action][1]
        if 0 <= new_r < WORLD_WIDTH and 0 <= new_c < WORLD_WIDTH and grid[new_r, new_c] != OBSTACLE:
            return new_r, new_c
    return r, c


def animate(agent, steps, initialize=None):
    """Animates an agent's performance in a random world for the specified number of steps. initialize is called
    once to provide additional parameters to the agent."""
    grid, r, c = random_world()
    image = plt.imshow(grid, cmap=cmap, norm=norm)
    if initialize:
        state = initialize()
    for t in range(steps):
        draw_world(grid, r, c, image)
        percept = get_percept(grid, r, c)
        if initialize:
            action, *state = agent(percept, *state)
        else:
            action = agent(percept)
        r, c = apply(grid, r, c, action)
        plt.pause(0.0001)
    plt.show()


def score(grid):
    """Returns the number of non-dirty squares in grid."""
    result = 0
    for r in range(WORLD_WIDTH):
        for c in range(WORLD_WIDTH):
            if grid[r, c] != DIRT:
                result += 1
    return result


def simulate(agent, steps, initialize=None):
    """Simulates an agent's performance in a random world for the specified number of steps. Returns the total score
    over this time. initialize is called once to provide additional parameters to the agent."""
    grid, r, c = random_world()
    if initialize:
        state = initialize()
    result = 0
    for t in range(steps):
        result += score(grid)
        percept = get_percept(grid, r, c)
        if initialize:
            action, *state = agent(percept, *state)
        else:
            action = agent(percept)
        r, c = apply(grid, r, c, action)
    return result


def experiment(agent, steps, runs, initialize=None):
    """Repeatedly simulates agent in runs random worlds for the specified number of steps each. Returns the average
    score across runs. initialize is called at the beginning of each run to provide additional parameters to the
    agent."""
    result = 0
    for r in range(runs):
        result += simulate(agent, steps, initialize)
    return result / runs


"""
INSTRUCTIONS:

You must define four functions here: reflex_agent, random_agent, state_agent, and init_state_agent

reflex_agent and random_agent each take a percept ('clean' or 'dirty') and return an action ('suck', 'north',
'south', 'east', or 'west').

state_agent takes a percept and any number of additional parameters recording the agent's state. It returns
an action plus updated versions of these parameters.

init_state_agent returns the original state parameters for state_agent.
"""
def reflex_agent(percept): #reflex agent, no random decision making, no memory
    if percept == 'dirty': 
        return('suck')
    return('south')


def random_agent(percept): #random agent, has no memory but is allowed to use randomness
    if percept == 'dirty':
        return 'suck'
    return(random.choice(['north','south','east','west']))


def full_random_agent(percept):#completely random agent
    return(random.choice(['suck','south','north','east','west'])) 



def state_agent(percept, state, cleanSquares, prev, prefDirs, moveCount):
                #state agent, remembers past percepts, is allowed to use randomness
    ''''
    Explanations for the state_agent arguments are as follows:
    percept - the input from the board, either 'clean' or 'dirty'
    state - the agent has 2 states, 0 and 1, 
        1 means it prioritizes random moves around itself(attempting to clean around itself)
        0 means it is prioritizing moving in a strightline(attempting to move to a new location)
    cleanSquares - how many consecutive clean squares has the agent seen
        this is used to guess when the agent may be stuck, and if agent is in an area that is already clean
    prev - the previous moves the agent has made
    prefDirs - the prefered directions the agent has used in state 0, this is used to make sure the agent 
        doesnt go back and forth over and over again
    moveCount - counts the amount of moves we have made since chaning states
    '''
    
    if percept == 'dirty': #if we see a dirty percept clean it and prioritize searching in the current area for more dirt
       
        cleanSquares = 0
        action = 'suck'

    elif percept == 'clean': 

        options = ['south','north','east','west']
        cleanSquares += 1
        if moveCount > 25 and not state: # after 25 moves of state 0, switch back to state 1 and reset moveCount
            moveCount = 0
            state = 1

        elif cleanSquares > 7 : #if we have seen 7 consecutive clean squares, switch to state 0
            
            options = ['south','north','east','west']
            state = 0
            moveCount = 0
            cleanSquares = 0
            tempdirs = list(set(options) - set(set(prefDirs))) #limits our options for prefered direction to new values
            if not tempdirs:#if there are no possible options(we have all of the options in prefDirs) shorten prefDirs
                prefDirs = prefDirs[2:]
                tempdirs = list(set(options) - set(set(prefDirs)))
            
            newprefdir = random.choice(tempdirs)
            prefDirs.append(newprefdir)

        if cleanSquares >= 5: #if agent has recieved 5 clean percepts in a row attempt to get unstuck by making a random move
           
            try:
                if len(prev[-1]) == 5: #exclude our previous move and its inverse, so we move around the obstacle
                    options.remove('north')
                    options.remove('south')
                else:
                    options.remove('east')
                    options.remove('west')
            except ValueError:
                pass

            if not state: #if in state 0, include our prefered direction, so we will continue to move through a potential white section
                for i in range(int(cleanSquares/2)):#we stack the odds to continue moving out of a white section
                    options.append(prefDirs[-1])

            action = random.choice(options)
            prev.append(action)

        else: #if we havent seen at least 3 consecutive clean squares, make a decision based on what state we are in
            if not state:#continue moving in the prefered direction
                action = prefDirs[-1]
            else:#make a random move, to clean the area around the agent
                action= random.choice(options)
                prev.append(action)
       
 
    return action, state, cleanSquares, prev, prefDirs, moveCount+1




def init_state_agent(): #initializes the state_agent
    state = 1
    cleanSquares = 0
    prev = ['east']
    prefDirs = ['east']
    moveCount = 0
    return state, cleanSquares, prev, prefDirs, moveCount



# Uncomment one of these to animate one of your agents
#animate(reflex_agent, 1000)
#animate(random_agent, 1000)
#animate(state_agent, 1000, init_state_agent)
# Uncomment these to run experiments comparing performance of diferent agents
# NOTE: This will take a while!
print('Reflex agent: ', experiment(reflex_agent, 10000, 20))
print('Random agent: ', experiment(random_agent, 10000, 20))
print('Full random agent: ', experiment(full_random_agent, 10000, 20))
print('State agent: ', experiment(state_agent, 10000, 20, init_state_agent))
