import search
import random
from statistics import mean

solves = 0
trials = 0


class EightPuzzle(search.Problem):
    """Traditional sliding-tile puzzle. A state is represented as a tuple of characters, '_' and '1' through '8'.
    The first three characters are the top row, the next three the middle row, and the last three the bottom row.
    An action is represented as the index of the position where the blank is being moved."""

    def __init__(self, n):
        """The initial state is formed by making n random moves from the goal state. Note that the shortest distance to
        the goal may be less than n because some random moves 'cancel out' others."""
        self.initial = tuple('_12345678')
        for i in range(n):
            action = random.choice(self.actions(self.initial))
            self.initial = self.result(self.initial, action)

    def actions(self, state): #TODO
        """Returns the list of actions available from state.
        returns the following list comprehension:

        index of sq, for each sq in the given state
        if the distance between that square and the blank square ('_') 
        is 3 (adjacent vertically), or 1, -1 (adjacent horizontally)
        we have specific conditions for horizontal adjacents to prevent
        illegal moves on the edge of the board.
        """
        blank = state.index('_')
        return [state.index(sq) for sq in state if abs(state.index(sq)-blank) == 3 \
        or (state.index(sq)-blank == -1 and blank % 3 != 0) or \
        (state.index(sq)-blank == 1 and blank % 3 != 2)]
        

    def goal_test(self, state):
        """Returns true if state corresponds to _12345678."""
        if state == tuple('_12345678'):
            solves+=1

        return state == tuple('_12345678')

    def result(self, state, action): #TODO
        """Returns the state resulting from taking action in state.
        does a simple swap between the blank and given action index 
        using a temporary variable (temp)
        """
        new = list(state)
        temp = state[action]
        new[state.index('_')] = temp
        new[action] = '_'
        return tuple(new)
        


def prettify(state):
    """Returns a more human-readable grid representing state."""
    result = ''
    for i, tile in enumerate(state):
        result += tile
        if i % 3 == 2:
            result += '\n'
    return result


def misplaced(node):
    """8-puzzle heuristic returning the number of mismatched tiles.
    uses a list comprehension to create a list of all squares that 
    have a value not equal to their index (ignoring the blank space). 
    we then return the length of this list
    """
    return len([i for i in node.state if i != "_" and int(i) != node.state.index(i)])



def manhattan(node):
    """8-puzzle heuristic returning the sum of Manhattan distance between tiles and their correct locations.
    uses a list comprehension to find the manhattan distance from each square to its coresponding "goal space"
    for each square (ignoring the blank space) in the state, we take the absolute value of the difference 
    between the given square's index and value and take the following sum: 
    difference // 3 + difference % 3 
    the first term accounts for the square's vertical difference and the second term 
    its horizontal difference we then return the sum of this list
    """
    return sum([ (abs(node.state.index(i)-int(i))//3 + abs(node.state.index(i)-int(i))%3) \
    for i in node.state if i != '_'])



if __name__ == '__main__':
    # TODO This should be unchanged in the final program you hand in, but it might be useful to make a copy,
    # comment out one copy, and modify the other to get things to run more quickly while you're debugging
    depths = (1, 2, 4, 8, 16)
    trials = 100
    path_lengths = {}
    state_counts = {}
    for depth in depths:
        print('Gathering data for depth ' + str(depth) + '...')
        path_lengths[depth] = {'BFS':[], 'IDS':[], 'A*-mis':[], 'A*-Man':[]}
        state_counts[depth] = {'BFS':[], 'IDS':[], 'A*-mis':[], 'A*-Man':[]}
        for trial in range(trials):
            puzzle = EightPuzzle(depth)
            p = search.InstrumentedProblem(puzzle)
            path_lengths[depth]['BFS'].append(len(search.breadth_first_search(p).path()))
            state_counts[depth]['BFS'].append(p.states)
            p = search.InstrumentedProblem(puzzle)
            path_lengths[depth]['IDS'].append(len(search.iterative_deepening_search(p).path()))
            state_counts[depth]['IDS'].append(p.states)
            p = search.InstrumentedProblem(puzzle)
            path_lengths[depth]['A*-mis'].append(len(search.astar_search(p, misplaced).path()))
            state_counts[depth]['A*-mis'].append(p.states)
            p = search.InstrumentedProblem(puzzle)
            path_lengths[depth]['A*-Man'].append(len(search.astar_search(p, manhattan).path()))
            state_counts[depth]['A*-Man'].append(p.states)
    print('Path lengths:')
    print('{:>5}  {:>8}  {:>8}  {:>8}  {:>8}'.format('Depth', 'BFS', 'IDS', 'A*-mis', 'A*-Man'))
    for depth in depths:
        print('{:>5}  {:>8}  {:>8}  {:>8}  {:>8}' \
              .format(depth,
                      mean(path_lengths[depth]['BFS']),
                      mean(path_lengths[depth]['IDS']),
                      mean(path_lengths[depth]['A*-mis']),
                      mean(path_lengths[depth]['A*-Man'])))
    print('Number of states generated (not counting initial state):')
    print('{:>5}  {:>8}  {:>8}  {:>8}  {:>8}'.format('Depth', 'BFS', 'IDS', 'A*-mis', 'A*-Man'))
    for depth in depths:
        print('{:>5}  {:>8.1f}  {:>8.1f}  {:>8.1f}  {:>8.1f}'\
              .format(depth,
                      mean(state_counts[depth]['BFS']),
                      mean(state_counts[depth]['IDS']),
                      mean(state_counts[depth]['A*-mis']),
                      mean(state_counts[depth]['A*-Man'])))