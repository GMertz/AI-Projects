from copy import deepcopy


def diagram_to_state(diagram):
    """Converts a list of strings into a list of lists of characters (strings of length 1.)"""
    return [list(a) for a in diagram]


INITIAL_STATE = diagram_to_state(['........',
                                  '........',
                                  '........',
                                  '...#O...',
                                  '...O#...',
                                  '........',
                                  '........',
                                  '........'])


def count_pieces(state):
    """Returns a dictionary of the counts of '#', 'O', and '.' in state."""
    result = {'#': 0, 'O': 0, '.': 0}
    for row in state:
        for square in row:
            result[square] += 1
    return result


def prettify(state):
    """
    Returns a single human-readable string representing state, including row and column indices and counts of
    each color.
    """
    result = ' 01234567\n'
    for i, row in enumerate(state):
        result += str(i)
        for char in row:
            result += char
        result += str(i) + '\n'
    result += ' 01234567\n' + count_pieces(state).__str__() + '\n'
    return result


def opposite(color):
    """opposite('#') returns 'O'. opposite('O') returns '#'."""
    if color == '#':
        return 'O'
    else:
        return '#'


def flips(state, r, c, color, dr, dc):
    """
    Returns a list of pieces that would be flipped if color played at r, c, but only searching along the line
    specified by dr and dc. For example, if dr is 1 and dc is -1, consider the line (r+1, c-1), (r+2, c-2), etc.

    :param state: The game state.
    :param r: The row of the piece to be  played.
    :param c: The column of the piece to be  played.
    :param color: The color that would play at r, c.
    :param dr: The amount to adjust r on each step along the line.
    :param dc: The amount to adjust c on each step along the line.
    :return A list of (r, c) pairs of pieces that would be flipped.

    we continue searching upon the given line as long as the current tile we are searching is the opposite
    of the given color, appending the current tile's coordinates to the result list as we search.
    If we end our search with a blank tile ('.') or if we end our search with our same color
    but steps = 0 (meaning we have seen 2 consecutive tiles of our color)
    we throw out results and just return an empty list, otherwise we return our result
    """
    result = []
    r1 = r + dr
    c1 = c + dc
    steps = 0
    while c1 in range(8) and r1 in range(8):
        if state[r1][c1] == '.':
            return []
        if state[r1][c1] == color and steps >= 1:
            return result
        elif state[r1][c1] == color and steps == 0:
            return []
        result.append((r1, c1))
        r1 += dr
        c1 += dc
        steps += 1
    return []


OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def flips_something(state, r, c, color):
    """Returns True if color playing at r, c in state would flip something.

    identical logic to flips_something, except that we iterate through every offset to check every direction from the piece being placed,
    and if we run into a case where we know something would be flipped, we immediately return True, otherwise False is returned at the end
    """
    for slope in OFFSETS:
        dr = slope[0]
        dc = slope[1]
        r1 = r + dr
        c1 = c + dc
        i = 0
        while c1 in range(8) and r1 in range(8):
            if state[r1][c1] == '.':
                break
            if state[r1][c1] == color and i >= 1:
                return True
            elif state[r1][c1] == color and i == 0:
                break
            r1 += dr
            c1 += dc
            i += 1
    return False


def legal_moves(state, color):
    """
    Returns a list of legal moves ((r, c) pairs) that color can make from state. Note that a player must flip
    something if possible; otherwise they must play the special move 'pass'.

    we iterate though all open blocks on the board ('.'), and if playing the given color there flips something we add
    it to the list legal. if legal is empty at the end (no legal moves) we return ['pass'] otherwise return legal
    """
    legal = []
    for r, row in enumerate(state):
        for c, tile in enumerate(row):
            if flips_something(state, r, c, color) and tile is '.':
                legal.append((r, c))
    if not legal:
        return ['pass']
    else:
        return legal


def successor(state, move, color):
    """
    Returns the state that would result from color playing move (which is either a pair (r, c) or 'pass'.
    Assumes move is legal.

     we start by making a deep copy of state so we wont change state,
     we then add the tiles that were flipped by playing 'move' to the list flipped, and then for each of these tiles we check
     if something else was flipped, and add those tiles onto flipped to repeat the process until all tiles
     that need to be flipped have been flipped, then we return the new state (oregon)
    """
    oregon = deepcopy(state)
    if move is 'pass':
        return state

    flipped = []
    oregon[move[0]][move[1]] = color
    for slope in OFFSETS:
        flipped += flips(oregon, move[0], move[1], color, slope[0], slope[1])
    for tile in flipped:
        oregon[tile[0]][tile[1]] = color
        if flips_something(oregon, tile[0], tile[1], color):
            for slope in OFFSETS:
                flipped += flips(oregon, tile[0], tile[1], color, slope[0], slope[1])

    return oregon


def score(state):
    """
    Returns the scores in state. More positive values (up to 64 for occupying the entire board) are better for '#'.
    More negative values (down to -64) are better for 'O'.
    """
    counts = count_pieces(state)
    return counts['#'] - counts['O']


def game_over(state):
    """
    Returns true if neither player can flip anything.
    """
    if legal_moves(state, '#') == ['pass'] and legal_moves(state, 'O') == ['pass']:
        return True
    return False
