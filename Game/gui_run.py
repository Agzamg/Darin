#!/usr/bin/python3
from gui import Runner
print('Available partners:\n\
 * MCTS - monte-carlo tree search')

play = ''

print(
'To run game just type the color you would like to play for e.g. white or black\n !!! Click twice in order to make a move !!!\n')

while (play != "white" and play != "black"):
    play = str(input())
    if (play != "white" and play != "black"):
        print("Oops, it seem you have mistyped, please enter \"black\" or \"white\"")

# first arg - black player, second arg - white player
# Attention! While playing you need to click one more time to submit the chosen move.
Runner(play)
