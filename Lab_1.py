#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:12:19 2019

@author: alexandre
"""

import numpy as np
import math
import copy
from collections import deque
import heapq

class State:
    
    """
    Contructeur d'un état initial
    """
    def __init__(self, pos):
        """
        pos donne la position de la voiture i (première case occupée par la voiture);
        """
        self.pos = np.array(pos)
        
        """
        c, d et prev premettent de retracer l'état précédent et le dernier mouvement effectué
        """
        self.c = self.d = self.prev = None
        
        self.nb_moves = 0
        self.h = 0

    """
    Constructeur d'un état à partir mouvement (c,d)
    """
    def move(self, c, d):
        
        temp_pos = self.pos.copy()
        temp_pos[c] = temp_pos[c] +  d
        new_state = State(temp_pos)
        new_state.prev = self
        new_state.c = c
        new_state.d = d
        
        return new_state


    """ est il final? """
    def success(self):
        
        return self.pos[0] == 4

    """
    Estimation du nombre de coup restants 
    """
    def estimee1(self):
        # TODO
        return 0

    def estimee2(self, rh):
        # TODO
        return 0
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if len(self.pos) != len(other.pos):
            print("les états n'ont pas le même nombre de voitures")
        
        return np.array_equal(self.pos, other.pos)
        
    def __hash__(self):
        h = 0
        for i in range(len(self.pos)):
            h = 37*h + self.pos[i]
        return int(h)
    
    def __lt__(self, other):
        return (self.nb_moves + self.h) < (other.nb_moves + other.h)
    
    
    
    
def test1():
    positioning = [1, 0, 1, 4, 2, 4, 0, 1]
    s0 = State(positioning)
    b = not s0.success()
    print(b)
    s = s0.move(1,1)
    print(s.prev == s0)
    b = b and s.prev == s0
    print(s0.pos[1], " ", s.pos[1])
    s = s.move(6,1)
    s = s.move(1,-1)
    s = s.move(6,-1)
    print(s == s0)
    b = b and s == s0
    s = s.move(1,1)
    s = s.move(2,-1)
    s = s.move(3,-1)
    s = s.move(4,1)
    s = s.move(4,-1)
    s = s.move(5,-1)
    s = s.move(5,1)
    s = s.move(5,-1)
    s = s.move(6,1)
    s = s.move(6,1)
    s = s.move(6,1)
    s = s.move(7,1)
    s = s.move(7,1)
    s = s.move(0,1)
    s = s.move(0,1)
    s = s.move(0,1)
    print(s.success())
    b = b and s.success()    
    print("\n", "résultat correct" if b else "mauvais résultat")
    
    
test1()

class Rushhour:
    
    def __init__(self, horiz, length, move_on, color=None):
        self.nbcars = len(horiz)
        self.horiz = horiz
        self.length = length
        self.move_on = move_on
        self.color = color
        
        self.free_pos = None
    
    def init_positions(self, state):
        self.free_pos = np.ones((6, 6), dtype=bool)
        for i in range((len(state.pos))):
            if self.horiz[i]:
                self.free_pos[self.move_on[i],state.pos[i]:state.pos[i]+self.length[i]] = 0
            else:
                self.free_pos[state.pos[i]:state.pos[i]+self.length[i],self.move_on[i]] = 0
        
    
    def possible_moves(self, state):
        self.init_positions(state)
        new_states = []
        for i in range((len(state.pos))):
            
            if self.horiz[i]:   # Horizontal cars 
                new_col_left = state.pos[i] - 1
                if new_col_left >= 0:
                    if self.free_pos[self.move_on[i],new_col_left]:
                        new_states.append(state.move(i,-1))
                
                new_col_right = state.pos[i] + self.length[i] 
                if new_col_right <= 5:
                    if self.free_pos[self.move_on[i],new_col_right]:
                        new_states.append(state.move(i,1))
                        
            else :     # Vertical cars 
                new_line_up = state.pos[i] - 1
                if new_line_up >= 0:
                    if self.free_pos[new_line_up, self.move_on[i]]:
                        new_states.append(state.move(i,-1))
                
                new_line_down = state.pos[i] + self.length[i] 
                if new_line_down <= 5:
                    if self.free_pos[new_line_down, self.move_on[i]]:
                        new_states.append(state.move(i,1))
            
        return new_states
    
    
    def solve(self, state):
        visited = set()
        fifo = deque([state])
        visited.add(state)
        
        # TODO
        
        
        return None
    
                    
    def solve_Astar(self, state):
        visited = set()
        visited.add(state)
        
        priority_queue = []
        state.h = state.estimee1()
        heapq.heappush(priority_queue, state)
        
        # TODO
        return None
    
                    
    def print_solution(self, state):
        # TODO
        return 0
    
    
def test2():
    rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3])
    s = State([1, 0, 1, 4, 2, 4, 0, 1])
    rh.init_positions(s)
    b = True
    print(rh.free_pos)
    ans = [[False,False,True,True,True,False],[False,True,True,False,True,False],[False,False,False,False,True,False],
           [False,True,True,False,True,True],[False,True,True,True,False,False],[False,True,False,False,False,True]]
    b = b and (rh.free_pos[i,j] == ans[i,j] for i in range(6) for j in range(6))
    print("\n", "résultat correct" if b else "mauvais résultat")
            
test2()


def test3():
    rh = Rushhour([True, False, True, False, False, True, False, True, False, True, False, True],
                 [2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3],
                 [2, 2, 0, 0, 3, 1, 1, 3, 0, 4, 5, 5])
    s = State([1, 0, 3, 1, 1, 4, 3, 4, 4, 2, 4, 1])
    s2 = State([1, 0, 3, 1, 1, 4, 3, 4, 4, 2, 4, 2])
    print(len(rh.possible_moves(s)))
    print(len(rh.possible_moves(s2)))

test3()






#rh = Rushhour([True, False, False],
#                 [2, 3, 2],
#                 [2, 3, 4])
#
#state = State([1, 1, 1])
    
rh = Rushhour([True, True, False, False, True, True, False, False],
             [2, 2, 3, 2, 3, 2, 3, 3],
             [2, 0, 0, 0, 5, 4, 5, 3],
             ["rouge", "vert clair", "violet", "orange", "vert", "bleu ciel", "jaune", "bleu"])
state = State([1, 0, 1, 4, 2, 4, 0, 1])


#%% Solve function 

visited = set()
fifo = deque([state])
visited.add(state)

n=0
while (len(fifo) != 0) & (n < 100000000):
    
    if (fifo[0].success()): # Check if first item in list fifo is success, if yes break while loop
        print('GG')
        break
    
    else : 
        #Check all possible new states
        moves = rh.possible_moves(fifo[0]) #When not success, check all possible moves from the state in first item of fifo
        
        #Add unseen new states to fifo
        for i in moves:
            if not (i in visited): #For all possible moves from that state, check if they are in list visited 
                fifo.append(i) #If not in list visited, add that new state to fifo (at the end)
        
        #remove state
        visited.add(fifo.popleft()) #Add the first item of fifo to visited list and remove the first item (the one that generated all new states)
        
    n = n + 1 
  

turn_state = fifo[0] #When while loop was broken, fifo[0] was a successful state
pos = turn_state.pos 
d=0
list_move = []

while not np.array_equal(pos,state.pos): # For that sucessful state, get the car information, direction of movement 
    
    if rh.horiz[turn_state.c]:
        if turn_state.d > 0:
            list_move.append('Voiture '+str(turn_state.c)+' vers '+'la droite')
        else:
            list_move.append('Voiture '+str(turn_state.c)+' vers '+'la gauche')
    else:
        if turn_state.d > 0:
            list_move.append('Voiture '+str(turn_state.c)+' vers '+'le bas')
        else:
            list_move.append('Voiture '+str(turn_state.c)+' vers '+'le haut')
       
    #rh.color[turn_state.prev.c]
    
    turn_state = turn_state.prev # From that successfull state, go back to the previous state to get the list of all movements 
    pos = turn_state.pos
    
    d=d+1


list_move = list_move[::-1] # Inverse the list of movements to be in the right order

     
