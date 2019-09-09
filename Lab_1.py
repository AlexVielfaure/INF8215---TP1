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

## essai
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

