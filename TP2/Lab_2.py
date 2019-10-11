"""
Created on Mon Sep  9 15:12:19 2019

@author: alexandre
"""

import numpy as np
import math
import copy
from collections import deque

#%%
class State:
    
    """
    Contructeur d'un état initial
    """
    def __init__(self, pos):
        """
        pos donne la position de la voiture i dans sa ligne ou colonne (première case occupée par la voiture);
        """
        self.pos = np.array(pos)
        
        """
        c,d et prev premettent de retracer l'état précédent et le dernier mouvement effectué
        """
        self.c = self.d = self.prev = None
        
        self.nb_moves = 0
        self.score = 0
        
        self.rock = ()

    """
    Constructeur d'un état à partir du mouvement (c,d)
    """
    def move(self, c, d):
        s = State(self.pos)
        s.prev = self
        s.pos[c] += d
        s.c = c
        s.d = d
        s.nb_moves = self.nb_moves + 1
        # TODO
        return s

    def put_rock(self, rock_pos):
        
        temp_pos = self.pos.copy()
        new_state = State(temp_pos)
        new_state.rock = rock_pos
        
        return new_state
            
    def score_state(self):
        
        cost = (4 - self.pos[0] + self.nb_moves)
        rh.init_positions(self)
        
        for i in range(rh.nbcars):
            if (not rh.horiz[i]) and (rh.move_on[i] > self.pos[0]+1):
                line_occupied = range(self.pos[i],self.pos[i]+rh.length[i])
                if 2 in line_occupied:
                    if rh.move_on[i] == 1:
                        if rh.free_pos[rh.move_on[i]-1,self.pos[i]] == False:
                            cost += 2
                    if rh.move_on[i] >= 2:
                        if rh.free_pos[rh.move_on[i]-1,self.pos[i]] == False:
                            cost += 2
                        if rh.free_pos[rh.move_on[i]-2,self.pos[i]] == False:
                            cost += 1
                    if rh.move_on[i] + rh.length[i] == 5 :
                        if rh.free_pos[rh.move_on[i]+rh.length[i],self.pos[i]] == False:
                            cost += 2
                    if rh.move_on[i] + rh.length[i] <= 4 :
                        if rh.free_pos[rh.move_on[i]+rh.length[i],self.pos[i]] == False:
                            cost += 2
                        if rh.free_pos[rh.move_on[i]+rh.length[i]+1,self.pos[i]] == False:
                            cost += 1
                    else:   
                        cost += 1
                        
        self.score = cost
        

    def success(self):
        return self.pos[0] == 4
    
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
        return (self.score) < (other.score)

#%%
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
        for i in range(self.nbcars):
            if self.horiz[i]:
                self.free_pos[self.move_on[i], state.pos[i]:state.pos[i]+self.length[i]] = False
            else:
                self.free_pos[state.pos[i]:state.pos[i]+self.length[i], self.move_on[i]] = False
        
        if state.rock:
            self.free_pos[state.rock] = False

    
    def possible_moves(self, state):
        self.init_positions(state)
        new_states = []
        for i in range(self.nbcars):
            if self.horiz[i]:
                if state.pos[i]+self.length[i]-1 < 5 and self.free_pos[self.move_on[i], state.pos[i]+self.length[i]]:
                    new_states.append(state.move(i, +1))
                if state.pos[i] > 0 and self.free_pos[self.move_on[i], state.pos[i] - 1]:
                    new_states.append(state.move(i, -1))
            else:
                if state.pos[i]+self.length[i]-1 < 5 and self.free_pos[state.pos[i]+self.length[i], self.move_on[i]]:
                    new_states.append(state.move(i, +1))
                if state.pos[i] > 0 and self.free_pos[state.pos[i] - 1, self.move_on[i]]:
                    new_states.append(state.move(i, -1))
        return new_states
    
    def possible_rock_moves(self, state):
        self.init_positions(state)
        new_states =[]
                
        for i in range(len(self.free_pos)):
            for j in range(len(self.free_pos.T)):
                
                if (not state.rock):
                    if (self.free_pos[i,j]) == True  and (i!=2):
                        new_states.append(state.put_rock((i,j)))                     
                else :
                    if (self.free_pos[i,j]) == True and (i!=state.rock[0]) and ((j!=state.rock[1])) and (i!=2):
                        new_states.append(state.put_rock((i,j)))
          
        return new_states
    

    def print_pretty_grid(self, state):
        self.init_positions(state)
        grid= np.chararray((6, 6))
        grid[:]='-'
        for car in range(self.nbcars):
            for pos in range(state.pos[car], state.pos[car] +self.length[car]):
                if self.horiz[car]:
                    grid[self.move_on[car]][pos] = self.color[car][0]
                else:
                    grid[pos][self.move_on[car]] = self.color[car][0]
        if state.rock:
            grid[state.rock] = 'x'
        print(grid)
        
#%%   
def testRocks():
    rh = Rushhour([True, False, True, False, False, True, False, True, False, True, False, True],
                 [2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3],
                 [2, 2, 0, 0, 3, 1, 1, 3, 0, 4, 5, 5],
                 ["rouge", "vert clair", "jaune", "orange", "violet clair", "bleu ciel", "rose", "violet", "vert", "noir", "beige", "bleu"])
    s0 = State([1, 0, 3, 1, 1, 4, 3, 4, 4, 2, 4, 1])
    
    s1= s0.put_rock((4,4))
    s2 = s1.put_rock((3,2)) 
    
    print("État initial")
    rh.print_pretty_grid(s0)
    print(rh.free_pos)
    print('\n')
    
    print("Roche 4-4")
    rh.print_pretty_grid(s1)
    print(rh.free_pos)
    print('\n')
    
    print("Roche 3-2")
    rh.print_pretty_grid(s2)
    print(rh.free_pos)
    print('\n')

testRocks()


#%%
def testPossibleRockMoves():
    rh = Rushhour([True, False, True, False, False, True, False, True, False, True, False, True],
                 [2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3],
                 [2, 2, 0, 0, 3, 1, 1, 3, 0, 4, 5, 5],
                 ["rouge", "vert clair", "jaune", "orange", "violet clair", "bleu ciel", "rose", "violet", "vert", "noir", "beige", "bleu"])
    s = State([1, 0, 3, 1, 1, 4, 3, 4, 4, 2, 4, 1])
    rh.print_pretty_grid(s)
    sols = rh.possible_rock_moves(s)
    print(len(sols))
    s1 = s.put_rock((3,4))
    sols = rh.possible_rock_moves(s1)
    print(len(sols))

testPossibleRockMoves()

#%%

class MiniMaxSearch:
    def __init__(self, rushHour, initial_state, search_depth):
        self.rushhour = rushHour
        self.state = initial_state
        self.search_depth = search_depth

    def minimax_1(self, current_depth, current_state): 
        
        if current_depth == 0  or current_state.success():
            current_state.score_state()
            return current_state
        
        v = State([0])
        v.score=999
        
        for successor in rh.possible_moves(current_state):
            v=min(v,self.minimax_1(current_depth - 1,successor))
            
        best_move = v
        if (self.search_depth - current_depth) >= 1:
            best_move = best_move.prev
        
        return best_move
    
    def minimax_2(self, current_depth, current_state, is_max): 
        #TODO
        return best_move

    def minimax_pruning(self, current_depth, current_state, is_max, alpha, beta):
        #TODO
        return best_move

    def expectimax(self, current_depth, current_state, is_max):
        #TODO
        return best_move

    def decide_best_move_1(self):
        #c,d = self.minimax(self.search_depth,self.state)
        
        #self.state = self.state.move(c,d)
        
        return 

    def decide_best_move_2(self, is_max):
        #TODO
        return

    def decide_best_move_pruning(self, is_max):
        # TODO
        return

    def decide_best_move_expectimax(self, is_max):
        # TODO
        return

    def solve(self, state, is_singleplayer):
        
        #Check if state is final
        
        #Call decie_best_move until game ends
        
        
        return

    def print_move(self, is_max, state):
        
        new_state = self.minimax_1(self.search_depth,state)
        car = self.rushhour.color[new_state.c]
        move = new_state.d
        
        print('La voiture '+car+' en ' + str(move))
          

#%%
def test_print_move():
    rh = Rushhour([True], [2], [2], ["rouge"])
    s = State([0])
    s = s.put_rock((3,1)) # Roche dans la case 3-1
    s = s.move(0, 1) # Voiture rouge vers la droite

    algo = MiniMaxSearch(rh, s, 1)
    algo.print_move(True, s)
    algo.print_move(False, s)

test_print_move()     
        
    
#    
#rh = Rushhour([True], [2], [2], ["rouge"])
#s = State([0])
#s = s.put_rock((3,1)) # Roche dans la case 3-1
#s = s.move(0, 1) # Voiture rouge vers la droite
#
#
#algo = MiniMaxSearch(rh, s, 1)
#algo.print_move(True, s)
#algo.print_move(False, s)
#
#test = algo.minimax_1(algo.search_depth,algo.state)
#
#test.pos
