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
            
    
    def which_car(self,rh,target_line,target_column):
        for car in range(rh.nbcars):
            if (not rh.horiz[car]): # vertical cars
                for line in range(rh.length[car]):
                    if((self.pos[car]+line == target_line) and (rh.move_on[car] == target_column)):
                        return car
                        break
            else:   # horizontal cars
                for column in range(rh.length[car]):
                    if((self.pos[car]+column == target_column) and (rh.move_on[car] == target_line)):
                        return car
                        break
        return -1
    
    def is_block(self,c,posb,rh):
    
        c=c
        posb=posb
        
        n = rh.length[c]
        rg = range(6)
        poss_rang = list(zip(*(rg[i:] for i in range(n))))
        ar = []
        for i in poss_rang:
            ar.append(abs(i[0] - self.pos[c]))
        sort_rang = [x for _,x in sorted(zip(ar,poss_rang))]
        
        test = ()
        for i in sort_rang:
            if not (posb in i):
                test = i
                break
        #print(test)
        
        new_block = []
        for i in test:
            if not rh.horiz[c]:
                blocking_car = self.which_car(rh,i,rh.move_on[c])
            else: 
                blocking_car = self.which_car(rh,rh.move_on[c],i)
                
            if (blocking_car != c) and (blocking_car != -1):
                if rh.horiz[blocking_car] == rh.horiz[c]:
                    ff = set(test).intersection(set(range(self.pos[blocking_car],self.pos[blocking_car]+rh.length[blocking_car]))).pop()
                    new_block.append((blocking_car,ff))
                else:
                    new_block.append((blocking_car,rh.move_on[c]))
                
        return new_block
    
    def score_state(self,rh):
        
        cost = (4 - self.pos[0] + self.nb_moves)
        rh.init_positions(self)
        
        for i in range(rh.nbcars):
            if (not rh.horiz[i]) and (rh.move_on[i] > self.pos[0]+1):
                line_occupied = range(self.pos[i],self.pos[i]+rh.length[i])
                if 2 in line_occupied:
                    new_c = [(i,2)]
                    compt = 0
                    while new_c:
                        #print(new_c)
                        val = new_c.pop()
                        c = val[0]
                        posb= val[1]
                        
                        new_c += self.is_block(c,posb,rh)
                        compt = compt+2
                    
                    cost += compt
                    
                    if rh.length[i] == 3:
                        cost += (3-self.pos[i])
        #print(cost)             
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
            current_state.score_state(self.rushhour)
            return current_state
        
        v = State([0])
        v.score = np.inf
        
        for successor in self.rushhour.possible_moves(current_state):
            if np.random.randint(2):
                v=min(v,self.minimax_1(current_depth - 1,successor))
            else:
                v= min(self.minimax_1(current_depth - 1,successor),v)
            
        best_move = v
        
        if (self.search_depth - current_depth) >= 1:
            new_best_move = best_move.prev
            new_best_move.score = best_move.score
            
        else:
            new_best_move = best_move
        
        #print(new_best_move.score)
        
        return new_best_move
    
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
        
        self.state = self.minimax_1(self.search_depth,self.state)
        
        #self.state = self.state.move(c,d)
        
        #return 

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
        
        comp = 0
        
        while not self.state.success() and comp < 20:
            self.decide_best_move_1()
            
            print(self.state.pos)
            self.print_move(False, self.state)
            
            comp = comp + 1
    
        return

    def print_move(self, is_max, state):
        
        if not is_max:
            new_state = state#self.minimax_1(self.search_depth,state)
            car = self.rushhour.color[new_state.c]
                        
            if self.rushhour.horiz[new_state.c]:
                if new_state.d > 0:
                    move = 'la droite'
                else:
                    move = 'la gauche'
            else:
                if new_state.d < 0:
                    move = 'le haut'
                else:
                    move = 'la bas'
            print('La voiture '+car+' vers ' + move)
            
        else :
             #Compléter pour roche
            print('roche')
          

#%%
#def test_print_move():
#    rh = Rushhour([True, False],
#                 [2, 3],
#                 [2, 2],
#                 ["rouge", "vert"])
#    s = State([0, 2])
#    algo = MiniMaxSearch(rh, s,3) 
#    algo.rushhour.init_positions(s)
#    
#    print(algo.rushhour.free_pos)
#    
#    #s = s.put_rock((3,1)) # Roche dans la case 3-1
#    #s = s.move(0, 1) # Voiture rouge vers la droite
#
#    algo.print_move(True, s)
#    algo.print_move(False, s)
#
#test_print_move()     
#        


#rh = Rushhour([True, False],
#             [2, 3],
#             [2, 2],
#             ["rouge", "vert"])
#s = State([0, 2])
#algo = MiniMaxSearch(rh, s, 1) 
#algo.rushhour.init_positions(s)
#algo.solve(s, True)


# Solution optimale: 9 moves
#rh = Rushhour([True, False, False, False, True],
#                 [2, 3, 2, 3, 3],
#                 [2, 4, 5, 1, 5],
#                 ["rouge", "vert", "bleu", "orange", "jaune"])
#s = State([1, 0, 1, 3, 2])
#algo = MiniMaxSearch(rh, s,1) 
#algo.rushhour.init_positions(s)
#print(algo.rushhour.free_pos)
#algo.solve(s, True)


# solution optimale: 16 moves
#rh = Rushhour([True, True, False, False, True, True, False, False],
#                 [2, 2, 3, 2, 3, 2, 3, 3],
#                 [2, 0, 0, 0, 5, 4, 5, 3],
#                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
#s = State([1, 0, 1, 4, 2, 4, 0, 1])
#algo = MiniMaxSearch(rh, s, 1) 
#algo.rushhour.init_positions(s)
#print(algo.rushhour.free_pos)
#algo.solve(s, True)


# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,1)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
algo.solve(s, True)


