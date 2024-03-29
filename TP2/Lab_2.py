"""
Created on Mon Sep  9 15:12:19 2019

@author: alexandre
"""

import numpy as np
import math
import copy
import pandas as pd
import random
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
        s.rock = self.rock
        # TODO
        return s

    def put_rock(self, rock_pos):
        

        temp_pos = self.pos.copy()
        new_state = State(temp_pos)
        new_state.prev = self
        new_state.c = self.c
        new_state.d = self.d
        new_state.nb_moves = self.nb_moves + 1
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
        dis = abs(test[0]-self.pos[c])
        
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
               
        return [new_block,dis,test[0]]
    
    def score_state(self,rh):
        
        cost = (4 - self.pos[0] + self.nb_moves)
        rh.init_positions(self)
        
        for i in range(rh.nbcars):
            if (not rh.horiz[i]) and (rh.move_on[i] > self.pos[0]+1):
                line_occupied = range(self.pos[i],self.pos[i]+rh.length[i])
                if 2 in line_occupied:
                    new_c = [(i,2)]
                    compt = 0
                    visited = []
                    while new_c:
                        #print(new_c)
                        val = new_c.pop()
                        c = val[0]
                        if not c in visited:
                            visited.append(c)
                            
                            posb= val[1]

                            block_val = self.is_block(c,posb,rh)
                            new_c += block_val[0]
                            
                            #If a 3 length blocking car in the loop, add weight
                            if new_c and not rh.horiz[c] and rh.length[c] == 3 and rh.move_on[c] > self.pos[0]+1:
                                compt +=2
                                
                            dis = block_val[1]
                            
                            compt = compt+dis+3
                            
                            if self.pos[c] - self.prev.pos[c] == 0 : #If you don't move a blocking car
                                compt += 1
                            
                            #If you move a blocking car in a position that cannot move
                            pos_car = 0
                            for j in rh.possible_moves(self) :
                                if self.pos[c] - j.pos[c] != 0:
                                    pos_car += 1
                            if pos_car == 0 :
                                pos_car = -20
                                
                            compt += (2-pos_car)
                    
                    cost += compt
                    
                    if rh.length[i] == 3: #If a blocking car of red of length 3 is not in good position 
                        if (3-self.pos[i]) > 0:
                            cost += 1      
        #Prevent cycles
        if self.prev.c:
            if (self.prev.c - self.c) == 0:
                if (self.prev.d + self.d) == 0 :
                    cost +=1000
                    
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
        self.search_car = search_depth
        self.search_rock = 2

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

        return new_best_move
    
    def minimax_2(self, current_depth, current_state, is_max): 
        global nb_state
        if current_depth == 0  or current_state.success():

            #print(current_state.rock)
            current_state.score_state(self.rushhour)
            nb_state = nb_state + 1
            return current_state
        
        v = State([0])
        
        if not is_max:
            v.score = np.inf
            for successor in self.rushhour.possible_moves(current_state):
                
                if np.random.randint(2):
                    v=min(v,self.minimax_2(current_depth - 1,successor,not is_max))
                else:
                    v=min(self.minimax_2(current_depth - 1,successor,not is_max),v)
        else:
            v.score = -1*np.inf
            #self.rushhour.print_pretty_grid(current_state)
            for successor in self.rushhour.possible_rock_moves(current_state):
                #self.rushhour.print_pretty_grid(successor)
                if np.random.randint(2):
                    v=max(v,self.minimax_2(current_depth - 1,successor,not is_max))
                else:
                    v= max(self.minimax_2(current_depth - 1,successor,not is_max),v)
         
        best_move = v
        
   
        if (self.search_depth - current_depth) >= 1:
            new_best_move = best_move.prev
            new_best_move.score = best_move.score  
        else:
            new_best_move = best_move

        return new_best_move
        
    
    def minimax_pruning(self, current_depth, current_state, is_max, alpha, beta):
        global nb_state
        if current_depth == 0  or current_state.success():

            current_state.score_state(self.rushhour)
            nb_state = nb_state + 1
            return current_state
        
        v = State([0])
        
        if not is_max:
            v.score = np.inf
            for successor in self.rushhour.possible_moves(current_state):
                if np.random.randint(2):
                    v=min(v,self.minimax_pruning(current_depth - 1,successor,not is_max, alpha, beta))
                else:
                    v=min(self.minimax_pruning(current_depth - 1,successor,not is_max, alpha, beta),v)
                
                if v.score <= alpha: # Pruning
                    break
                beta = min(beta,v.score)
        else:
            v.score = -1*np.inf
            for successor in self.rushhour.possible_rock_moves(current_state):
                if np.random.randint(2):
                    v=max(v,self.minimax_pruning(current_depth - 1,successor,not is_max, alpha, beta))
                else:
                    v= max(self.minimax_pruning(current_depth - 1,successor,not is_max, alpha, beta),v)
                if v.score >= beta: # Pruning
                    break
                alpha = min(alpha,v.score)
         
        best_move = v
     
        if (self.search_depth - current_depth) >= 1:
            new_best_move = best_move.prev
            new_best_move.score = best_move.score  
        else:
            new_best_move = best_move

        return new_best_move
        


    def expectimax(self, current_depth, current_state, is_max):
        global nb_state
        global option
        if current_depth == 0  or current_state.success():

            current_state.score_state(self.rushhour)
            nb_state = nb_state + 1
            return current_state
        
        v = State([0])

        if not is_max:
            v.score = np.inf
            for successor in self.rushhour.possible_moves(current_state):
                if np.random.randint(2):
                    v=min(v,self.expectimax(current_depth - 1,successor,not is_max))
                else:
                    v=min(self.expectimax(current_depth - 1,successor,not is_max),v)
        else: # expectimax
            if option == 0:
            # alea
                v = self.rushhour.possible_rock_moves(current_state)[0]
                v.score = 0
                for successor in self.rushhour.possible_rock_moves(current_state):
                    p = 1/len(self.rushhour.possible_rock_moves(current_state))
                    v.score = v.score + p*(self.expectimax(current_depth - 1,successor,not is_max).score)             
            elif option == 1:
            # pessimiste
                v = self.rushhour.possible_rock_moves(current_state)[0]
                v.score = 0
                maxim = -np.inf
                for successor in self.rushhour.possible_rock_moves(current_state):
                    p = 1/len(self.rushhour.possible_rock_moves(current_state))
                    maxim = max(maxim,self.expectimax(current_depth - 1,successor,not is_max).score)
                for successor in self.rushhour.possible_rock_moves(current_state):
                    p = 1/len(self.rushhour.possible_rock_moves(current_state))
                    # On favorise les grandes valeurs
                    k = self.expectimax(current_depth - 1,successor,not is_max).score / maxim
                    v.score = v.score + p*k*(self.expectimax(current_depth - 1,successor,not is_max).score)
            # optimiste
            else:
                v = self.rushhour.possible_rock_moves(current_state)[0]
                v.score = 0
                maxim = -np.inf
                for successor in self.rushhour.possible_rock_moves(current_state):
                    p = 1/len(self.rushhour.possible_rock_moves(current_state))
                    maxim = max(maxim,self.expectimax(current_depth - 1,successor,not is_max).score)
                for successor in self.rushhour.possible_rock_moves(current_state):
                    p = 1/len(self.rushhour.possible_rock_moves(current_state))
                    # On favorise les petites valeurs
                    k = self.expectimax(current_depth - 1,successor,not is_max).score / maxim
                    v.score = v.score + p/k*(self.expectimax(current_depth - 1,successor,not is_max).score)
            
        best_move = v
     
        if (self.search_depth - current_depth) >= 1:
            new_best_move = best_move.prev
            new_best_move.score = best_move.score  
        else:
            new_best_move = best_move

        return new_best_move
        
    

    def decide_best_move_1(self):
        
        self.state = self.minimax_1(self.search_depth,self.state)
                
        return 

    def decide_best_move_2(self, is_max):

        if not is_max:
            self.search_depth = self.search_car
            self.state = self.minimax_2(self.search_depth,self.state,is_max)
        else:
            self.search_depth = self.search_rock
            self.state = self.minimax_2(self.search_depth,self.state,is_max)
        
        return
        

    def decide_best_move_pruning(self, is_max):
        
        if not is_max:
            self.search_depth = self.search_car
            self.state = self.minimax_pruning(self.search_depth,self.state,is_max, -np.inf, np.inf)
        else:
            self.search_depth = self.search_rock
            self.state = self.minimax_pruning(self.search_depth,self.state,is_max, -np.inf, np.inf)
        
        return

    def decide_best_move_expectimax(self, is_max):
        
        if not is_max:
            self.search_depth = self.search_car
            self.state = self.expectimax(self.search_depth,self.state,is_max)
        else:
            self.search_depth = self.search_rock
            all_moves = self.rushhour.possible_rock_moves(self.state)
            self.state = random.choice(all_moves) # random move
                
        return

    def solve(self, state, is_singleplayer, pruning=False, expectimax=False): # default arguments
        
        if is_singleplayer:
            comp = 0
            is_max = False
            while not self.state.success() and comp < 40:
                if not pruning:
                    self.decide_best_move_1()
                else:
                    self.decide_best_move_pruning(is_max)
                
                #print(self.state.pos)
                self.print_move(False, self.state)
                
                comp = comp + 1
                
        else :
            comp = 0
            is_max = False
            while not self.state.success() and comp < 160:
                if not pruning:
                    self.decide_best_move_2(is_max)
                else:
                    if not expectimax:
                        self.decide_best_move_pruning(is_max)
                    else:
                        self.decide_best_move_expectimax(is_max)

                self.print_move(is_max, self.state)
                
                is_max=not is_max
                
                comp = comp + 1

        return

    def print_move(self, is_max, state):
        global cpt
        new_state = state
        
        if not is_max:

            car = self.rushhour.color[new_state.c]
                        
            if self.rushhour.horiz[new_state.c]:
                cpt = cpt + 1 # counter of player moves
                if new_state.d > 0:
                    move = 'la droite'
                else:
                    move = 'la gauche'
            else:
                cpt = cpt + 1
                if new_state.d < 0:
                    move = 'le haut'
                else:
                    move = 'le bas'
            print('La voiture '+car+' vers ' + move)
            
        else :
             #Compléter pour roche
            print('Roche : ' + str(new_state.rock))
          

#%%

# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s,1) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
algo.solve(s, True)



# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 1) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
algo.solve(s, True)

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


#%%  NORMAL
tableau = np.zeros((5,3))
# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s, 3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
nb_state = 0
algo.solve(s, False)
tableau[0,0] = nb_state


# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 3) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
nb_state = 0
algo.solve(s, False)
tableau[0,1] = nb_state

# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
nb_state = 0
algo.solve(s, False)
tableau[0,2] = nb_state

#%% PRUNING
# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s, 3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True) # testing pruning
tableau[1,0] = nb_state


# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 3) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False) 
nb_state = 0
algo.solve(s, False,True) # testing pruning
tableau[1,1] = nb_state


# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True) # testing pruning
tableau[1,2] = nb_state



<<<<<<< Updated upstream
#%% Expectimax
tab = np.zeros((3,3))
for option in range(3):
    # Solution optimale: 9 moves
    rh = Rushhour([True, False, False, False, True],
                     [2, 3, 2, 3, 3],
                     [2, 4, 5, 1, 5],
                     ["rouge", "vert", "bleu", "orange", "jaune"])
    s = State([1, 0, 1, 3, 2])
    algo = MiniMaxSearch(rh, s, 3)
    algo.rushhour.init_positions(s)
    print(algo.rushhour.free_pos)
    #algo.solve(s, False)
    nb_state = 0
    cpt = 0
    algo.solve(s, False,True,True) # testing expectimax
    tableau[2+option,0] = nb_state
    tab[option,0] = cpt
    
    
    # solution optimale: 16 moves
    rh = Rushhour([True, True, False, False, True, True, False, False],
                     [2, 2, 3, 2, 3, 2, 3, 3],
                     [2, 0, 0, 0, 5, 4, 5, 3],
                     ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
    s = State([1, 0, 1, 4, 2, 4, 0, 1])
    algo = MiniMaxSearch(rh, s, 3) 
    algo.rushhour.init_positions(s)
    print(algo.rushhour.free_pos)
    #algo.solve(s, False) 
    nb_state = 0
    cpt = 0
    algo.solve(s, False,True,True) # testing expectimax
    tableau[2+option,1] = nb_state
    tab[option,1] = cpt
    # solution optimale: 14 moves

    rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                     [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                     [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                     ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
    algo = MiniMaxSearch(rh, s,3)
    algo.rushhour.init_positions(s)
    print(algo.rushhour.free_pos)
    #algo.solve(s, False)
    nb_state = 0
    cpt = 0
    algo.solve(s, False,True,True) # testing expectimax
    tableau[2+option,2] = nb_state
    tab[option,2] = cpt
#%%
=======
#%% Expectimax Optimiste
option = 0
# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s, 3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[2,0] = nb_state



# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 3) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False) 
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[2,1] = nb_state


# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[2,2] = nb_state

#%% Expectimax Pessimiste
option = 1
# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s, 3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[3,0] = nb_state


# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 3) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False) 
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[3,1] = nb_state


# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[3,2] = nb_state
#%% Expectimax Pessimiste
option = 2
# Solution optimale: 9 moves
rh = Rushhour([True, False, False, False, True],
                 [2, 3, 2, 3, 3],
                 [2, 4, 5, 1, 5],
                 ["rouge", "vert", "bleu", "orange", "jaune"])
s = State([1, 0, 1, 3, 2])
algo = MiniMaxSearch(rh, s, 3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[4,0] = nb_state



# solution optimale: 16 moves
rh = Rushhour([True, True, False, False, True, True, False, False],
                 [2, 2, 3, 2, 3, 2, 3, 3],
                 [2, 0, 0, 0, 5, 4, 5, 3],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
s = State([1, 0, 1, 4, 2, 4, 0, 1])
algo = MiniMaxSearch(rh, s, 3) 
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False) 
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[4,1] = nb_state


# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
#algo.solve(s, False)
nb_state = 0
algo.solve(s, False,True,True) # testing expectimax
tableau[4,2] = nb_state
>>>>>>> Stashed changes

tableau_frame2 = pd.DataFrame(tableau,columns = ["9 moves","16 moves","14 moves"],
                               index = ["No Pruning","Pruning","Expectimax Optimiste"
                                        ,"Expectimax Pessimiste","Expectimax Aléatoire"])


