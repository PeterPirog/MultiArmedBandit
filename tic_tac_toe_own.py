#https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
#https://www.youtube.com/watch?v=ISk80iLhdfU&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs&index=4


from __future__ import print_function,division
from builtins import range, input

import numpy as np
import matplotlib.pyplot as plt

LENGTH=3 # size of board

class Agent:
    def __init__(self,eps=0.1,alpha=0.5):
        self.eps=eps # probability of random acyion instead of greedy
        self.alpha=alpha #learning rate
        self.verbose=False
        self.state_history=[]

    def setV(self,V):
        self.V=V

    def set_symbol(self,sym):
        self.sym=sym

    def set_verbose(self,v):
        self.verbose=v

    def reset_history(self):
        self.state_history=[]

    def take_action(self,env):
        #choose an action based on epsilon greedy strategy
        r=np.random.rand()
        best_state=None
        if r<self.eps
            #take random action
            if self.verbose:
                print("Taking a random action")
            possible_moves=[]
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i,j):
                        possible_moves.append((i,j))
            idx=np.random.choice(len(possible_moves))   #tutaj wyjaśnić dlaczego efektem len jest wektor a nie wartość ????
            next_move=possible_moves[idx]
        else:
            #choose the best action based on current values of states
            pos2value={} # for debuging
            next_move=None
            best_value=-1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i,j):
                        #what is the state if we made this move?
                        env.board[i,j]=self.sym #change only to read this state
                        state=env.get.state()
                        env.board[i,j]=0 #back to 0
                        pos2value[(i,j)]=self.V[state]
                        if self.V[state]>best_value:
                            best_value=self.V[state]
                            best_state=state
                            next_move=(i,j)
            #drawing
            if self.verbose:
                print("Taking a greedy action")
                for i in range(LENGTH):
                    print("-------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i,j):
                            print(" %.2f|" % pos2value[(i,j)],end="")
                        else:
                            print("  ",end="")
                            if env.board[i,j]==env.x:
                                print("x  |",end="")
                            elif env.board[i,j]==env.o:
                                print("O  |",end="")
                            else:
                                print("  |",end="")
                    print("")
                print("-------------------")
        env.board[next_move[0],next_move[1]]=self.sym

    def update_state_history(self,s):
        self.state_history.append(s)

    def update(self,env):
        reward=env.reward(self.sym)
        target=reward
        for prev in reversed(self.state_history):
            value=self.V[prev]+self.alpha*(target-self.V[prev])
            self.V[prev]=value
            target=value
        self.reset_history()


if __name__=='__main__':
    #train the agent
    p1=Agent()
    p2=Agent()