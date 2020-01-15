import numpy as np
from see import see

class bandits:
    def __init__(self):
        probabililty=[0.1,0.3,0.5]
        rewards_default=[1,1,1]
        self.probability=np.array(probabililty)
        self.rewards_default=rewards_default
        self.K=len(self.probability)
        if self.K!=len(self.rewards_default):
            print("Warning - reward vector isn't equal probability vector, reward vector  initiated by ones")
            self.rewards_default=np.ones(self.K)
            print("Machine reward vector:",self.rewards_default)
        self.reward_total=0
        self.Nj=np.zeros(self.K)
        self.rewards_Nj_total=np.zeros(self.K)
        self.N=[]
        self.mean_Nj=np.zeros(self.K)  # mean for j- machine
        self.mean_total=0
        self.j=0 # active machine
        self.iteration=0

    def create(self,number_of_iterations=1):
        self.N=number_of_iterations
        self.history=np.zeros(self.N)

    def pull(self,machine_number=0):
        self.j=machine_number
        p=self.probability[self.j]
        reward_default=self.rewards_default[self.j]

        #print('reward_default=',reward_default)
        #get reward from j-machine
        if np.random.random_sample()<=p:
            reward=reward_default
        else:
            reward = 0

        # update total reward with reward in this iteration
        self.reward_total += reward

        #update reward for j-machine
        self.rewards_Nj_total[self.j]+=reward

        #update number of usage for j-machine
        self.Nj+=1

        #update history of machine choosing
        self.history[self.iteration]=self.j
        self.iteration+=1


        return reward



N=10
env=bandits() #environment initialization

env.create(number_of_iterations=N)

for i in range(N):
    reward=env.pull(machine_number=2)
    print('Reward from ',i, 'iteration is:', reward)


print('total reward=', env.reward_total)
print('Rewards from j-machine=', env.rewards_Nj_total)
print('History of usage=', env.history)
print('Means=', env.mean_Nj)
#print(see(env))



#print(env.N)