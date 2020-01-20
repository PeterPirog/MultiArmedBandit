import numpy as np

def UCB_selection(env):
    # calculation for UCB method
    action=np.argmax(env.X_UCB)
    #print(' choosen action= ',action)
    return action

def update_mean(old_mean,new_reward,Nj_index):
    mean=old_mean
    x=new_reward
    Nj=Nj_index+1
    mean = ((Nj - 1) / Nj) * mean + x / Nj
    return mean

class bandits:
    def __init__(self):
        probabililty=[0.49,0.5,0.51,0.3]
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

        # mean for j- machine
        self.mean_Nj=np.zeros(self.K)  # <--------- initial mean
        self.mean_total=0
        self.j=np.random.randint(self.K) # active machine choosen randomly
        self.iteration=0
        self.X_UCB=self.mean_Nj

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
        self.Nj[self.j]+=1

        #update mean for j-machine
        self.mean_Nj[self.j]=update_mean(self.mean_Nj[self.j],reward,self.Nj[self.j])

        #update total mean
        N=self.iteration+1
        self.mean_total=update_mean(self.mean_total,reward,N)

        #calculations for UCB
        Nj=self.Nj
        Nj[Nj==0]=1 # correction to prevent division by 0
        self.X_UCB = self.mean_Nj + np.sqrt(2 * np.log(N) / Nj)
        if False:    #set value as True to show debuging commands
            print('-------------------------------------')
            print('N=', N)
            print('reward=',reward)
            print(' self.Nj=', self.Nj)
            print('Corrected Nj=',Nj)
            print('self.mean_Nj', self.mean_Nj)
            print('np.sqrt(2 * np.log(N) /Nj', np.sqrt(2 * np.log(N) /Nj))
            print('X_UCB', self.X_UCB,'\n \n')

        #update history of machine choosing
        self.history[self.iteration]=self.j
        self.iteration+=1
        return reward

#######################   MAIN #################################################

N=10000
env=bandits() #environment initialization
env.create(number_of_iterations=N) #create machines prepared for N iterations

print('\n \n EXPERIMENT:')
print('Number of iterations=', env.N)
print('Number of machines=', env.K)
print('Machines probabilities=', env.probability)
print('Reward set for j-machine=', env.rewards_default)
print('Algorithm UCB1')

for i in range(N):
    action=UCB_selection(env)
    reward=env.pull(machine_number=action)

print('\n \n SUMMARY:')
print('Total reward from experiment=', env.reward_total)
print('Rewards from j-machine=', env.rewards_Nj_total)
print('Usage of j-machine=', env.Nj,'times, in percents',100*env.Nj/env.N,'%')
print('History of machines usage=', env.history)
print('Means for j-machine=', env.mean_Nj)
print('Mean after experiment =', env.mean_total,', in percents % of maximum possible mean',100*env.mean_total/np.max(env.probability))



