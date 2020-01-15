import numpy as np

def epsilon_greed_selection(env, epsilon):
    means=env.mean_Nj
    idx_max=np.argmax(means)

    if np.random.random_sample() <= epsilon: #explore option
        action=np.random.randint(env.K)
    else:   #exploit option
        action = idx_max
    return action

def UCB_selection(env):
    # calculation for UCB method

    action=np.argmax(env.X_UCB)
    print('Means j:', env.mean_Nj)
    print('UCB= ',env.X_UCB,' choosen= ',action)
    return action




def update_mean(old_mean,new_reward,Nj_index):
    mean=old_mean
    x=new_reward
    Nj=Nj_index+1
    mean = ((Nj - 1) / Nj) * mean + x / Nj
    return mean

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

        # mean for j- machine
        #self.mean_Nj=np.zeros(self.K)  # <--------- line for epsilon greed with initial 0  means value
        self.mean_Nj =np.ones(self.K)*10 # <--------- line for epsilon greed with initial high value

        self.mean_total=0
        self.j=np.random.randint(self.K) # active machine choseen randomly
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
        print('X_UCB[self.j]',self.X_UCB[self.j])
        print('self.mean_Nj[self.j]',self.mean_Nj[self.j])
        print('N=',N)
        print(' self.Nj[self.j]=',  self.Nj[self.j])
        print('np.sqrt(2 * np.log(N) / self.Nj[self.j]',np.sqrt(2 * np.log(N) / self.Nj[self.j]))

        self.X_UCB[self.j] = self.mean_Nj[self.j] + np.sqrt(2 * np.log(N) / self.Nj[self.j])

        #update history of machine choosing
        self.history[self.iteration]=self.j
        self.iteration+=1


        return reward

#######################   MAIN #################################################

N=2000
epsilon=0.10

env=bandits() #environment initialization

env.create(number_of_iterations=N) #create machines prepared for N iterations

for i in range(N):
    #action=epsilon_greed_selection(env,epsilon)
    action=UCB_selection(env)
    reward=env.pull(machine_number=action)
    #print('Reward from ',i, 'iteration is:', reward)

print('\n \n')
print('Number of iterations=', env.N)
print('Total reward from experiment=', env.reward_total)
print('Rewards from j-machine=', env.rewards_Nj_total)
print('Usage of j-machine=', env.Nj)
print('History of machines usage=', env.history)
print('Means for j-machine=', env.mean_Nj)
print('Mean total=', env.mean_total)



#print(env.N)