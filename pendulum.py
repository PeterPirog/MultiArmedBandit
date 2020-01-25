import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Agent:
    def __init__(self):
        self.action_space=1
        self.states=[]
        self.actions=[]
        self.rewards=[]
        self.np_states=[]
        self.np_actions=[]
        self.np_rewards=[]

        #agent parameters
        self.iteration=0
        self.epoch=0
        self.gamma=0.9
        #environment response results
        self.obs=[]
        self.reward=[]
        self.done=False
        self.info=[]

    def run(self,env):
        self.env=env # can storage environment data



    def get_first_observation(self,obs):
        self.obs=obs
        print('First state=', self.obs)
        self.states.append(self.obs)

    def action(self):
        #angle=self.obs[2]

        action=np.random.randint(2) #random agent

        #print('action taken=',action)
        #action=0 if angle<0 else 1
        self.actions.append(action)
        self.iteration+=1
        return action


    def read_env_response(self,result):
        self.obs=result[0]
        self.reward=result[1]
        self.done=result[2]
        self.info=result[3]
        #print('Obs =',self.obs)
        #print('Reward =',self.reward)
        #print('Done =', self.done)
        #print('self.info=', self.info)
        self.states.append(self.obs)
        self.rewards.append(self.reward)

        self.store_epoch_data()

        if self.done: #go to next epoch
            self.calculate_V_values()
            self.prepare_keras_data()
            self.epoch+=1

    def store_epoch_data(self):
        self.states_stored = len(self.states)
        self.rewards_stored = len(self.rewards)
        self.actions_stored = len(self.actions)

        self.np_rewards=np.asarray(self.rewards)
        self.np_actions=np.asarray(self.actions)
        self.np_states=np.asarray(self.states)
        self.np_V_values=np.zeros(self.states_stored)

    def calculate_V_values(self):
        Agent.np_V_values = np.zeros(Agent.states_stored)
        for i in reversed(range(Agent.states_stored - 1)):
            # print('i=',i)
            Agent.np_V_values[i] = Agent.np_rewards[i] + Agent.gamma * Agent.np_V_values[i + 1]
        # print('V',i,'=R',i,'+V',i+1)
    def prepare_keras_data(self):
        #preparing input data for state+action=>V_value
        a = Agent.np_states[:-1] # states matrix is longer than action vector, the last state has no action
        b = Agent.np_actions.reshape((Agent.states_stored - 1, 1))

        self.data_states_and_actions=np.hstack((a, b)) #merge matrixes
        #self.data_states_and_actions=np.transpose(self.data_states_and_actions) #data transposition

        print('self.data_states_and_actions shape', self.data_states_and_actions.shape)
        #preparing Q_values data
        self.data_Q_values=self.np_V_values[1:] #you dont need to predict Q value for first state
        self.data_Q_values= np.reshape(self.data_Q_values,(Agent.states_stored - 1,1)) #reshape to 2 dimensional matrix
        #self.data_Q_values=np.transpose(self.data_Q_values)

        #preparing V_values data
        self.data_V_values=np.transpose(self.np_V_values)

    ############################################################
env=gym.make("CartPole-v0")
Agent=Agent()
Agent.run(env)
episodes=100
iterations=50

for epoch in range(episodes):
    obs = env.reset()
    Agent.get_first_observation(obs) #all data are stored in agent structure
    for i in range(iterations):
        action=Agent.action()
        result=env.step(action)
        Agent.read_env_response(result)









        #env.render()
        if Agent.done:
            #print('Finished in iteration=',i)
            break


#rint('Agent.states=',Agent.states)
#print('Agent.actions=',Agent.actions)
#print('Agent.rewards=',Agent.rewards)
#print("states stored",Agent.states_stored)
#print("rewards stored",Agent.rewards_stored)
#print("actions stored",Agent.actions_stored)

#print("numpy rewards stored",Agent.np_rewards)
#print("numpy actions stored",Agent.np_actions)
#print("numpy states stored",Agent.np_states)
#print("numpyV values stored",Agent.np_V_values)







#print('Reward values=',Agent.rewards)
#print('V values=',Agent.np_V_values)
#print('data states_actions',Agent.data_states_and_actions)
print('data Q values' ,Agent.data_Q_values)

#from keras.models import Model
#from keras.layers import Input, Dense

#X=np.ones((10, 5))
#Y=np.ones((10, 1))
X=Agent.data_states_and_actions
Y=Agent.data_Q_values

print('Agent.data_states_and_actions=',Agent.data_states_and_actions.shape)
print('Agent.data_Q_values=',Agent.data_Q_values.shape)

input_1 = keras.layers.Input(shape=(5,))
b = keras.layers.Dense(10)(input_1)
b=keras.layers.Dense(1)(b)
model = keras.models.Model(inputs=input_1, outputs=b)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error


model.fit(x=X,y=Y,epochs=10)