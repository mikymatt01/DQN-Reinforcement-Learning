import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import math

EPOCHS = 2000
THRESHOLD = 45

data=[
	{
		'altezza':(54,60),
		'lunghezza schiena':(55,60),
		'peso':(28,35),
		'circonferenza collo':(50,55)
	},
	{
		'altezza':(16,22),
		'lunghezza schiena':(20,30),
		'peso':(1.5,2.5),
		'circonferenza collo':(25,32)
	},
	{
		'altezza':(51,60),
		'lunghezza schiena':(60,70),
		'peso':(20,24),
		'circonferenza collo':(45,50)
	},
	{
		'altezza':(25,32),
		'lunghezza schiena':(37,42),
		'peso':(6,8.5),
		'circonferenza collo':(33,43)
	}
]

result=['Labrador', 'Chihuahua', 'Husky', 'Carlino']

class DATA():
	def __init__(self):
		print('init data')
		self.data=data
		self.success=0
		self.input=len(data[0])
		self.output=int(math.log(len(result), 2))
		print("input atteso", self.input)
		print("output atteso", self.output)
		self.question=10
		self.reward=5


	def randomChoice(self):
		return np.random.randint(low=0, high=1, size=(self.output)).tolist()

	def randomQuestion(self):
		i =  int(random.uniform(0, len(result)))
		q = data[i]
		a = random.uniform(q['altezza'][0], q['altezza'][1])
		l = random.uniform(q['lunghezza schiena'][0], q['lunghezza schiena'][1])
		p = random.uniform(q['peso'][0], q['peso'][1])
		c = random.uniform(q['circonferenza collo'][0], q['circonferenza collo'][1])
		print("altezza\t\t\t", a)
		print("lunghezza schiena\t", l)
		print("peso\t\t\t", p)
		print("circonferenza collo\t", c)
		return [a,l,p,c]

	def step(self, action):
		result = format(int(input('inserisci risposta giusta: (action : ' + str(action) + ')')), "b")
		result = [int(x) for x in result]
		for _ in range(len(result), self.output):
			result.insert(0, 0)
		print("neural action is:{} and your action is:{}".format(action, result))
		self.question=self.question-1
		done = 1 if self.question==0 else 0
		self.success = self.success+1 if action==result else self.success
		reward = self.reward if action==result else 0
		return self.randomQuestion(), reward, done, 0 #next_state, reward, done, _

	def preprocessState(self, state):
		return np.reshape(state, [1,4])

	def normalizeResult(self, list):
		return [1 if x > 0 else 0 for x in list]

class DQN():
	def __init__(self, env, batch_size=64):
		self.gamma = 1.0
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.batch_size = batch_size
		self.memory = deque(maxlen=100000)

		alpha=0.01
		alpha_decay=0.01
		self.env=env
		input=self.env.input #input number
		output=self.env.output #output number

		#create model
		self.model = Sequential()
		self.model.add(Dense(24,input_dim=input, activation='tanh'))
		self.model.add(Dense(48, activation='tanh'))
		self.model.add(Dense(output, activation='tanh')) #prevision must be 1 output
		self.model.compile(loss='mse', optimizer=Adam(learning_rate=alpha, decay=alpha_decay))
		self.model.summary()

	#store tuple in the memory
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	#to maintain a balance between exploration and exploitation
	def choose_action(self, state, epsilon): #epsilon-Greedy policy
		if np.random.random() <= epsilon:
			return self.env.randomChoice() #random output number
		else:
			return self.env.normalizeResult(self.model.predict(state).tolist()[0]) #predict

	#to avoid catastrophic forgetting
	def replay(self, batch_size):
		x_batch, y_batch = [], []
		minibatch = random.sample(self.memory, min(len(self.memory), batch_size)) #take random batches up to batch_size in a memory
		for state, action, reward, next_state, done in minbatch:
			y_target = self.model.predict(state)
			y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
			x_batch.append(state[0])
			y_batch.append(y_target[0])

		self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

	def train(self):
		scores = deque(maxlen=100)
		avg_scores = []

		for e in range(EPOCHS):
			done = False
			i = 0
			state = self.env.randomQuestion()
			state = self.env.preprocessState(state)
			while not done:
				action = self.choose_action(state,self.epsilon)
				next_state, reward, done, _ = self.env.step(action)
				next_state = self.env.preprocessState(next_state)
				self.remember(state, action, reward, next_state, done)
				state = next_state
				self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon) # decrease epsilon
				i += reward

			scores.append(i)
			mean_score = np.mean(scores)
			avg_scores.append(mean_score)
			if mean_score >= THRESHOLD and e >= 100:
				print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
				return avg_scores
			self.replay(self.batch_size)

		print('Did not solve after {} episodes 😞'.format(e))
		return avg_scores
'''
def test(env):
	print("random choice", env.randomChoice())
	print("random question", env.randomQuestion())
	print("step", env.step(env.randomChoice()), "(next_state, reward, done, _)")
'''
if __name__ == "__main__":
	env = DATA()
	#test(env)
	print(env)
	agent = DQN(env)
	print(agent.train())
