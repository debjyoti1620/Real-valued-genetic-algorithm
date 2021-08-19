import numpy as np
import pandas as pd



def initialization(fun, xGuess, popSize, minBound, maxBound):
	xSize = len(xGuess)
	columns = ['x'+str(i) for i in range(xSize)]
	df = pd.DataFrame(np.random.uniform(minBound, maxBound, (popSize, xSize)), columns=columns)
	df.at[0, :] = xGuess
	df['fitness'] = df.apply(fun, axis=1)

	return df


def selection(df, popSize):
	df = df.copy()
	# sorting according to fitness
	df.sort_values(by = ['fitness'], inplace=True, ignore_index=True)
	# dropping last half rows
	df = df.drop(df.tail(int(popSize/2)).index)

	return df


def crossover(fun, df, popSize):
	popSize = int(popSize)
	halfPopSize = int(popSize/2)
	alpha = np.random.rand(popSize)
	offsprings = pd.DataFrame(columns=df.columns.values)
	variables = df.columns.values[:-1]
	# uniform crossover/ Arithmatic crossover
	for i in range(halfPopSize):
		offsprings.loc[2*i] = (alpha[2*i] * df.loc[2*i % halfPopSize] + (1 - alpha[2*i]) * df.loc[(2*i + 1) % halfPopSize]).tolist()
		offsprings.loc[2*i + 1] = ((1 - alpha[2*i + 1]) * df.loc[2*i % halfPopSize] + alpha[2*i + 1] * df.loc[(2*i + 1) % halfPopSize]).tolist()

	# updating fitness values
	offsprings['fitness'] = offsprings[variables].apply(fun, axis=1)

	return offsprings

def mutation(fun, df, popSize):
	df = df.copy()
	df.sort_values(by=['fitness'], ignore_index=True)
	variables = df.columns.values[:-1]
	varSize = len(variables)
	
	for i in range(popSize):
		if np.random.rand() < 0.05:
			mutIndex = np.random.randint(0, varSize)
			df.loc[i, variables[mutIndex]] = df.loc[i, variables[mutIndex]] + np.random.normal(0, df.loc[:, variables[mutIndex]].var(), popSize)[i]

	# updating fitness values
	df['fitness'] = df[variables].apply(fun, axis=1)

	return df


def RVGA(fun, guess, popSize, minBound, maxBound, tolerance=1e-10, maxIter=1000, k = 10):
	#  Real Valued Genetic Algorithm

	x = np.array(guess)
	pop = initialization(fun, x, popSize, minBound, maxBound)
	variables = pop.columns.values[:-1]

	consIter = 0
	iter = 0

	while consIter < k and iter < maxIter:
		# selected population
		selPop = selection(pop, popSize)

		# crossover
		offsprings = crossover(fun, selPop, popSize)


		# mutation
		mutPop = mutation(fun, offsprings, popSize)

		# merging and selecting the best n points
		tempPop = pop.copy()
		pop = pd.concat([offsprings, mutPop, tempPop], ignore_index=True)
		pop.drop_duplicates(inplace=True, ignore_index=True)
		pop.sort_values(by=['fitness'], ignore_index=True)
		pop.drop(pop.tail(len(pop.index) - popSize).index, inplace=True)

		# norm of difference of x in current and last iteration
		xNorm = np.linalg.norm(np.array(pop.loc[0,variables].values.tolist()) - x)

		if xNorm < tolerance:
			# increasing consecutive iteration when tolerance condition met
			consIter += 1
		elif consIter > 0 and xNorm > tolerance:
			# resetting consecutive iteration when tolerance condition not met
			consIter = 0

		x = np.array(pop.loc[0, variables].values.tolist())
		iter += 1
		print(iter)

		print(pop.sort_values(by =['fitness'], ignore_index=True))

	return pop, iter




def sphere(x):
	return (x[0] - 5)**2 / 4 + (x[1] + 2)**2/9 -1



populationSize = 300
minBound = -100
maxBound = 100
guess = [100, 50]


population, iteration = RVGA(sphere,  guess, populationSize, minBound, maxBound, 1e-6, 1000, 6)

print(population)
print(iteration)