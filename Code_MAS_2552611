import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline

def beta_exp_var(a, b):
    exp = a/(a+b)
    var = (a*b)/(((a+b)**2)*(a+b+1))
    return exp, var

def q1():
    tests = [(1, 1), (50, 25), (25, 50), (25, 25)]
    for a, b in tests:
        sample = np.random.beta(a, b, 1000)
        exp, var = beta_exp_var(a, b)

        plt.hist(sample, range=(0,1), bins=25)
        plt.title('Distribution for:\na: ' + str(a) + ' || b: ' + str(b) +
                 '\nExpectation = ' + str(round(exp, 3))+ ' || Variance = ' + str(round(var, 3)))
        plt.style.use('seaborn')
        plt.show()

def q2(p = None, plot=True):
    a, b = 1, 1
    if not p:
        p = np.random.random()
    data = {'Expectation': [], 'Variance': []}

    for i in range(500):
        if np.random.random() < p:
            a += 1
        else:
            b += 1

        exp, var = beta_exp_var(a, b)    
        data['Expectation'].append(exp)
        data['Variance'].append(var)

    if plot:
        plt.clf()
        x = np.linspace(0, 500, 500)
        
        y, error = np.array(data['Expectation']), np.array(data['Variance'])
        plt.plot(x, y, 'k', color='#CC4F1B')
        plt.fill_between(x, y-error, y+error,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

        plt.title('Succes Probability: ' + str(round(p, 3)) +
                      '\nExpectation = ' + str(round(data['Expectation'][-1], 3))+ 
                      ' || Variance = ' + str(round(data['Variance'][-1], 3)))
        plt.ylim([0, 1])
        plt.show()
    
    return a


def q3(p1 = None, p2 = None, plot=True):
    a1, b1, a2, b2 = 1, 1, 1, 1
    if not p1 or not p2:
        p1, p2 = np.random.random(), np.random.random()

    data = {'Arm1': {'Expectation': [], 'Variance': []}, 'Arm2': {'Expectation': [], 'Variance': []}}

    for i in range(500):
        v1, v2 = np.random.beta(a1, b1, 1)[0], np.random.beta(a2, b2, 1)[0]

        if v1 > v2:
            if np.random.random() < p1:
                a1 += 1
            else:
                b1 += 1

        else:
            if np.random.random() < p2:
                a2 += 1
            else:
                b2 += 1

        for arm, alpha, beta in [('Arm1', a1, b1), ('Arm2', a2, b2)]:
            exp, var = beta_exp_var(alpha, beta)
            data[arm]['Expectation'].append(exp)
            data[arm]['Variance'].append(var)


    if plot:
        plt.clf()
        x = np.linspace(0, 500, 500)
        
        y, error = np.array(data['Arm1']['Expectation']), np.array(data['Arm1']['Variance'])
        plt.plot(x, y, 'k', label = 'Arm 1', color='#CC4F1B')
        plt.fill_between(x, y-error, y+error,
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

        y, error = np.array(data['Arm2']['Expectation']), np.array(data['Arm2']['Variance'])
        plt.plot(x, y, 'k', label = 'Arm 2', color='#1B2ACC')
        plt.fill_between(x, y-error, y+error,
            alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.ylim([0, 1])
        plt.legend()
        plt.title('\nArm1 || p: ' + str(round(p1, 3)) + 
                  ' || Exp: ' + str(round(data['Arm1']['Expectation'][-1], 3)) + 
                  ' || Var: ' + str(round(data['Arm1']['Variance'][-1], 3)) + 
                  ' || N: ' + str(a1 + b1 - 2) +
                 '\nArm2 || p: ' + str(round(p2, 3)) + 
                  ' || Exp: ' + str(round(data['Arm2']['Expectation'][-1], 3)) + 
                  ' || Var: ' + str(round(data['Arm2']['Variance'][-1], 3)) + 
                  ' || N: ' + str(a2 + b2 - 2))
        plt.show()
    return a1 + a2


def q4_greedy(p1 = None, p2 = None, plot=True):
    epsilon = .1
    if not p1 or not p2:
        p1, p2 = np.random.random(), np.random.random()
        
    data = {'Arm1': {'Samples': [], 'Mean': [], 'Var': []}, 'Arm2': {'Samples': [], 'Mean': [], 'Var': []}}

    for i in range(1000):

        if len(data['Arm1']['Samples']) == 0:
            if np.random.random() < p1:
                data['Arm1']['Samples'].append(1)
            else:
                data['Arm1']['Samples'].append(0)
        if len(data['Arm2']['Samples']) == 0:
            if np.random.random() < p2:
                data['Arm2']['Samples'].append(1)
            else:
                data['Arm2']['Samples'].append(0)

        for arm in data:
            data[arm]['Mean'].append(np.array(data[arm]['Samples']).mean())
            data[arm]['Var'].append(np.array(data[arm]['Samples']).var())

        if np.random.random() > epsilon:
            if data['Arm1']['Mean'][-1] > data['Arm2']['Mean'][-1]:
                if np.random.random() < p1:
                    data['Arm1']['Samples'].append(1)
                else:
                    data['Arm1']['Samples'].append(0)
            else:  # p1_exp < p2_exp
                if np.random.random() < p2:
                    data['Arm2']['Samples'].append(1)
                else:
                    data['Arm2']['Samples'].append(0)

        else:  # choose something other than the best option
            if data['Arm1']['Mean'][-1] > data['Arm2']['Mean'][-1]:
                if np.random.random() < p2:
                    data['Arm2']['Samples'].append(1)
                else:
                    data['Arm2']['Samples'].append(0) 
            else:  # p1_exp < p2_exp
                if np.random.random() < p1:
                    data['Arm1']['Samples'].append(1)
                else:
                    data['Arm1']['Samples'].append(0)    

    if plot:
        plt.clf()
        x = np.linspace(0, 1000, 1000)
        
        y, error = np.array(data['Arm1']['Mean']), np.array(data['Arm1']['Var'])
        plt.plot(x, y, 'k', label = 'Arm 1', color='#CC4F1B')
        plt.fill_between(x, y-error, y+error,
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

        y, error = np.array(data['Arm2']['Mean']), np.array(data['Arm2']['Var'])
        plt.plot(x, y, 'k', label = 'Arm 2', color='#1B2ACC')
        plt.fill_between(x, y-error, y+error,
            alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.ylim([0, 1])
        plt.title('\nArm1 || p: ' + str(round(p1, 3)) + 
                  ' || Mean: ' + str(round(data['Arm1']['Mean'][-1], 3)) + 
                  ' || Var: ' + str(round(data['Arm1']['Var'][-1], 3)) + 
                  ' || N: ' + str(len(data['Arm1']['Samples'])) +
                 '\nArm2 || p: ' + str(round(p2, 3)) + 
                  ' || Mean: ' + str(round(data['Arm2']['Mean'][-1], 3)) + 
                  ' || Var: ' + str(round(data['Arm2']['Var'][-1], 3)) + 
                  ' || N: ' + str(len(data['Arm2']['Samples'])))
        plt.legend()
        plt.show()
    
    return np.asarray(data['Arm1']['Samples']).sum() + np.asarray(data['Arm2']['Samples']).sum()


def q4_ucb(p1 = None, p2 = None, plot=True):
    c = 1
    if not p1 or not p2:
        p1, p2 = np.random.random(), np.random.random()

    data = {'Arm1': {'Samples': [], 'Mean': [], 'Bound': []}, 'Arm2': {'Samples': [], 'Mean': [], 'Bound': []}}

    for i in range(1, 1001):

        if len(data['Arm1']['Samples']) == 0:
            if np.random.random() < p1:
                data['Arm1']['Samples'].append(1)
            else:
                data['Arm1']['Samples'].append(0)
        if len(data['Arm2']['Samples']) == 0:
            if np.random.random() < p2:
                data['Arm2']['Samples'].append(1)
            else:
                data['Arm2']['Samples'].append(0)

        for arm in data:
            data[arm]['Mean'].append(np.array(data[arm]['Samples']).mean())
            data[arm]['Bound'].append(np.sqrt(np.log(i)/(2*len(data[arm]['Samples']))))

        ucb1 = data['Arm1']['Mean'][-1] + (c*data['Arm1']['Bound'][-1])
        ucb2 = data['Arm2']['Mean'][-1] + (c*data['Arm2']['Bound'][-1])

        if ucb1 > ucb2:
            if np.random.random() < p1:
                data['Arm1']['Samples'].append(1)
            else:
                data['Arm1']['Samples'].append(0)
        else:
            if np.random.random() < p2:
                data['Arm2']['Samples'].append(1)
            else:
                data['Arm2']['Samples'].append(0)    

    if plot:
        plt.clf()
        x = np.linspace(0, 1000, 1000)
        
        y, error = np.array(data['Arm1']['Mean']), np.array(data['Arm1']['Bound'])
        plt.plot(x, y, 'k', label = 'Arm 1', color='#CC4F1B')
        plt.fill_between(x, y, y+error,
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

        y, error = np.array(data['Arm2']['Mean']), np.array(data['Arm2']['Bound'])
        plt.plot(x, y, 'k', label = 'Arm 2', color='#1B2ACC')
        plt.fill_between(x, y, y+error,
            alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.ylim([0,1])
        plt.title('\nArm1 || p: ' + str(round(p1, 3)) + 
                  ' || Mean: ' + str(round(data['Arm1']['Mean'][-1], 3)) + 
                  ' || Bound: ' + str(round(data['Arm1']['Bound'][-1], 3)) + 
                  ' || N: ' + str(len(data['Arm1']['Samples'])) +
                 '\nArm2 || p: ' + str(round(p2, 3)) + 
                  ' || Mean: ' + str(round(data['Arm2']['Mean'][-1], 3)) + 
                  ' || Bound: ' + str(round(data['Arm2']['Bound'][-1], 3)) + 
                  ' || N: ' + str(len(data['Arm2']['Samples'])))
        plt.legend()
        plt.show()
    return np.asarray(data['Arm1']['Samples']).sum() + np.asarray(data['Arm2']['Samples']).sum()

def experiment_q4():

    for p1, p2 in [(.2, .8), (.3, .7), (.4, .6), (.5, .5), (.2, .4), (.7, .9)]:
        print('Arm1:', p1, 'Arm2:', p2)
        beta = np.asarray([])
        ucb = np.asarray([])
        greedy = np.asarray([])

        for i in range(25):
            r_beta = q3(p1, p2, plot=False)
            r_ucb = q4_ucb(p1, p2, plot=False)
            r_gre = q4_greedy(p1, p2, plot=False)

            beta = np.append(beta, r_beta)
            ucb = np.append(ucb, r_ucb)
            greedy = np.append(greedy, r_gre)

        print('Beta: M =', beta.mean(), 'std =', round(beta.std(),2), 
              ' || UCB: M =', ucb.mean(), 'std =', round(ucb.std(),2), 
              ' || Greedy: M =', greedy.mean(), 'std =', round(greedy.std(),2))
        print()

class Coordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.code = 10*(4 - y) + x
        
    def move(self, direction, max_y):
        max_x = 9
        
        if direction == 3 and self.y < max_y:  # D
            self.y += 1
        elif direction == 2 and self.y > 0:  # U
            self.y -= 1
        elif direction == 1 and self.x < max_x:  # R
            self.x += 1
        elif direction == 0 and self.x > 0:  # L
            self.x -= 1
        self.code = 10*(4 - self.y) + self.x
    
        
    def copy(self):
        return Coordinate(self.x, self.y)
            

class Gridworld():
    def __init__(self, rows, columns, start, goal, cliff = None):
        self.rows = rows
        self.columns = columns
        self.grid = np.zeros((rows, columns), dtype=object)
        self.start = start
        self.goal = goal
        self.cliff = cliff       
        
    def initialize(self):
        
        for i in range(self.columns):
            for j in range(self.rows):
                self.grid[len(self.grid) - 1 - j][i] = 'N'
        
        
        self.grid[self.start.y][self.start.x] = 'S'
        self.grid[self.goal.y][self.goal.x] = 'G'
        
        if self.cliff:
            for cliff_ in self.cliff:
                self.grid[cliff_.y][cliff_.x] = 'C'
            
    def get_reward(self, coordinate):
        state = self.grid[coordinate.y][coordinate.x]
        
        if state == 'C':
            return -100, False
        elif state == 'G':
            return 20, False
        else:
            return -1, True

        
def show_policy(reward_table, rows):
    policy_table = np.zeros((rows, 10), dtype=object)
    
    state = 0
    for state_action in reward_table:
        best_actions = np.argwhere(state_action == np.amax(state_action)).flatten().tolist()
        
        best_action = ''
        for action in best_actions:
            if action == 0:
                best_action += 'L'
            elif action == 1:
                best_action += 'R'
            elif action == 2:
                best_action += 'U'
            elif action == 3:
                best_action += 'D'
                
        if (len(best_action) == 0) or (len(best_action) == 4):
            best_action = '-'
            
        best_action += (' '*(2 - len(best_action)))      
        y_coor = 4 - int(state/10)
        x_coor = state%10

        policy_table[y_coor][x_coor] = best_action
        
        state +=1
    
    return policy_table
        
    
def game(iterations, alpha, gamma, method, epsilon=None, rows=5):
    '''
    iterations: number of times the game is played
    alpha: learning rate
    gamma: discounting rate
    method: SARSA or QLearning
    epsilon: epsilon in the epsilon greedy policy
    rows: number of rows (<10)
    '''
    columns = 10
    
    cliff = []
    for i in range(1, columns-1):
        cliff.append(Coordinate(i, rows-1))

    start = Coordinate(0, rows-1)
    goal = Coordinate(columns-1, rows-1)
    gridworld = Gridworld(rows, columns, start, goal, cliff)
    gridworld.initialize()
    
    reward_table = np.zeros((rows*columns, 4))  # L, R, U, D
    policies = []
    rewards = []
    
    for i in range(iterations):
        current_state, cont, total_reward = Coordinate(0,4), True, 0

        while cont == True:
            old_state = current_state.copy()
            
            #move from the old state
            move = random.sample(np.argwhere(reward_table[old_state.code] == np.amax(reward_table[old_state.code]))
                                 .flatten().tolist(), 1)[0]
            if epsilon and random.random() < epsilon:
                options = [0, 1, 2, 3]
                options.remove(move)
                move = random.sample(options, 1)[0]
                
            current_state.move(move, rows-1)
            reward, cont = gridworld.get_reward(current_state)
            total_reward += reward
            
            #move from the current state
            next_move = random.sample(np.argwhere(reward_table[current_state.code] == np.amax(reward_table[current_state.code]))
                                 .flatten().tolist(), 1)[0]
            if epsilon and random.random() < epsilon:
                options = [0, 1, 2, 3]
                options.remove(move)
                next_move = random.sample(options, 1)[0]
            
            if method == 'SARSA':
                reward_table[old_state.code][move] = round(reward_table[old_state.code][move] + \
                                           alpha*(reward + gamma*reward_table[current_state.code][next_move] \
                                           - reward_table[old_state.code][move]), 4)
            else:  #method == 'QLearning'
                reward_table[old_state.code][move] = round(reward_table[old_state.code][move] + \
                                           alpha*(reward + gamma*reward_table[current_state.code].max() \
                                           - reward_table[old_state.code][move]), 4)                
            
        policies.append(show_policy(reward_table, rows))
        rewards.append(total_reward)

    return policies, rewards

def RL_q1(epsilon=.05):
    pol_q, rew_q = game(2000, alpha=.5, gamma=.8, method='QLearning', epsilon=epsilon)
    pol_s, rew_s = game(2000, alpha=.5, gamma=.8, method='SARSA', epsilon=epsilon)
    
    for method, policy, rewards in [('Q-Learning', pol_q, rew_q), ('SARSA', pol_s, rew_s)]:
        print(method)
        print()
        
        print('Iteration:', 2000)
        print(policy[1999])
        print(np.asarray(rewards).mean())
        print()
    
    plt.plot(rew_q, label='Q-Learning')
    plt.plot(rew_s, label='SARSA')
    plt.legend()
    plt.style.use('seaborn')
    plt.show()
    
def RL_q2():
    for epsilon in [.1, .25, .5]:
        print('Epsilon:', epsilon)
        RL_q1(epsilon)
        print()
