import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

pygame.init()

# Screen setup
screen_size = 800
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Treasure Hunting with UCB")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Game settings
num_quadrants = 4
box_size = 40
treasure_probabilities = [0.4, 0.5, 0.6, 0.7]  # Probability of finding treasure in each quadrant
max_treasures_per_quad = 100  # Maximum number of treasures that can appear in a quadrant

# Initialize treasures with unique positions
def initialize_treasures():
    treasures = []
    for quad in range(num_quadrants):
        # Calculate the expected number of treasures in this quadrant based on its probability
        expected_treasures = int(round(max_treasures_per_quad * treasure_probabilities[quad]))
        
        quad_treasures = set()
        while len(quad_treasures) < expected_treasures:
            x = (quad % 2) * 400 + np.random.randint(0, 10) * box_size
            y = (quad // 2) * 400 + np.random.randint(0, 10) * box_size
            quad_treasures.add((x, y))
        treasures.extend([(x, y, quad) for x, y in quad_treasures])
    return treasures

treasures = initialize_treasures()

class BernoulliUCBBandit:
    def __init__(self, k):
        self.k = k
        self.q = np.zeros(k)  # Estimated probabilities of success
        self.n = np.zeros(k)  # Times each arm has been pulled
        self.total_pulls = 0

    def pull_arm(self):
        if self.total_pulls < self.k:  # Explore each arm once before using UCB
            return self.total_pulls
        ucb_values = self.q + np.sqrt(2 * np.log(self.total_pulls) / (self.n + 1e-10))  # UCB1 formula
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
        self.total_pulls += 1

bandit = BernoulliUCBBandit(num_quadrants)
cumulative_regrets = []
optimal_reward_probability = max(treasure_probabilities)  # Best possible probability of finding a treasure
total_rewards = 0

# Game loop 
running = True
clock = pygame.time.Clock()
for i in range(3000):
    screen.fill(WHITE)
    # Draw treasures
    for treasure in treasures:
        pygame.draw.rect(screen, GREEN, (treasure[0], treasure[1], box_size, box_size))

    # Draw quadrants
    for i in range(0, screen_size, box_size):
        pygame.draw.line(screen, BLUE if i % 400 != 0 else BLACK, (i, 0), (i, screen_size), 2)
        pygame.draw.line(screen, BLUE if i % 400 != 0 else BLACK, (0, i), (screen_size, i), 2)

    arm = bandit.pull_arm()    
    
    # Display the agent's current quadrant selection
    quadrant_x = (arm % 2) * 400
    quadrant_y = (arm // 2) * 400
    pygame.draw.rect(screen, RED, (quadrant_x, quadrant_y, 400, 400), 5)  # Highlight the selected quadrant

    selected_box_x = quadrant_x + np.random.randint(0, 10) * box_size
    selected_box_y = quadrant_y + np.random.randint(0, 10) * box_size
    
    # Draw agent
    pygame.draw.rect(screen, RED, (selected_box_x, selected_box_y, box_size, box_size))
    reward = any(treasure[:2] == (selected_box_x, selected_box_y) for treasure in treasures if treasure[2] == arm)
    total_rewards += reward

    bandit.update(arm, reward)

    # Calculate regret for not choosing the best possible arm (in terms of probability)
    expected_best_reward = optimal_reward_probability
    expected_reward_this_round = treasure_probabilities[arm]
    regret = expected_best_reward - expected_reward_this_round
    cumulative_regrets.append(regret if len(cumulative_regrets) == 0 else cumulative_regrets[-1] + regret)    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)  # Slow down the loop for visibility

pygame.quit()

# Results and plotting
plt.plot(cumulative_regrets, label='Cumulative Regret')
plt.xlabel('Time step')
plt.ylabel('Cumulative Regret')
plt.ylim(0,200)
plt.legend()
plt.show()






