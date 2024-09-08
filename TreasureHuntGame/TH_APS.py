import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

pygame.init()

# Screen setup
screen_size = 800
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Treasure Hunting with APS")

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

class BernoulliAPSBandit:
    def __init__(self, n_arms, eta=0.05):        
        self.n_arms = n_arms
        self.eta = eta
        self.exploration_weights = np.ones(n_arms)  # Initialize exploration weights uniformly
        
    def pull_arm(self):        
        normalized_weights = self.exploration_weights / np.sum(self.exploration_weights)
        cum_prob = np.cumsum(normalized_weights)
        rand_num = np.random.rand()
        location_index = next(i for i in range(len(cum_prob)) if cum_prob[i] > rand_num)
        return location_index
    
    def update_exploration_weights(self, chosen_location, reward):  # Ensure arguments are in the correct order
        k = self.n_arms
        w_t = self.exploration_weights[chosen_location]  # Weight before update
        
        # Ensure w_t is never exactly 1 to avoid division by zero
        w_t = np.clip(w_t, None, 0.9999)

        if reward == 1:
            self.exploration_weights[chosen_location] = (1 - np.exp(-self.eta)) / (1 - np.exp(-self.eta / w_t))
        else:
            self.exploration_weights[chosen_location] = (np.exp(self.eta) - 1) / (np.exp(self.eta / w_t) - 1)
        
        # Adjust other weights
        for i in range(k):
            if i != chosen_location:
                # Protect against division by zero or undefined operations
                try:
                    adjustment = ((1 - self.exploration_weights[chosen_location]) / max(1 - w_t, 0.0001))
                    self.exploration_weights[i] = np.clip(self.exploration_weights[i] * adjustment, 0.001, None)
                except RuntimeWarning as e:
                    print(f"Warning: {e}")


    
    # def update_exploration_weights(self, reward, chosen_location):        
    #     k = self.n_arms
    #     w_t = self.exploration_weights[chosen_location]  # Weight before update

    #     if reward == 1:
    #         self.exploration_weights[chosen_location] = (1 - np.exp(-self.eta)) / (1 - np.exp(-self.eta / w_t))
    #     else:
    #         self.exploration_weights[chosen_location] = (np.exp(self.eta) - 1) / (np.exp(self.eta / w_t) - 1)
        
    #     for i in range(k):
    #         if i != chosen_location:
    #             self.exploration_weights[i] = max(self.exploration_weights[i] * ((1 - self.exploration_weights[chosen_location]) / (1 - w_t)), 0.001)        

# Example of usage
n_arms = 4  # Suppose there are 4 locations
bandit = BernoulliAPSBandit(n_arms)
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

    bandit.update_exploration_weights(arm, reward)

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

