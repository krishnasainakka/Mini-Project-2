import pygame
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Set up the screen
screen_width = 850
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Dragon Ball Z")

# Load start screen background image
start_background_img = pygame.image.load("start_bg.jpg")
start_background_img = pygame.transform.scale(start_background_img, (screen_width, screen_height))

# Load background image
background_img = pygame.image.load("bg.jpeg")
background_img = pygame.transform.scale(background_img, (screen_width, screen_height))

# Load character images and scale them down
character1_img = pygame.image.load("naruto.png")
character1_img = pygame.transform.scale(character1_img, (240, 280))

character2_img = pygame.image.load("goku.png")
character2_img = pygame.transform.scale(character2_img, (160, 240))

# Load button images and scale them down
attack_btn_img = pygame.image.load("laser.png")
attack_btn_img = pygame.transform.scale(attack_btn_img, (50, 50))

defend_btn_img = pygame.image.load("shield.png")
defend_btn_img = pygame.transform.scale(defend_btn_img, (50, 50))

gold_btn_img = pygame.image.load("gold.png")
gold_btn_img = pygame.transform.scale(gold_btn_img, (50, 50))

special_btn_img = pygame.image.load("special.png")
special_btn_img = pygame.transform.scale(special_btn_img, (50, 50))

# Set initial positions for characters
char1_x = 100
char1_y = screen_height // 2 - character1_img.get_height() // 2

char2_x = screen_width - 100 - character2_img.get_width()
char2_y = screen_height // 2 - character2_img.get_height() // 2

# Load sound effects
bg_music = pygame.mixer.Sound("background_music.mp3")
start_music = pygame.mixer.Sound("start_music.mp3")

# Player attributes
player1_health = 500
player1_gold = 0

player2_health = 500
player2_gold = 0

# Define colors
green = (0, 255, 0)
yellow = (255, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
black = (0, 0, 0)

# Font
font = pygame.font.Font(None, 24)

# Define actions
A_ATTACK = 0
A_DEFEND = 1
A_BUILD_GOLD = 2
A_SPECIAL_POWER = 3

#APS
class BernoulliAPSBandit:
    def __init__(self, n_arms, eta=0.08):        
        self.n_arms = n_arms
        self.eta = eta
        self.exploration_weights = np.ones(n_arms)  # Initialize exploration weights uniformly
        
    def pull_arm(self):        
        normalized_weights = self.exploration_weights / np.sum(self.exploration_weights)
        cum_prob = np.cumsum(normalized_weights)
        rand_num = np.random.rand()
        location_index = next(i for i in range(len(cum_prob)) if cum_prob[i] > rand_num)
        return location_index
    
    def update(self, chosen_location, reward):  
        k = self.n_arms
        w_t = self.exploration_weights[chosen_location]  # Weight before update
        
        # Ensuring w_t is never exactly 1 to avoid division by zero
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

# Create bandit for player 2
player2_bandit = BernoulliAPSBandit(4)

# Lists to store regrets
regret_player2 = []
cumulative_regret = 0

# Function to draw buttons
def draw_buttons(selected_action, player):
    button_size = 50
    button_padding = 20
    button_margin = 30 
    button_radius = 25
    button_positions = [
        (button_margin, screen_height - button_size - button_padding),
        (2*button_margin + button_size, screen_height - button_size - button_padding),
        (3*button_margin + 2*button_size, screen_height - button_size - button_padding),
        (4*button_margin + 3*button_size, screen_height - button_size - button_padding)
    ]
    button_images = [attack_btn_img, defend_btn_img, gold_btn_img, special_btn_img]
    
    # Draw buttons for player 1
    if player == 1:
        for i, (x, y) in enumerate(button_positions):
            pygame.draw.circle(screen, black, (x + button_radius, y + button_radius), button_radius + 3, 2)  # Border
            if selected_action == i:
                pygame.draw.circle(screen, yellow, (x + button_radius, y + button_radius), button_radius)  # Highlight
            screen.blit(button_images[i], (x, y))

# Function to animate players entering the screen
def animate_players_entering():
    frames = 30
    for i in range(frames):
        ratio = i / frames
        char1_current_x = int(ratio * (100 + character1_img.get_width()))  # Move from -width to 100
        char2_current_x = int(screen_width - ratio * (100 + character2_img.get_width()))  # Move from width to screen_width - 100
        screen.blit(start_background_img, (0, 0))  # Draw start screen background
        screen.blit(character1_img, (char1_current_x, char1_y))  # Draw character 1 entering from left
        screen.blit(character2_img, (char2_current_x, char2_y))  # Draw character 2 entering from right
        pygame.display.flip()
        pygame.time.delay(20)

# Function to handle events on the start screen
def handle_start_screen_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse click is within the bounds of the start button
            mouse_x, mouse_y = pygame.mouse.get_pos()
            button_rect = start_button_img.get_rect(topleft=(350, 400))
            if button_rect.collidepoint(mouse_x, mouse_y):
                return True  # Start button clicked
    return False

def calculate_optimal_payoff(player1_action, player2_action, player1_health, player2_health):    
    damage = 0
    if player2_action == A_ATTACK:
        damage = 10  # Assuming attack deals 10 damage
    elif player2_action == A_SPECIAL_POWER and player2_gold >= 50:
        damage = 20

    if player1_action == A_DEFEND:        
        damage = max(0, damage - 5)  # Defend reduces attack damage by 5

    return max(0, player1_health - damage )

def calculate_regret(player1_action, player2_action, player1_health, player2_health):
    # Calculate the payoff for player 1's action against each possible action of player 2
    player1_payoffs = []
    for action in [A_ATTACK, A_DEFEND, A_BUILD_GOLD, A_SPECIAL_POWER]:
        player1_payoff = calculate_optimal_payoff(player1_action, action, player1_health, player2_health)
        player1_payoffs.append(player1_payoff)

    # Calculate the minimum possible payoff for player 1 given player 2's action
    min_player1_payoff = min(player1_payoffs)

    # Calculate the payoff for player 2's actual action against player 1's actual action
    actual_player1_payoff = calculate_optimal_payoff(player1_action, player2_action, player1_health, player2_health)

    # Calculate the regret
    regret = actual_player1_payoff - min_player1_payoff
    return regret

# Function to play start music
def play_start_music():
    bg_music.stop()
    start_music.play()
        
# Load start button image and scale it down
start_button_img = pygame.image.load("start_btn.png")
start_button_img = pygame.transform.scale(start_button_img, (300, 200))

# Main loop for start screen
while True:
    bg_music.play()
    if handle_start_screen_events():
        play_start_music()  # Start button clicked
        break  # Break out of the loop when the start button is clicked
    
    # Draw start screen background
    screen.blit(start_background_img, (0, 0))
    
    # Draw start button
    screen.blit(start_button_img, (250, 400))
    
    # Update display
    pygame.display.flip()

# Animate players entering the screen
animate_players_entering()

# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()        

    # Clear the screen
    screen.fill((255, 255, 255))  # Fill screen with white color

    # Draw background
    screen.blit(background_img, (0, 0))

    # Draw characters on top of background
    screen.blit(character1_img, (char1_x, char1_y))
    screen.blit(character2_img, (char2_x, char2_y))

    # Draw health and gold bars for player 1
    player1_health_width = min(player1_health * (400 / 500), 400)  # Scale health to fit within 400 pixels
    pygame.draw.rect(screen, green, (10, 10, player1_health_width, 20))
    pygame.draw.rect(screen, yellow, (10, 40, min(player1_gold * 2, 400), 20))  # Cap at 400 pixels width

    # Draw health and gold bars for player 2
    player2_health_width = min(player2_health * (400 / 500), 400)  # Scale health to fit within 400 pixels
    pygame.draw.rect(screen, green, (screen_width - player2_health_width - 10, 10, player2_health_width, 20))
    pygame.draw.rect(screen, yellow, (screen_width - min(player2_gold * 2, 400) - 10, 40, min(player2_gold * 2, 400), 20))  # Cap at 400 pixels width

    # Render text
    player1_health_text = font.render("Health: " + str(player1_health), True, black)
    player1_gold_text = font.render("Gold: " + str(player1_gold), True, black)
    player2_health_text = font.render("Health: " + str(player2_health), True, black)
    player2_gold_text = font.render("Gold: " + str(player2_gold), True, black)

    # Draw text on screen
    screen.blit(player1_health_text, (10, 70))
    screen.blit(player1_gold_text, (10, 100))
    screen.blit(player2_health_text, (screen_width - player2_health_text.get_width() - 10, 70))
    screen.blit(player2_gold_text, (screen_width - player2_gold_text.get_width() - 10, 100))

    # Player 1's action selection (random)    
    while(1):
        player1_action = random.choice([A_ATTACK, A_DEFEND, A_BUILD_GOLD, A_SPECIAL_POWER])
        if(player1_action == A_SPECIAL_POWER and player1_gold < 50):
            pass
        else:
            break
    
    # Draw buttons for player 1
    draw_buttons(player1_action, player=1)

    # Player 2's action selection (UCB)
    player2_action = player2_bandit.pull_arm()

    # Execute actions
    if player1_action == A_ATTACK:
        if player2_action == A_ATTACK:
            player1_health -= 10
            player2_health -= 10
            player2_reward = 5
        elif player2_action == A_DEFEND:            
            player2_health -= 5
            player2_reward = 20
        elif player2_action == A_BUILD_GOLD:
            player2_health -= 10
            player2_gold += 10
            player2_reward = 10
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                player1_health -= 20
                player2_health -= 10
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_health -= 10
                player2_reward = 5
                
    elif  player1_action == A_DEFEND:
        if player2_action == A_ATTACK:
            player1_health -= 5
            player2_reward = 5
        elif player2_action == A_DEFEND:
            player2_health -= 0
            player2_reward = 5
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_reward = 5
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                player1_health -= 15
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5                
                
    elif  player1_action == A_BUILD_GOLD:
        player1_gold += 10
        if player2_action == A_ATTACK:
            player1_health -= 10    
            player2_reward = 15                
        elif player2_action == A_DEFEND:
            player2_health -= 0
            player2_reward = 5
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_reward = 10
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                player1_health -= 20
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5                
                
    elif  player1_action == A_SPECIAL_POWER and player1_gold >= 50: 
        player1_gold -= 50               
        if player2_action == A_ATTACK:
            player1_health -= 10    
            player2_health -= 20  
            player2_reward = 5                              
        elif player2_action == A_DEFEND:            
            player2_health -= 15
            player2_reward = 5
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_health -= 20
            player2_reward = 5
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                player1_health -= 20
                player2_health -= 20
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5
                player2_health -= 20
    else:
        player2_reward = 0
    
    # Calculate regret for player 2
    # actual_action_payoff = max(0, player1_health - player2_health)
    # optimal_action_payoff = max(0, (player1_health - (player2_action == A_ATTACK) * 10) - player2_health)
    # regret = optimal_action_payoff - actual_action_payoff
    # cumulative_regret += regret
    # regret_player2.append(cumulative_regret)
    
    regret = calculate_regret(player1_action, player2_action, player1_health, player2_health)
    cumulative_regret += regret 
    regret_player2.append(cumulative_regret)

    # Update bandit for player 2 with reward
    player2_bandit.update(player2_action, player2_reward)

    # Check if game should end
    if player1_health <= 0 or player2_health <= 0:
        break

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

# Plot regret for player 2
print(regret_player2[-1])
plt.plot(regret_player2, label='Player 2 Regret')
plt.xlabel('Time')
plt.ylabel('Regret')
plt.ylim(0,2000)
plt.title('Regret of Player 2 Over Time')
plt.legend()
plt.show()

pygame.quit()
sys.exit()
