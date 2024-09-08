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
    
    def update(self, chosen_location, reward):  
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

# Create bandit for player 2
player2_bandit = BernoulliAPSBandit(4)

# Lists to store regrets
regret_player2 = []
cumulative_regret = 0
player1_action = None

# Function to draw buttons
def draw_buttons(selected_action, player):
    button_size = 50  #The width and height of each button.
    button_padding = 20  #The space between the bottom of the screen and the buttons.
    button_margin = 30  #The space between buttons.
    button_radius = 25  #The radius of the circular buttons.
    button_positions = [
        (button_margin, screen_height - button_size - button_padding),
        (2*button_margin + button_size, screen_height - button_size - button_padding),
        (3*button_margin + 2*button_size, screen_height - button_size - button_padding),
        (4*button_margin + 3*button_size, screen_height - button_size - button_padding)
    ]  #A list of tuples representing the (x, y) coordinates of each button's top-left corner.
    button_images = [attack_btn_img, defend_btn_img, gold_btn_img, special_btn_img]
     
    # Draw buttons for player 1
    if player == 1:
        for i, (x, y) in enumerate(button_positions):
            pygame.draw.circle(screen, (0, 0, 0), (x + button_radius, y + button_radius), button_radius + 3, 2)  # Border with thickness 2 and radius btn_radius+3
            if selected_action == i:
                pygame.draw.circle(screen, (255, 255, 0), (x + button_radius, y + button_radius), button_radius)  # Highlight
            screen.blit(button_images[i], (x, y))

# Function to animate attack
def animate_attack(attacker_x, attacker_y, target_x, target_y, player, rotate=False):
    frames = 30
    for i in range(frames):
        ratio = i / frames
        size = int(50 + 150 * ratio)
        # Rotate the image if needed
        if rotate and player == 2:
            rotated_attack_img = pygame.transform.rotate(attack_btn_img, 180)
            attack_img = pygame.transform.scale(rotated_attack_img, (size, size))
        else:
            attack_img = pygame.transform.scale(attack_btn_img, (size, size))
        if player == 1:
            x = int(attacker_x + (target_x - attacker_x) * ratio)
            y = int(attacker_y + (target_y - attacker_y) * ratio)
        else:
            x = int(attacker_x + (target_x - attacker_x) * (1 - ratio))
            y = int(attacker_y + (target_y - attacker_y) * (1 - ratio))
        screen.blit(attack_img, (x, y))
        pygame.display.flip()
        pygame.time.delay(20)
        screen.blit(background_img, (0, 0))
        screen.blit(character1_img, (char1_x, char1_y))
        screen.blit(character2_img, (char2_x, char2_y))
        draw_buttons(player1_action, player=1)
    
def animate_shield(player_x, player_y):
    shield_size = (150, 150)  # Adjust shield size
    shield_offset = (20, -20)  # Adjust shield offset
    shield_img = pygame.transform.scale(defend_btn_img, shield_size)
    
    # Calculate shield position
    shield_x = int(player_x + character2_img.get_width() // 2 - shield_size[0] // 2 + shield_offset[0])
    shield_y = int(player_y + character2_img.get_height() // 2 - shield_size[1] // 2 + shield_offset[1])
    
    # Draw the shield
    screen.blit(shield_img, (shield_x, shield_y))
    pygame.display.flip()
    
    # Pause for 2 seconds
    pygame.time.delay(1000)

# Function to animate special power
def animate_special_power(attacker_x, attacker_y, target_x, target_y, player):
    if player == 1:
        attacker_img = character1_img
        opponent_img = character2_img
        direction = -1  # Move left for player 1
    else:
        attacker_img = character2_img
        opponent_img = character1_img
        direction = 1   # Move right for player 2

    # Enlarge the attacker image
    size = 0
    for i in range(50):
        size += 5
        special_img = pygame.transform.scale(attacker_img, (size, size))
        x = int(attacker_x + direction * attacker_img.get_width() // 2 - size // 2)
        y = int(attacker_y + attacker_img.get_height() // 2 - size // 2)
        screen.blit(special_img, (x, y))
        pygame.display.flip()
        pygame.time.delay(20)

    # Draw the special power animation
    for i in range(30):
        ratio = i / 30
        x = int(attacker_x + direction * attacker_img.get_width() // 2 - size // 2 + (target_x - attacker_x) * ratio)
        y = int(attacker_y + attacker_img.get_height() // 2 - size // 2 + (target_y - attacker_y) * ratio)
        special_img = pygame.transform.scale(special_btn_img, (50, 50))
        screen.blit(special_img, (x, y))
        pygame.display.flip()
        pygame.time.delay(20)

    # Remove the enlarged attacker image
    screen.blit(background_img, (0, 0))
    screen.blit(opponent_img, (target_x, target_y))
    draw_buttons(player1_action, player=1)
    
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

    # Calculate the maximum possible payoff for player 1 given player 2's action
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
        play_start_music()
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
        elif event.type == pygame.KEYDOWN:            
            if event.key == pygame.K_j:
                player1_action = A_ATTACK
            elif event.key == pygame.K_k:
                player1_action = A_DEFEND
            elif event.key == pygame.K_l:
                player1_action = A_BUILD_GOLD
            elif event.key == pygame.K_i:
                if player1_gold >= 50:
                    player1_action = A_SPECIAL_POWER            

    # Clear the screen
    screen.fill((255, 255, 255))  # Fill screen with white color

    # Draw background
    screen.blit(background_img, (0, 0))

    # Draw characters on top of background
    screen.blit(character1_img, (char1_x, char1_y))
    screen.blit(character2_img, (char2_x, char2_y))

    # Draw health and gold bars for player 1
    player1_health_width = min(player1_health * (400 / 1000), 400)  # Scale health to fit within 400 pixels
    pygame.draw.rect(screen, green, (10, 10, player1_health_width, 20))
    pygame.draw.rect(screen, yellow, (10, 40, min(player1_gold * 2, 400), 20))  # Cap at 400 pixels width

    # Draw health and gold bars for player 2
    player2_health_width = min(player2_health * (400 / 1000), 400)  # Scale health to fit within 400 pixels
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

    # Draw buttons for player 1
    draw_buttons(player1_action, player=1)

    # Player 2's action selection (UCB)
    player2_action = player2_bandit.pull_arm()    
    # Execute actions
    if player1_action == A_ATTACK:
        animate_attack(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                       char2_x, char2_y + character2_img.get_height() // 2, player=1)
        
        if player2_action == A_ATTACK:
            player1_health -= 10
            player2_health -= 10
            player2_reward = 5
            animate_attack(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                       char2_x, char2_y + character2_img.get_height() // 2, player=2, rotate = True)
        elif player2_action == A_DEFEND:
            player2_health -= 5
            player2_reward = 20
            animate_shield(char2_x, char2_y)
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_reward = 10
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                animate_special_power(char2_x + character2_img.get_width(), char2_y + character2_img.get_height() // 2,
                                       char1_x, char1_y + character1_img.get_height() // 2, player=2)
                player1_health -= 20
                player2_health -= 10
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5
                player2_health -= 10
        
    elif  player1_action == A_DEFEND:
        animate_shield(char1_x, char1_y)
        if player2_action == A_ATTACK:            
            player2_reward = 5
            animate_attack(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                       char2_x, char2_y + character2_img.get_height() // 2, player=2, rotate = True)
        elif player2_action == A_DEFEND:
            player1_health -= 5
            player2_reward = 5
            animate_shield(char2_x, char2_y)
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_reward = 5
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                animate_special_power(char2_x + character2_img.get_width(), char2_y + character2_img.get_height() // 2,
                                       char1_x, char1_y + character1_img.get_height() // 2, player=2)
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
            animate_attack(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                       char2_x, char2_y + character2_img.get_height() // 2, player=2, rotate = True)             
        elif player2_action == A_DEFEND:
            player2_reward = 5
            animate_shield(char2_x, char2_y)
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_reward = 10
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                animate_special_power(char2_x + character2_img.get_width(), char2_y + character2_img.get_height() // 2,
                                       char1_x, char1_y + character1_img.get_height() // 2, player=2)
                player1_health -= 20
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5                
                
    elif player1_action == A_SPECIAL_POWER and player1_gold >= 50:
        player1_gold -= 50
        animate_special_power(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                               char2_x, char2_y + character2_img.get_height() // 2, player=1)
        if player2_action == A_ATTACK:
            player1_health -= 10    
            player2_health -= 20  
            player2_reward = 5  
            animate_attack(char1_x + character1_img.get_width(), char1_y + character1_img.get_height() // 2,
                       char2_x, char2_y + character2_img.get_height() // 2, player=2, rotate = True)                            
        elif player2_action == A_DEFEND:            
            player2_health -= 15
            player2_reward = 5
            animate_shield(char2_x, char2_y)
        elif player2_action == A_BUILD_GOLD:
            player2_gold += 10
            player2_health -= 20
            player2_reward = 5
        elif player2_action == A_SPECIAL_POWER:
            if player2_gold >= 50:
                animate_special_power(char2_x + character2_img.get_width(), char2_y + character2_img.get_height() // 2,
                                       char1_x, char1_y + character1_img.get_height() // 2, player=2)
                player1_health -= 20
                player2_health -= 20
                player2_reward = 25
                player2_gold -= 50
            else:                
                player2_reward = 5
                player2_health -= 20
    else:
        player2_reward = 0
    player1_action = None                   
    
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
plt.plot(regret_player2, label='Player 2 Regret')
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Regret of Player 2 Over Time')
plt.legend()
plt.show()

pygame.quit()
sys.exit()
