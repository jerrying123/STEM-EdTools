"""
GridWorld Treasure Hunt Environment

A custom environment where an agent navigates a grid world to collect treasures
while avoiding obstacles and managing limited energy. This environment demonstrates:
- Multi-objective rewards (treasures, energy efficiency, time)
- Dynamic obstacles
- Resource management (energy)
- Sparse rewards with exploration challenges
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap
import random

class GridWorldTreasureEnv(gym.Env):
    """
    GridWorld Treasure Hunt Environment
    
    The agent must navigate a grid to collect treasures while managing energy.
    
    Observation Space:
        - Agent position (x, y)
        - Energy level (0-100)
        - Treasure locations (binary grid)
        - Obstacle locations (binary grid)
        - Steps remaining
    
    Action Space:
        - 0: Move Up
        - 1: Move Right  
        - 2: Move Down
        - 3: Move Left
        - 4: Stay (rest to recover energy)
    
    Rewards:
        - +100 for collecting a treasure
        - -1 for each step (time penalty)
        - -5 for hitting an obstacle
        - -10 for running out of energy
        - +50 bonus for collecting all treasures
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'matplotlib'],
        'render_fps': 4
    }
    
    def __init__(self, 
                 grid_size=8,
                 num_treasures=3,
                 num_obstacles=6,
                 max_energy=100,
                 max_steps=200,
                 energy_cost_move=2,
                 energy_cost_stay=0,
                 energy_recovery_rate=5,
                 render_mode=None):
        """
        Initialize the GridWorld Treasure Hunt environment.
        
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            num_treasures: Number of treasures to place
            num_obstacles: Number of obstacles to place
            max_energy: Maximum energy level
            max_steps: Maximum steps per episode
            energy_cost_move: Energy cost for moving
            energy_cost_stay: Energy cost for staying
            energy_recovery_rate: Energy recovered when staying
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_treasures = num_treasures
        self.num_obstacles = num_obstacles
        self.max_energy = max_energy
        self.max_steps = max_steps
        self.energy_cost_move = energy_cost_move
        self.energy_cost_stay = energy_cost_stay
        self.energy_recovery_rate = energy_recovery_rate
        self.render_mode = render_mode
        
        # Action space: Up, Right, Down, Left, Stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [agent_x, agent_y, energy, steps_remaining, treasure_grid_flat, obstacle_grid_flat]
        obs_size = 4 + (grid_size * grid_size * 2)  # position + energy + steps + treasure grid + obstacle grid
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size, max_energy, max_steps),
            shape=(obs_size,), dtype=np.float32
        )
        
        # Action mappings
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 0)    # Stay
        }
        
        # Initialize state
        self.reset()
        
        # Rendering
        self.screen = None
        self.clock = None
        self.cell_size = 60
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            random.seed(seed)
        
        # Initialize grids
        self.treasure_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.collected_treasures = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Place agent at random position
        self.agent_pos = [
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size)
        ]
        
        # Place treasures randomly (not on agent)
        treasure_positions = []
        for _ in range(self.num_treasures):
            while True:
                pos = [
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                ]
                if pos != self.agent_pos and pos not in treasure_positions:
                    treasure_positions.append(pos)
                    self.treasure_grid[pos[0], pos[1]] = True
                    break
        
        # Place obstacles randomly (not on agent or treasures)
        obstacle_positions = []
        for _ in range(self.num_obstacles):
            while True:
                pos = [
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                ]
                if (pos != self.agent_pos and 
                    pos not in treasure_positions and 
                    pos not in obstacle_positions):
                    obstacle_positions.append(pos)
                    self.obstacle_grid[pos[0], pos[1]] = True
                    break
        
        # Initialize state
        self.energy = self.max_energy
        self.steps_taken = 0
        self.treasures_collected = 0
        self.total_reward = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = 0
        terminated = False
        truncated = False
        
        # Get action direction
        dx, dy = self.action_to_direction[action]
        
        # Calculate new position
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Check bounds
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            # Check for obstacles
            if self.obstacle_grid[new_x, new_y]:
                # Hit obstacle - don't move, penalty, lose energy
                reward -= 5
                self.energy -= self.energy_cost_move
            else:
                # Valid move
                self.agent_pos = [new_x, new_y]
                
                # Check for treasure
                if (self.treasure_grid[new_x, new_y] and 
                    not self.collected_treasures[new_x, new_y]):
                    reward += 100
                    self.collected_treasures[new_x, new_y] = True
                    self.treasures_collected += 1
                
                # Energy cost for moving
                if action != 4:  # Not staying
                    self.energy -= self.energy_cost_move
                else:
                    # Staying - recover energy
                    self.energy = min(self.max_energy, 
                                    self.energy + self.energy_recovery_rate - self.energy_cost_stay)
        else:
            # Tried to move out of bounds - penalty
            reward -= 2
            self.energy -= self.energy_cost_move
        
        # Time penalty
        reward -= 1
        self.steps_taken += 1
        
        # Check termination conditions
        if self.energy <= 0:
            reward -= 10
            terminated = True
        elif self.treasures_collected == self.num_treasures:
            reward += 50  # Bonus for collecting all treasures
            terminated = True
        elif self.steps_taken >= self.max_steps:
            truncated = True
        
        self.total_reward += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        """Get current observation."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Agent position
        obs[0] = self.agent_pos[0]
        obs[1] = self.agent_pos[1]
        
        # Energy and steps
        obs[2] = self.energy
        obs[3] = self.max_steps - self.steps_taken
        
        # Treasure grid (remaining treasures)
        remaining_treasures = self.treasure_grid & ~self.collected_treasures
        obs[4:4 + self.grid_size**2] = remaining_treasures.flatten()
        
        # Obstacle grid
        obs[4 + self.grid_size**2:] = self.obstacle_grid.flatten()
        
        return obs
    
    def _get_info(self):
        """Get additional info."""
        return {
            'agent_pos': self.agent_pos.copy(),
            'energy': self.energy,
            'treasures_collected': self.treasures_collected,
            'total_treasures': self.num_treasures,
            'steps_taken': self.steps_taken,
            'total_reward': self.total_reward
        }
    
    def render(self, mode=None):
        """Render the environment."""
        if mode is None:
            mode = self.render_mode
        
        if mode == 'matplotlib':
            return self._render_matplotlib()
        elif mode in ['human', 'rgb_array']:
            return self._render_pygame(mode)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_matplotlib(self):
        """Render using matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create color map
        colors = ['white', 'brown', 'gold', 'red', 'blue', 'green']
        cmap = ListedColormap(colors)
        
        # Create display grid
        display_grid = np.zeros((self.grid_size, self.grid_size))
        
        # Add obstacles (1)
        display_grid[self.obstacle_grid] = 1
        
        # Add remaining treasures (2)
        remaining_treasures = self.treasure_grid & ~self.collected_treasures
        display_grid[remaining_treasures] = 2
        
        # Add collected treasures (3)
        display_grid[self.collected_treasures] = 3
        
        # Add agent (4)
        display_grid[self.agent_pos[0], self.agent_pos[1]] = 4
        
        # Display grid
        ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=5)
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # Add text information
        info_text = f"Energy: {self.energy}/{self.max_energy}\n"
        info_text += f"Treasures: {self.treasures_collected}/{self.num_treasures}\n"
        info_text += f"Steps: {self.steps_taken}/{self.max_steps}\n"
        info_text += f"Reward: {self.total_reward:.1f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='brown', label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gold', label='Treasure'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Collected'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', label='Agent')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        ax.set_title('GridWorld Treasure Hunt')
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def _render_pygame(self, mode):
        """Render using pygame."""
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for pygame rendering. Install with: pip install pygame")
        
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100)
            )
            pygame.display.set_caption("GridWorld Treasure Hunt")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Create surface
        if mode == 'rgb_array':
            canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size + 100))
        else:
            canvas = self.screen
        
        # Fill background
        canvas.fill((255, 255, 255))  # White
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size, x * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                # Draw cell background
                if self.obstacle_grid[x, y]:
                    pygame.draw.rect(canvas, (139, 69, 19), rect)  # Brown
                elif self.collected_treasures[x, y]:
                    pygame.draw.rect(canvas, (255, 0, 0), rect)  # Red
                elif self.treasure_grid[x, y]:
                    pygame.draw.rect(canvas, (255, 215, 0), rect)  # Gold
                else:
                    pygame.draw.rect(canvas, (255, 255, 255), rect)  # White
                
                # Draw grid lines
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)
        
        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.cell_size + 5,
            self.agent_pos[0] * self.cell_size + 5,
            self.cell_size - 10, self.cell_size - 10
        )
        pygame.draw.ellipse(canvas, (0, 0, 255), agent_rect)  # Blue
        
        # Draw info panel
        info_y = self.grid_size * self.cell_size + 10
        font = pygame.font.Font(None, 24)
        
        info_texts = [
            f"Energy: {self.energy}/{self.max_energy}",
            f"Treasures: {self.treasures_collected}/{self.num_treasures}",
            f"Steps: {self.steps_taken}/{self.max_steps}",
            f"Reward: {self.total_reward:.1f}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (10, info_y + i * 20))
        
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment."""
        if PYGAME_AVAILABLE and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# Register the environment
def register_gridworld_treasure():
    """Register the GridWorld Treasure environment with Gymnasium."""
    try:
        gym.register(
            id='GridWorldTreasure-v0',
            entry_point='rled.environments.gridworld_treasure:GridWorldTreasureEnv',
            max_episode_steps=200,
            kwargs={
                'grid_size': 8,
                'num_treasures': 3,
                'num_obstacles': 6,
                'max_energy': 100,
                'max_steps': 200
            }
        )
        print("✅ GridWorldTreasure-v0 environment registered successfully!")
    except gym.error.Error as e:
        if "already registered" in str(e):
            print("ℹ️  GridWorldTreasure-v0 environment already registered")
        else:
            print(f"❌ Error registering environment: {e}")

if __name__ == "__main__":
    # Test the environment
    print("Testing GridWorld Treasure Hunt Environment")
    print("=" * 50)
    
    # Create environment
    env = GridWorldTreasureEnv(grid_size=6, num_treasures=2, num_obstacles=4)
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.1f}, "
              f"Energy={info['energy']}, Treasures={info['treasures_collected']}")
        
        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print(f"Total reward: {total_reward:.1f}")
    
    # Test rendering
    try:
        fig = env.render('matplotlib')
        print("✅ Matplotlib rendering works!")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Matplotlib rendering error: {e}")
    
    env.close()
    print("Environment test completed!") 