import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import time
from collections import deque
import heapq
from typing import List, Tuple, Set, Optional
import threading

# ============================================================================
# NODE AND GRID CLASSES
# ============================================================================

class Node:
    """Represents a cell in the grid."""
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.is_obstacle = False
        self.g = float('inf')  # Cost from start
        self.h = 0  # Heuristic cost to goal
        self.f = float('inf')  # Total cost
        self.parent = None
        self.in_frontier = False
        self.visited = False
    
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.f < other.f
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def reset(self):
        """Reset node for new search."""
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.in_frontier = False
        self.visited = False


class Grid:
    """Manages the grid and obstacle placement."""
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.nodes = [[Node(i, j) for j in range(cols)] for i in range(rows)]
        self.start_node = None
        self.goal_node = None
    
    def get_node(self, row: int, col: int) -> Optional[Node]:
        """Get node at position, return None if out of bounds."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.nodes[row][col]
        return None
    
    def set_obstacle(self, row: int, col: int, is_obstacle: bool):
        """Set obstacle status at position."""
        node = self.get_node(row, col)
        if node and (node != self.start_node and node != self.goal_node):
            node.is_obstacle = is_obstacle
    
    def get_neighbors(self, node: Node) -> List[Node]:
        """Get valid neighbors (4-directional movement)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            neighbor = self.get_node(node.row + dr, node.col + dc)
            if neighbor and not neighbor.is_obstacle:
                neighbors.append(neighbor)
        
        return neighbors
    
    def generate_random_obstacles(self, density: float):
        """Generate random obstacles with given density (0.0 to 1.0)."""
        for i in range(self.rows):
            for j in range(self.cols):
                node = self.nodes[i][j]
                if node != self.start_node and node != self.goal_node:
                    node.is_obstacle = random.random() < density
    
    def reset_search_state(self):
        """Reset all nodes for new search."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.nodes[i][j].reset()
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.nodes[i][j] != self.start_node and self.nodes[i][j] != self.goal_node:
                    self.nodes[i][j].is_obstacle = False


# ============================================================================
# HEURISTIC FUNCTIONS
# ============================================================================

class Heuristics:
    """Collection of heuristic functions."""
    
    @staticmethod
    def manhattan(node: Node, goal: Node) -> float:
        """Manhattan distance heuristic."""
        return abs(node.row - goal.row) + abs(node.col - goal.col)
    
    @staticmethod
    def euclidean(node: Node, goal: Node) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((node.row - goal.row)**2 + (node.col - goal.col)**2)


# ============================================================================
# SEARCH ALGORITHMS
# ============================================================================

class PathfindingAlgorithm:
    """Base class for pathfinding algorithms."""
    
    def __init__(self, grid: Grid, heuristic_func):
        self.grid = grid
        self.heuristic_func = heuristic_func
        self.nodes_visited = 0
        self.execution_time = 0
        self.frontier = []
        self.closed_set: Set[Node] = set()
    
    def search(self) -> Optional[List[Node]]:
        """Execute search algorithm."""
        raise NotImplementedError
    
    def reconstruct_path(self, node: Node) -> List[Node]:
        """Reconstruct path from start to node."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]


class AStarSearch(PathfindingAlgorithm):
    """A* search algorithm implementation."""
    
    def search(self) -> Optional[List[Node]]:
        """Execute A* search."""
        start_time = time.time()
        self.nodes_visited = 0
        
        self.grid.reset_search_state()
        start = self.grid.start_node
        goal = self.grid.goal_node
        
        self.frontier = []
        self.closed_set = set()
        
        start.g = 0
        start.h = self.heuristic_func(start, goal)
        start.f = start.g + start.h
        start.in_frontier = True
        
        heapq.heappush(self.frontier, (start.f, id(start), start))
        
        while self.frontier:
            _, _, current = heapq.heappop(self.frontier)
            current.in_frontier = False
            
            if current == goal:
                self.execution_time = (time.time() - start_time) * 1000
                return self.reconstruct_path(current)
            
            if current in self.closed_set:
                continue
            
            self.closed_set.add(current)
            current.visited = True
            self.nodes_visited += 1
            
            for neighbor in self.grid.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue
                
                tentative_g = current.g + 1
                
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic_func(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    if not neighbor.in_frontier:
                        neighbor.in_frontier = True
                        heapq.heappush(self.frontier, (neighbor.f, id(neighbor), neighbor))
        
        self.execution_time = (time.time() - start_time) * 1000
        return None


class GreedyBestFirstSearch(PathfindingAlgorithm):
    """Greedy Best-First search algorithm implementation."""
    
    def search(self) -> Optional[List[Node]]:
        """Execute Greedy Best-First search."""
        start_time = time.time()
        self.nodes_visited = 0
        
        self.grid.reset_search_state()
        start = self.grid.start_node
        goal = self.grid.goal_node
        
        self.frontier = []
        self.closed_set = set()
        
        start.h = self.heuristic_func(start, goal)
        start.f = start.h
        start.in_frontier = True
        
        heapq.heappush(self.frontier, (start.f, id(start), start))
        
        while self.frontier:
            _, _, current = heapq.heappop(self.frontier)
            current.in_frontier = False
            
            if current == goal:
                self.execution_time = (time.time() - start_time) * 1000
                return self.reconstruct_path(current)
            
            if current in self.closed_set:
                continue
            
            self.closed_set.add(current)
            current.visited = True
            self.nodes_visited += 1
            
            for neighbor in self.grid.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue
                
                if not neighbor.in_frontier:
                    neighbor.parent = current
                    neighbor.h = self.heuristic_func(neighbor, goal)
                    neighbor.f = neighbor.h
                    neighbor.in_frontier = True
                    heapq.heappush(self.frontier, (neighbor.f, id(neighbor), neighbor))
        
        self.execution_time = (time.time() - start_time) * 1000
        return None


# ============================================================================
# GUI APPLICATION
# ============================================================================

class PathfindingGUI:
    """Tkinter GUI for the pathfinding agent."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent")
        self.root.geometry("1400x800")
        
        # Configuration variables
        self.grid_rows = 20
        self.grid_cols = 20
        self.cell_size = 30
        self.grid = Grid(self.grid_rows, self.grid_cols)
        
        # Algorithm and heuristic selection
        self.algorithm_var = tk.StringVar(value="A*")
        self.heuristic_var = tk.StringVar(value="Manhattan")
        self.obstacle_density = tk.DoubleVar(value=0.3)
        self.dynamic_mode = tk.BooleanVar(value=False)
        self.spawn_probability = tk.DoubleVar(value=0.02)
        
        # Search results
        self.path = None
        self.current_position = None
        self.searching = False
        self.is_animating = False
        self.stop_animation = False
        
        # Build GUI
        self.build_layout()
        
        # Set initial start and goal
        self.grid.start_node = self.grid.get_node(1, 1)
        self.grid.goal_node = self.grid.get_node(self.grid_rows - 2, self.grid_cols - 2)
        self.current_position = self.grid.start_node
    
    def build_layout(self):
        """Build the GUI layout."""
        # Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Grid Configuration
        ttk.Label(control_frame, text="Grid Configuration", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(control_frame, text="Rows:").pack(anchor=tk.W)
        rows_spinbox = ttk.Spinbox(control_frame, from_=5, to=50, textvariable=tk.IntVar(value=self.grid_rows))
        rows_spinbox.pack(anchor=tk.W, fill=tk.X)
        
        ttk.Label(control_frame, text="Columns:").pack(anchor=tk.W)
        cols_spinbox = ttk.Spinbox(control_frame, from_=5, to=50, textvariable=tk.IntVar(value=self.grid_cols))
        cols_spinbox.pack(anchor=tk.W, fill=tk.X)
        
        ttk.Button(control_frame, text="Create Grid", command=self.create_grid).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Algorithm Selection
        ttk.Label(control_frame, text="Algorithm", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        for algo in ["A*", "GBFS"]:
            ttk.Radiobutton(control_frame, text=algo, variable=self.algorithm_var, value=algo).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Heuristic Selection
        ttk.Label(control_frame, text="Heuristic", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        for heuristic in ["Manhattan", "Euclidean"]:
            ttk.Radiobutton(control_frame, text=heuristic, variable=self.heuristic_var, value=heuristic).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Obstacle Generation
        ttk.Label(control_frame, text="Obstacle Generation", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(control_frame, text="Density (0-1):").pack(anchor=tk.W)
        ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.obstacle_density).pack(fill=tk.X)
        ttk.Button(control_frame, text="Generate Random Obstacles", command=self.generate_obstacles).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Dynamic Mode
        ttk.Label(control_frame, text="Dynamic Mode", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Enable Dynamic Obstacles", variable=self.dynamic_mode).pack(anchor=tk.W)
        ttk.Label(control_frame, text="Spawn Probability:").pack(anchor=tk.W)
        ttk.Scale(control_frame, from_=0, to=0.1, orient=tk.HORIZONTAL, variable=self.spawn_probability).pack(fill=tk.X)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Action Buttons
        ttk.Button(control_frame, text="Start Search", command=self.start_search).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop Animation", command=self.stop_animation_request).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Instructions
        ttk.Label(control_frame, text="Instructions", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        instructions = "Click: Add obstacle\nCtrl+Click: Remove obstacle\nShift+Click: Set start\nAlt+Click: Set goal"
        ttk.Label(control_frame, text=instructions, justify=tk.LEFT, font=("Arial", 9)).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Metrics Display
        ttk.Label(control_frame, text="Metrics", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.metrics_frame = ttk.Frame(control_frame)
        self.metrics_frame.pack(fill=tk.X)
        
        self.nodes_visited_label = ttk.Label(self.metrics_frame, text="Nodes Visited: 0")
        self.nodes_visited_label.pack(anchor=tk.W)
        
        self.path_cost_label = ttk.Label(self.metrics_frame, text="Path Cost: 0")
        self.path_cost_label.pack(anchor=tk.W)
        
        self.execution_time_label = ttk.Label(self.metrics_frame, text="Execution Time: 0 ms")
        self.execution_time_label.pack(anchor=tk.W)
        
        # Canvas for grid visualization
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(canvas_frame, text="Grid Visualization", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Control-Button-1>", self.on_remove_obstacle_click)
        self.canvas.bind("<Shift-Button-1>", self.on_set_start_click)
        self.canvas.bind("<Alt-Button-1>", self.on_set_goal_click)
    
    def create_grid(self):
        """Create a new grid."""
        self.grid = Grid(self.grid_rows, self.grid_cols)
        self.grid.start_node = self.grid.get_node(1, 1)
        self.grid.goal_node = self.grid.get_node(self.grid_rows - 2, self.grid_cols - 2)
        self.current_position = self.grid.start_node
        self.path = None
        self.draw_grid()
    
    def generate_obstacles(self):
        """Generate random obstacles."""
        self.grid.generate_random_obstacles(self.obstacle_density.get())
        self.draw_grid()
    
    def clear_all(self):
        """Clear all obstacles and reset."""
        self.grid.clear_obstacles()
        self.grid.reset_search_state()
        self.path = None
        self.current_position = self.grid.start_node
        self.stop_animation = True
        self.draw_grid()
    
    def stop_animation_request(self):
        """Stop the animation."""
        self.stop_animation = True
    
    def on_canvas_click(self, event):
        """Handle canvas click to add obstacles."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        self.grid.set_obstacle(row, col, True)
        self.draw_grid()
    
    def on_remove_obstacle_click(self, event):
        """Handle Ctrl+Click to remove obstacles."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        self.grid.set_obstacle(row, col, False)
        self.draw_grid()
    
    def on_set_start_click(self, event):
        """Handle Shift+Click to set start node."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        node = self.grid.get_node(row, col)
        if node and not node.is_obstacle:
            self.grid.start_node = node
            self.current_position = node
            self.draw_grid()
    
    def on_set_goal_click(self, event):
        """Handle Alt+Click to set goal node."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        node = self.grid.get_node(row, col)
        if node and not node.is_obstacle:
            self.grid.goal_node = node
            self.draw_grid()
    
    def start_search(self):
        """Start the search algorithm in a separate thread."""
        if self.searching:
            messagebox.showwarning("Warning", "Search already in progress")
            return
        
        if self.grid.start_node.is_obstacle or self.grid.goal_node.is_obstacle:
            messagebox.showerror("Error", "Start or goal is on an obstacle")
            return
        
        self.searching = True
        self.stop_animation = False
        thread = threading.Thread(target=self._perform_search)
        thread.daemon = True
        thread.start()
    
    def _perform_search(self):
        """Perform the search algorithm."""
        try:
            # Get selected heuristic function
            heuristic_map = {
                "Manhattan": Heuristics.manhattan,
                "Euclidean": Heuristics.euclidean,
            }
            heuristic = heuristic_map[self.heuristic_var.get()]
            
            # Get selected algorithm
            if self.algorithm_var.get() == "A*":
                algo = AStarSearch(self.grid, heuristic)
            else:
                algo = GreedyBestFirstSearch(self.grid, heuristic)
            
            # Perform search
            self.path = algo.search()
            
            # Update metrics
            self.nodes_visited_label.config(text=f"Nodes Visited: {algo.nodes_visited}")
            self.execution_time_label.config(text=f"Execution Time: {algo.execution_time:.2f} ms")
            
            if self.path:
                path_cost = len(self.path) - 1
                self.path_cost_label.config(text=f"Path Cost: {path_cost}")
                
                if self.dynamic_mode.get():
                    self.animate_with_dynamic_obstacles()
                else:
                    self.animate_path()
            else:
                messagebox.showinfo("Info", "No path found!")
                self.draw_grid()
            
        finally:
            self.searching = False
    
    def animate_path(self):
        """Animate the path movement."""
        if not self.path or len(self.path) < 2:
            self.draw_grid()
            return
        
        self.is_animating = True
        
        for i, node in enumerate(self.path):
            if self.stop_animation:
                break
            
            self.current_position = node
            self.draw_grid()
            self.root.update()
            time.sleep(0.1)
        
        self.is_animating = False
        self.draw_grid()
    
    def animate_with_dynamic_obstacles(self):
        """Animate path with dynamic obstacles and re-planning."""
        if not self.path or len(self.path) < 2:
            return
        
        self.is_animating = True
        current_path = self.path[:]
        path_index = 0
        
        while path_index < len(current_path) and not self.stop_animation:
            # Move agent
            self.current_position = current_path[path_index]
            
            # Spawn new obstacles
            if random.random() < self.spawn_probability.get():
                row = random.randint(0, self.grid.rows - 1)
                col = random.randint(0, self.grid.cols - 1)
                node = self.grid.get_node(row, col)
                if node and node != self.grid.start_node and node != self.grid.goal_node:
                    node.is_obstacle = True
                    
                    # Check if path is obstructed
                    if path_index < len(current_path) and current_path[path_index + 1].is_obstacle:
                        # Re-plan
                        heuristic_map = {
                            "Manhattan": Heuristics.manhattan,
                            "Euclidean": Heuristics.euclidean,
                        }
                        heuristic = heuristic_map[self.heuristic_var.get()]
                        
                        if self.algorithm_var.get() == "A*":
                            algo = AStarSearch(self.grid, heuristic)
                        else:
                            algo = GreedyBestFirstSearch(self.grid, heuristic)
                        
                        # Temporarily set current position as start
                        old_start = self.grid.start_node
                        self.grid.start_node = self.current_position
                        
                        new_path = algo.search()
                        
                        self.grid.start_node = old_start
                        
                        if new_path:
                            current_path = new_path
                            path_index = 0
            
            self.draw_grid()
            self.root.update()
            time.sleep(0.1)
            path_index += 1
        
        self.is_animating = False
        self.draw_grid()
    
    def draw_grid(self):
        """Draw the grid on canvas."""
        self.canvas.delete("all")
        
        # Calculate canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.canvas.after(100, self.draw_grid)
            return
        
        # Recalculate cell size to fit canvas
        self.cell_size = min(
            canvas_width // self.grid.cols,
            canvas_height // self.grid.rows
        )
        
        # Draw cells
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                node = self.grid.nodes[i][j]
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Determine color
                color = "white"
                
                if node.is_obstacle:
                    color = "black"
                elif node == self.grid.start_node:
                    color = "green"
                elif node == self.grid.goal_node:
                    color = "red"
                elif node == self.current_position and node != self.grid.start_node:
                    color = "cyan"
                elif self.path and node in self.path:
                    color = "lime"
                elif node.visited:
                    color = "lightblue"
                elif node.in_frontier:
                    color = "yellow"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
        
        self.canvas.update_idletasks()


def main():
    """Main entry point."""
    root = tk.Tk()
    gui = PathfindingGUI(root)
    gui.draw_grid()
    root.mainloop()


if __name__ == "__main__":
    main()