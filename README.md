**Dynamic Pathfinding Visualizer**

A grid-based pathfinding visualizer implementing:

 A* (A-Star) Search
 Greedy Best-First Search (GBFS)
 Manhattan & Euclidean heuristics
 Static and Dynamic obstacles
 Real-time GUI visualization

This project demonstrates how different search algorithms behave in best-case and worst-case scenarios.

**Features**
Interactive grid environment
Start & Goal selection
Random obstacle generation
Dynamic obstacle mode (re-planning supported)
Algorithm comparison (A* vs GBFS)

Real-time metrics:
   Nodes visited
   Path length
   Execution time

**Algorithms Implemented**
**A* Search**
f(n) = g(n) + h(n)
**Greedy Best-First Search (GBFS)**
f(n) = h(n)

**Heuristics Used**
Manhattan Distance
|x1 - x2| + |y1 - y2|
Euclidean Distance
sqrt((x1 - x2)^2 + (y1 - y2)^2)

**Project Structure**
project/
│
├── app.py

**Clone the Repository**
git clone https://github.com/0687M-Abdullah/Informed-Searches.git
cd Informed-Searches

**Install Dependencies**
pip install tkinter

**Run the Project**
python app.py
