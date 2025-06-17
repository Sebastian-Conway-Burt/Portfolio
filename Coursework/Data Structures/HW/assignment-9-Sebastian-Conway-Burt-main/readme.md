# Search and Rescue

This assignment is set in the mountains after an avalanche, and you are a robot on a search and rescue mission. The objective is to get from the starting point to the evacuation area while picking up all of the people trapped on the mountain.

This mountain will be represented by a grid filled with numbers 1-3. These numbers represent the following:
```
1: a location that you can move to
2: a location that you can move to where a person is trapped
3: the evacuation area
```

The issue is that as the robot, you can only see what is in locations directly to your north, east, west, and south. You can only move in these directions as well.

# Task

All of the searching algorithms you've seen have been performed on graphs. However, in this project you will be doing a state-search. This can be conceptualized in the following manner. 

Assume each vertex in our graph represents a possible state of the world. In our case, a state would be one way the stack of pancakes could look. Now imagine we have to discover the state just-in-time. In this way, we don't actually have a graph or vertices anywhere, we simply discover the states by performing actions that would take us to another state (moving in some direction).

In your case, you will have an additional graph (the grid) representing the terrain of the mountain that will tell you where you can go.

There are two components to a state-search problem:
1. The states themselves (the location in our grid and how many people we have saved)
2. The actions available to an agent in/viewing a given state (the movements we can make)

## Operations at States

There are 3 main functions to be implemented for use at each state

### possibleActions

This function takes the current state and adds the possible actions to be taken in said state to its possibleActions vector.

```
ex.
For any location possibleActions should add:
	"up", "down", "left", "right" to the vector if you would not be moving out of bounds in those directions.
```

### result

This function takes the current state and an action and returns the resulting state. Add the action you took to get here to the state's prev_action attribute.

```
ex.
input: state with location: [1,3] & action: right
in code: result(State{1,3,...},'right')
output: State{2,3,...}
```

### expand

This function takes a state and generates all possible resultant states for all actions (should call result and possibleActions).

```
ex.
input: State with location: [1,2]
output: Vector of states with locations [1,3], [1,1], [0,2], [2,2]
```

## Auxiliary Functions

You will also need to implement the following functions:

### printPath

This function prints the states in our path vector as we iterate through them

```
ex.
cout << "x: " << <the x value> << "\ty: " << <the y value> << "\tprev_action: " << <the value of prev_action> << "\n";
```

### isGoal

This function takes a location and checks if it is equal to the evacuation area. It also checks if the number of saved people is equivalent to PEOPLE_TO_SAVE. If both of these conditions are met, return true, else return false.

## Searching: DFS/IDS

You should understand the algorithm of DFS at this point, but please look [here](https://en.wikipedia.org/wiki/Depth-first_search) if you require a reference.

In this project, however, you will be implementing a slightly modified version of DFS. This algorithm limits the depth we travel to at any given point, and if we haven't found the item we are looking for after traveling to a certain depth, we increase the depth we are allowed to travel to. This is called [Iterative Deepening Search](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search). This is guaranteed to give us the shortest path that gets us to our goal.

You will need to implement both a wrapper function and the actual search.

### iterativeDeepeningWrapper

This function takes a starting location and finds the actions that lead to the evacuation point while saving all of the trapped people. Calls iterativeDeepeningSearch and increments the depth limit if the path to the goal state could not be found. Returns true if a path is found, false if not (if there is a path, this is not possible).

### iterativeDeepeningSearch

Calls expand. This function takes the current location and a depth limit. For each recursive call, you should decrement the depth limit. If the depth limit is 0, you should simply return false. Return true only when ifGoal returns true. You should increment the state's saved_people parameter before expansion, and decrement it if you find the state was not on the correct path. Similarly, new states should be added to the path before making the recursive call and removed if they were not on the path to the goal state.

## Existing Data & Structure

There is just one class and one struct you will make use of in this assignment.

``` C++
struct State
{
	int x;
	int y;
	int saved_people;
	string prev_action;
	vector<string> possible_actions;
};

class SearchAndRescue {
	public:
		SearchAndRescue();
		~SearchAndRescue();
		void possibleActions(State*);
		State* result(State*, string);
		vector<State*> expand(State*);
		bool iterativeDeepeningWrapper(State*);
		bool iterativeDeepeningSearch(State*, int);
		void printPath();
		bool isGoal(State*);
		void printTerrain();
		
	private:
		int** terrain;
		vector<State*> path;
};
```
