#ifndef SEARCH_RESCUE_H
#define SEARCH_RESCUE_H
#define STARTING_DEPTH 2
#define PEOPLE_TO_SAVE 4
#include <vector>
#include <iostream>
#include <string>
using namespace std;

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
		SearchAndRescue(string);
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

#endif