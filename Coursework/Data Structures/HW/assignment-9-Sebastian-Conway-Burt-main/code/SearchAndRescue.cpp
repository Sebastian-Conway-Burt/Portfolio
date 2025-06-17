#include "SearchAndRescue.hpp"
#include <vector>
#include <stack>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;


SearchAndRescue::SearchAndRescue(string fileName)
{
    terrain = new int*[4];
    string line;
    ifstream infile(fileName);
    if (infile.is_open())
    {
        for(int i = 0; getline(infile, line); i++)
        {   
            terrain[i] = new int[4];
            stringstream ss(line);
            string s;
            for (int j = 0; getline(ss, s, ' '); j++)
            {
                terrain[i][j] = stoi(s);
            }
        }
    }
    path = vector<State*>();
}


SearchAndRescue::~SearchAndRescue()
{
    State* to_del = path.back();
    delete to_del;
    path.pop_back();

    for (int i = 0; i < 4; i++)
    {
        delete terrain[i];
    }
    delete terrain;
}

void SearchAndRescue::possibleActions(State* current) {
    string up = "up";
    string down = "down";
    string left = "left";
    string right = "right";
    
    bool m_up = true;
    if(current->y == 3) {  
        m_up = false;
    }
    bool m_down = true;
    if(current->y == 0) {
        m_down = false;
    }
    bool m_left = true;
    if(current->x == 0) {
        m_left = false;
    }
    bool m_right = true;
    if(current->x == 3) {
        m_right = false;
    }
    if(m_up) {
        current->possible_actions.push_back(up);
    }
    if(m_down) {
        current->possible_actions.push_back(down);
    }
    if(m_left) {
        current->possible_actions.push_back(left);
    }
    if(m_right) {
        current->possible_actions.push_back(right);
    }
}


State* SearchAndRescue::result(State* current, string action) {
    State* newState = new State{current->x, current->y, current->saved_people, action, vector<string>()};
    if (action == "up") {
        newState->y += 1; 
    } else if (action == "down") {
        newState->y -= 1;
    } else if (action == "left") {
        newState->x -= 1; 
    } else if (action == "right") {
        newState->x += 1; 
    }
    return newState;
}

vector<State*> SearchAndRescue::expand(State* current) {
    possibleActions(current);
    vector<State*> expansion;
    for (const auto& action : current->possible_actions) {
        State* newState = result(current, action);
        expansion.push_back(newState);
    }
    return expansion;
}




bool SearchAndRescue::iterativeDeepeningWrapper(State* start) {
    int depth_limit = STARTING_DEPTH;
    bool personToSaveStart = terrain[start->x][start->y] == 2;
    if (personToSaveStart) {
        start->saved_people = 1;

        terrain[start->x][start->y] = 1; // Mark as visited or saved.
    } else {
        start->saved_people = 0;
    }
    // Proceed with adding the initial state to the path and starting the iterative deepening search.
    path.push_back(start);
    while (!iterativeDeepeningSearch(start, depth_limit)) {
        depth_limit++;
    }
    return true; 
}


bool SearchAndRescue::iterativeDeepeningSearch(State* current, int depth_limit) {
    if (isGoal(current)) {
        return true; 
    }
    if (depth_limit == 0) {
        return false; 
    }
    bool personToSave = terrain[current->x][current->y] == 2;
    if(personToSave){
        current->saved_people +=1;
        terrain[current->x][current->y] = 1;
    }
    vector<State*> successors = expand(current);
    for (size_t i = 0; i <successors.size(); i++ ) {
        path.push_back(successors[i]); // Add the next state to the path
        if (iterativeDeepeningSearch(successors[i], depth_limit - 1)) {
           
        return true;
        }
        State* toDelete = path.back();
        path.pop_back(); 

        delete toDelete; 
        }
         if (personToSave) {
        current->saved_people--;
        terrain[current->x][current->y] = 2;
    }
    return false;
}
   





void SearchAndRescue::printPath()
{
    for(size_t i = 0; i < path.size(); i++){
        cout << "x: " << path[i]->x  << "\ty: " << path[i]->y << "\tprev_action: " << path[i]->prev_action << "\n";

    }
}


bool SearchAndRescue::isGoal(State* current) {
    if (terrain[current->x][current->y] == 3 && current->saved_people == PEOPLE_TO_SAVE) {
        return true;
    }
    return false; 
}


void SearchAndRescue::printTerrain() {
    for (int i = 3; i >=0; i--) {
        for (int j = 0; j < 4; j++) { // Start from the last column and go backwards
            cout << terrain[j][i] << " ";
        }
        cout << endl; 
    }
}