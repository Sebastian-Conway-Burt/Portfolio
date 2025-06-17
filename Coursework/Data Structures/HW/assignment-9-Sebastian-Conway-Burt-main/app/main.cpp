#include <iostream>
#include <vector>
#include "../code/SearchAndRescue.hpp"
#include <string>
using namespace std;

int main(int argc, char** argv) {

    SearchAndRescue* sr = new SearchAndRescue("../input.txt");
    sr->printTerrain();
    cout << endl;

    State* start = new State{0, 0, 0, "start", vector<string>()};
    sr->iterativeDeepeningWrapper(start);
    sr->printPath();

    delete sr;
    return 0;
}