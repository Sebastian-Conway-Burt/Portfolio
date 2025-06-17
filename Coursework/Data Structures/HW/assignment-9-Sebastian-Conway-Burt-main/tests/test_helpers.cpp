#include <gtest/gtest.h>
#include "test_helpers.hpp"

#include "../code/SearchAndRescue.hpp"


string test_printTerrain(SearchAndRescue &tmp_sr) {
    testing::internal::CaptureStdout();

    tmp_sr.printTerrain();

    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_printPath(SearchAndRescue &tmp_sr) {
    testing::internal::CaptureStdout();

    State* start = new State{0, 0, 0, "start", vector<string>()};
    tmp_sr.iterativeDeepeningWrapper(start);
    tmp_sr.printPath();
    
    string output = testing::internal::GetCapturedStdout();
    return output;
}
string returnPath(string fileName){
    SearchAndRescue* sr_a = new SearchAndRescue(fileName);
    return test_printPath(*sr_a);
}

bool processAndReturnMatch(SearchAndRescue &tmp_sr, vector<vector<int>> expectedOutput, int x, int y){
    State* startState = new State{x, y};
	vector<State*> startIndexExpansion = tmp_sr.expand(startState);

	if (startIndexExpansion.size()!= expectedOutput.size()){
        return false;
    }

	for (int i = 0; i < startIndexExpansion.size(); i++){
		if ((startIndexExpansion[i]->x!= expectedOutput[i][0]) || (startIndexExpansion[i]->y!= expectedOutput[i][1])){
            return false;
        }
	}
    return true;
}