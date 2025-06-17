#ifndef TEST_H__
#define TEST_H__
#include <string>
#include <iostream>
#include <fstream>
#include <array>

#include "../code/SearchAndRescue.hpp"

using namespace std;

string test_printTerrain(SearchAndRescue&);
string test_printPath(SearchAndRescue&);
string returnPath(string);
bool processAndReturnMatch(SearchAndRescue&, vector<vector<int>>, int, int);

#endif // TEST_H__