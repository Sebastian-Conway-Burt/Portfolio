#ifndef TEST_H__
#define TEST_H__
#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include "../code_1/ShowCatalog.hpp"

using namespace std;

// TODO add detailed explanation on what the function should do
std::string exec(const char* cmd);
std::string readFileIntoString(std::string fileName);
string test_add(string* name,int* year, string* showRating, float* userRating, int len);
string test_find(string* title,string* showRating, int* year, float* userRating, int len, string key);
string test_query(string* title,string* showRating, int* year, float* userRating, char titleChar, int len);
string test_displayNumShowRating(ShowCatalog* tree, string showRating);
string test_printLeafNodes(string* name,int* year, string* showRating, float* userRating, int len);

#endif // TEST_H__