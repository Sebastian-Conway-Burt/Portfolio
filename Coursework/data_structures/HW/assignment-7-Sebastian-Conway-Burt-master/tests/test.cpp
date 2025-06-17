// If you change anything in this file, your changes will be ignored 
// If you change anything in this file, your changes will be ignored 
// in your homework submission.
// Chekout TEST_F functions bellow to learn what is being tested.
#include <gtest/gtest.h>
// #include "../code_1/<HEADER FILE>.hpp"
#include "../code_1/ShowCatalog.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include "test_helpers.hpp"

using namespace std;

class test_x : public ::testing::Test {
protected:
	// This function runs only once before any TEST_F function
	static void SetUpTestCase(){
		std::ofstream outgrade("./total_grade.txt");
		if(outgrade.is_open()){
			outgrade.clear();
			outgrade << (int)0;
			outgrade.close();
		}
	}

	// This function runs after all TEST_F functions have been executed
	static void TearDownTestCase(){
		std::ofstream outgrade("./total_grade.txt");
		if(outgrade.is_open()){
			outgrade.clear();
			outgrade << (int)std::ceil(100*total_grade/max_grade);
			outgrade.close();
			std::cout << "Total Grade is : " << (int)std::ceil(100*total_grade/max_grade) << std::endl;
		}
	}

	void add_points_to_grade(double points){
		if(!::testing::Test::HasFailure()){
			total_grade += points;
		}
	}

	// this function runs before every TEST_F function
	void SetUp() override {}

	// this function runs after every TEST_F function
	void TearDown() override {
		std::ofstream outgrade("./total_grade.txt");
		if(outgrade.is_open()){
			outgrade.clear();
			outgrade << (int)std::ceil(100*total_grade/max_grade);
			outgrade.close();
		}
	}
	
	static double total_grade;
	static double max_grade;
};

double test_x::total_grade = 0;
double test_x::max_grade = 100;



/////////////////////////////////////////
// Tests start here
/////////////////////////////////////////
TEST_F(test_x, removeShow){
    string result;
    string expected;
       
    int len = 1;
    string showRating[] = {"PG-13"};
    string title[] = {"Stranger Things"};
    int year[] = {2017}; 
    float userRating[] = {10};
    
    result = test_RemoveShowNode(title, showRating, year, userRating, len, "Stranger Things");
    expected = "[GRADER] Adding:(Stranger Things, 2017, 10, PG-13)\n[GRADER] Deleting: Stranger Things\n[GRADER] Printing movie inventory:\nTree is Empty. Cannot print\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(8);
    
    
    int len1 = 9;
    string showRating1[] ={"PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "R", "PG"};
    string title1[] = {"Stranger Things", "Downton Abbey", "Young Justice", "The Grand Tour", "Sacred Games","Peep Show", "Justice League", "Sinner", "Little Things"};
    int year1[] = {1994,1942,1994,1979,1999,1981,1972, 2020, 2019};
    float userRating1[] = {8.9,8.5,9.3,8.5,8.8,8.5,9.2, 10, 7};
    
    result = test_RemoveShowNode(title1, showRating1, year1, userRating1, len1,"Young Justice");
    expected ="[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Adding:(Sinner, 2020, 10, R)\n[GRADER] Adding:(Little Things, 2019, 7, PG)\n[GRADER] Deleting: Young Justice\n[GRADER] Printing movie inventory:\nShow: Stranger Things 8.9\nShow: Downton Abbey 8.5\nShow: Sacred Games 8.8\nShow: Peep Show 8.5\nShow: Justice League 9.2\nShow: Little Things 7\nShow: Sinner 10\nShow: The Grand Tour 8.5\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(8);
    
    result = test_RemoveShowNode(title1, showRating1, year1, userRating1, len1,"Justice League");
    expected ="[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Adding:(Sinner, 2020, 10, R)\n[GRADER] Adding:(Little Things, 2019, 7, PG)\n[GRADER] Deleting: Justice League\n[GRADER] Printing movie inventory:\nShow: Stranger Things 8.9\nShow: Downton Abbey 8.5\nShow: Sacred Games 8.8\nShow: Peep Show 8.5\nShow: Little Things 7\nShow: Sinner 10\nShow: Young Justice 9.3\nShow: The Grand Tour 8.5\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(8);
    
    int len2 = 10;
    string showRating2[] ={"PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "PG-13","PG-13","R","PG","R"};
    string title2[] = {"Stranger Things", "Downton Abbey", "Young Justice", "The Grand Tour", "Sacred Games","Peep Show", "Justice League", "Sinner", "Little Things", "Avatar: Airbender"};
    int year2[] = {1994,1942,1994,1979,1999,1981,1972,2020,2019,2018};
    float userRating2[] = {8.9,8.5,9.3,8.5,8.8,8.5,9.2,10,7,6,5};

    result = test_RemoveShowNode(title2,showRating2,year2,userRating2,len2,"Sacred Games");
    expected ="[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Adding:(Sinner, 2020, 10, R)\n[GRADER] Adding:(Little Things, 2019, 7, PG)\n[GRADER] Adding:(Avatar: Airbender, 2018, 6, R)\n[GRADER] Deleting: Sacred Games\n[GRADER] Printing movie inventory:\nShow: Stranger Things 8.9\nShow: Downton Abbey 8.5\nShow: Avatar: Airbender 6\nShow: Sinner 10\nShow: Peep Show 8.5\nShow: Justice League 9.2\nShow: Little Things 7\nShow: Young Justice 9.3\nShow: The Grand Tour 8.5\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(8);
}

TEST_F(test_x, rightRotation){
    
    string result;
    string expected;
       
    int len = 4;
    string showRating[] ={"PG-13", "PG-13", "PG", "R"};
    string name[] = {"Dekalog","Chernobyl", "Band of Brothers", "Africa"};
    int year[] = {1950, 2012, 1986, 1974};
    float userRating[] = {9, 9.1, 9.4, 8.8}; 
    
    result = test_RightRotation(name, showRating, year, userRating, len, "Chernobyl");
    expected = "[GRADER] Adding:(Dekalog, 1950, 9, PG-13)\n[GRADER] Adding:(Chernobyl, 2012, 9.1, PG-13)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Africa, 1974, 8.8, R)\nShow: Dekalog 9\nShow: Band of Brothers 9.4\nShow: Africa 8.8\nShow: Chernobyl 9.1\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
    
    
    result = test_RightRotation(name, showRating, year, userRating, len, "Dekalog");
    expected = "[GRADER] Adding:(Dekalog, 1950, 9, PG-13)\n[GRADER] Adding:(Chernobyl, 2012, 9.1, PG-13)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Africa, 1974, 8.8, R)\nShow: Chernobyl 9.1\nShow: Band of Brothers 9.4\nShow: Africa 8.8\nShow: Dekalog 9\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
    
    int len2= 8;
    string showRating2[]={"PG-13", "R", "PG-13", "PG", "PG-13", "R", "PG-13", "R"};
    string name2[] = {"Pride and Prejudice","Africa","Dekalog","Band of Brothers","Chernobyl", "Friends", "Monty Python's Flying Circus", "The Blue Planet"};
    int year2[]={2007,1974,1950,1986,2012,2008,1990,1942};
    float userRating2[] = {8.8,8.8,9,9.4,9.1,8.8,8.8,8.9};
    
    result = test_RightRotation(name2, showRating2, year2, userRating2, len2, "Dekalog");
    expected = "[GRADER] Adding:(Pride and Prejudice, 2007, 8.8, PG-13)\n[GRADER] Adding:(Africa, 1974, 8.8, R)\n[GRADER] Adding:(Dekalog, 1950, 9, PG-13)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Chernobyl, 2012, 9.1, PG-13)\n[GRADER] Adding:(Friends, 2008, 8.8, R)\n[GRADER] Adding:(Monty Python's Flying Circus, 1990, 8.8, PG-13)\n[GRADER] Adding:(The Blue Planet, 1942, 8.9, R)\nShow: Pride and Prejudice 8.8\nShow: Africa 8.8\nShow: Band of Brothers 9.4\nShow: Dekalog 9\nShow: Chernobyl 9.1\nShow: Friends 8.8\nShow: Monty Python's Flying Circus 8.8\nShow: The Blue Planet 8.9\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
}

TEST_F(test_x, leftRotation){
    
    string result;
    string expected;
       
    int len = 4;
    string showRating[] ={"PG-13", "PG-13", "PG", "R"};
    string name[] = {"Africa", "Band of Brothers", "Chernobyl", "Dekalog"};
    int year[] = {1950, 2012, 1986, 1974};
    float userRating[] = {9, 9.1, 9.4, 8.8}; 
    
    result = test_LeftRotation(name, showRating, year, userRating, len, "Chernobyl");
    expected = "[GRADER] Adding:(Africa, 1950, 9, PG-13)\n[GRADER] Adding:(Band of Brothers, 2012, 9.1, PG-13)\n[GRADER] Adding:(Chernobyl, 1986, 9.4, PG)\n[GRADER] Adding:(Dekalog, 1974, 8.8, R)\nShow: Africa 9\nShow: Band of Brothers 9.1\nShow: Dekalog 8.8\nShow: Chernobyl 9.4\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
    
    
    result = test_LeftRotation(name, showRating, year, userRating, len, "Band of Brothers");
    expected = "[GRADER] Adding:(Africa, 1950, 9, PG-13)\n[GRADER] Adding:(Band of Brothers, 2012, 9.1, PG-13)\n[GRADER] Adding:(Chernobyl, 1986, 9.4, PG)\n[GRADER] Adding:(Dekalog, 1974, 8.8, R)\nShow: Africa 9\nShow: Chernobyl 9.4\nShow: Band of Brothers 9.1\nShow: Dekalog 8.8\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
    
    int len2= 8;
    string showRating2[]={"PG-13", "R", "PG-13", "PG", "PG-13", "R", "PG-13", "R"};
    string name2[] = {"Pride and Prejudice","Africa","Dekalog","Band of Brothers","Chernobyl", "Friends", "Monty Python's Flying Circus", "The Blue Planet"};
    int year2[]={2007,1974,1950,1986,2012,2008,1990,1942};
    float userRating2[] = {8.8,8.8,9,9.4,9.1,8.8,8.8,8.9};
    
    result = test_LeftRotation(name2, showRating2, year2, userRating2, len2, "Dekalog");
    expected = "[GRADER] Adding:(Pride and Prejudice, 2007, 8.8, PG-13)\n[GRADER] Adding:(Africa, 1974, 8.8, R)\n[GRADER] Adding:(Dekalog, 1950, 9, PG-13)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Chernobyl, 2012, 9.1, PG-13)\n[GRADER] Adding:(Friends, 2008, 8.8, R)\n[GRADER] Adding:(Monty Python's Flying Circus, 1990, 8.8, PG-13)\n[GRADER] Adding:(The Blue Planet, 1942, 8.9, R)\nShow: Pride and Prejudice 8.8\nShow: Africa 8.8\nShow: Friends 8.8\nShow: Dekalog 9\nShow: Band of Brothers 9.4\nShow: Chernobyl 9.1\nShow: Monty Python's Flying Circus 8.8\nShow: The Blue Planet 8.9\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(7);
}

TEST_F(test_x, TestMain)
{
    string desired = readFileIntoString("../tests/output/output1.txt");
    string resp = exec("./run_app_1 ../tests/input/shows1.csv ../tests/input/input1.txt");
    ASSERT_EQ(desired, resp);
    add_points_to_grade(9);
    
    string desired2 = readFileIntoString("../tests/output/output2.txt");
    string resp2 = exec("./run_app_1 ../tests/input/shows2.csv ../tests/input/input2.txt");
    
    ASSERT_EQ(desired2, resp2);
    add_points_to_grade(9);
    
    string desired4 = readFileIntoString("../tests/output/output3.txt");
    string resp4 = exec("./run_app_1 ../shows.csv ../tests/input/input3.txt");
    
    ASSERT_EQ(desired4, resp4);
    add_points_to_grade(8);
}