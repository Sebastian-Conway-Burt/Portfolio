// If you change anything in this file, your changes will be ignored 
// in your homework submission.
// Chekout TEST_F functions bellow to learn what is being tested.
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include "test_helpers.hpp"
#include "../code/SearchAndRescue.hpp"

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
// Tests start here for Assignment-9
/////////////////////////////////////////

TEST_F(test_x, Test_PrintTerrain) {
	string result, expected;
	string test_a = "../tests/data/test_a.txt";
    SearchAndRescue* sr_a = new SearchAndRescue(test_a);
    result = test_printTerrain(*sr_a);
	expected = "1 2 1 3 \n1 1 2 1 \n2 1 1 1 \n1 1 2 1 \n";
    ASSERT_EQ(expected, result);

	string test_b = "../tests/data/test_b.txt";
    SearchAndRescue* sr_b = new SearchAndRescue(test_b);
    result = test_printTerrain(*sr_b);
	expected = "2 1 1 3 \n1 1 2 1 \n1 1 1 2 \n2 1 1 1 \n";
    ASSERT_EQ(expected, result);

	string test_c = "../tests/data/test_c.txt";
    SearchAndRescue* sr_c = new SearchAndRescue(test_c);
    result = test_printTerrain(*sr_c);
	expected = "2 1 1 3 \n1 2 2 1 \n1 1 1 1 \n1 1 1 2 \n";
    ASSERT_EQ(expected, result);

    add_points_to_grade(15.0);
}


TEST_F(test_x, Test_IsGoal) {
	State* start = new State{0, 0, 0, "start", vector<string>()};
	string fileName = "../tests/data/test_a.txt";
    SearchAndRescue* sr_a = new SearchAndRescue(fileName);
	State* threePeopleSaved = new State{3, 3, 3};
	State* noDestSavedFour = new State{3, 2, 4};
	State* reachedDestSavedFour = new State{3, 3, 4};

	ASSERT_EQ(false, sr_a->isGoal(threePeopleSaved));
	ASSERT_EQ(false, sr_a->isGoal(noDestSavedFour));
	ASSERT_EQ(true, sr_a->isGoal(reachedDestSavedFour));

    add_points_to_grade(5.0);
}

TEST_F(test_x, Test_Result) {
	State* start = new State{0, 0, 0, "start", vector<string>()};
	string fileName = "../tests/data/test_a.txt";
    SearchAndRescue* sr_a = new SearchAndRescue(fileName);
	State* current = new State{2, 2, 3};
	State* up = sr_a->result(current, "up");
	ASSERT_EQ(2,up->x);
	ASSERT_EQ(3,up->y);

	State* down = sr_a->result(current, "down");
	ASSERT_EQ(2,down->x);
	ASSERT_EQ(1,down->y);

	State* right = sr_a->result(current, "right");
	ASSERT_EQ(3,right->x);
	ASSERT_EQ(2,right->y);

	State* left = sr_a->result(current, "left");
	ASSERT_EQ(1,left->x);
	ASSERT_EQ(2,left->y);

    add_points_to_grade(10.0);
}


TEST_F(test_x, Test_ExpandAndPossibleActions) {
	State* start = new State{0, 0, 0, "start", vector<string>()};
	string fileName = "../tests/data/test_a.txt";
    SearchAndRescue* sr = new SearchAndRescue(fileName);
	vector<vector<int>> expectedOutputForZeroZeroIndex = {{0,1},{1,0}};
	vector<vector<int>> expectedOutputForFourFourIndex = {{3,2},{2,3}};
	vector<vector<int>> expectedOutputForThreeThreeIndex = {{2,3},{2,1},{1,2},{3,2}};

	ASSERT_EQ(true, processAndReturnMatch(*sr, expectedOutputForZeroZeroIndex, 0, 0));
	ASSERT_EQ(true, processAndReturnMatch(*sr, expectedOutputForFourFourIndex, 3, 3));
	ASSERT_EQ(true, processAndReturnMatch(*sr, expectedOutputForThreeThreeIndex, 2, 2));
	
    add_points_to_grade(20.0);
}

TEST_F(test_x, Test_IterativeDeepeningAndPrintPath) {
	string expected_a, expected_b, expected_c;
	
	expected_a = "x: 0\ty: 0\tprev_action: start\nx: 0\ty: 1\tprev_action: up\nx: 0\ty: 0\tprev_action: down\nx: 1\ty: 0\tprev_action: right\nx: 2\ty: 0\tprev_action: right\nx: 2\ty: 1\tprev_action: up\nx: 2\ty: 2\tprev_action: up\nx: 2\ty: 3\tprev_action: up\nx: 1\ty: 3\tprev_action: left\nx: 2\ty: 3\tprev_action: right\nx: 3\ty: 3\tprev_action: right\n";
    ASSERT_EQ(expected_a, returnPath("../tests/data/test_a.txt"));

	expected_b = "x: 0\ty: 0\tprev_action: start\nx: 0\ty: 1\tprev_action: up\nx: 0\ty: 2\tprev_action: up\nx: 0\ty: 3\tprev_action: up\nx: 0\ty: 2\tprev_action: down\nx: 1\ty: 2\tprev_action: right\nx: 2\ty: 2\tprev_action: right\nx: 2\ty: 1\tprev_action: down\nx: 3\ty: 1\tprev_action: right\nx: 3\ty: 2\tprev_action: up\nx: 3\ty: 3\tprev_action: up\n";
    ASSERT_EQ(expected_b, returnPath("../tests/data/test_b.txt"));

	expected_c = "x: 0\ty: 0\tprev_action: start\nx: 0\ty: 1\tprev_action: up\nx: 0\ty: 2\tprev_action: up\nx: 0\ty: 3\tprev_action: up\nx: 0\ty: 2\tprev_action: down\nx: 1\ty: 2\tprev_action: right\nx: 2\ty: 2\tprev_action: right\nx: 2\ty: 1\tprev_action: down\nx: 2\ty: 0\tprev_action: down\nx: 3\ty: 0\tprev_action: right\nx: 3\ty: 1\tprev_action: up\nx: 3\ty: 2\tprev_action: up\nx: 3\ty: 3\tprev_action: up\n";
    ASSERT_EQ(expected_c, returnPath("../tests/data/test_c.txt"));

    add_points_to_grade(50.0);
}
