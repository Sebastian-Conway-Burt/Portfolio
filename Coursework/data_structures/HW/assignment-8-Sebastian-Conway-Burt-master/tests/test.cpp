// If you change anything in this file, your changes will be ignored 
// in your homework submission.
// Chekout TEST_F functions bellow to learn what is being tested.
#include <gtest/gtest.h>
// #include "../code_1/<HEADER FILE>.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include "test_helpers.hpp"
#include "../code_1/Graph.hpp"

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

TEST_F(test_x, AddVertex){
    string result;
    string expected;
    
    string names[] = {"NYC", "Chicago", "Boston", "Boulder", "Denver", "Miami"};
    int len = 6;
    result = test_addVertex(names,len);
    expected = "[GRADER] Adding:NYC\n[GRADER] Adding:Chicago\n[GRADER] Adding:Boston\n[GRADER] Adding:Boulder\n[GRADER] Adding:Denver\n[GRADER] Adding:Miami\nadded vertices are:\nNYC\nChicago\nBoston\nBoulder\nDenver\nMiami\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    
    string names1[] = {"Aspen", "Atlanta", "NYC", "Chicago", "Boston", "Boulder", "Denver", "Miami"};
    int len1 = 8;
    result = test_addVertex(names1,len1);
    expected = "[GRADER] Adding:Aspen\n[GRADER] Adding:Atlanta\n[GRADER] Adding:NYC\n[GRADER] Adding:Chicago\n[GRADER] Adding:Boston\n[GRADER] Adding:Boulder\n[GRADER] Adding:Denver\n[GRADER] Adding:Miami\nadded vertices are:\nAspen\nAtlanta\nNYC\nChicago\nBoston\nBoulder\nDenver\nMiami\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    string names2[] = {};
    int len2 = 0;
    result = test_addVertex(names2,len2);
    expected = "added vertices are:\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    string names3[] = {"NYC"};
    int len3 = 1;
    result = test_addVertex(names3,len3);
    expected = "[GRADER] Adding:NYC\nadded vertices are:\nNYC\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(1);
}



TEST_F(test_x, AddShowEdge){
    string result;
    string expected;
    
    string names[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len = 6;
    string edges[] = {"Boulder-->Denver", "Boulder-->Veil", "NYC-->Boston" };
    int elen = 3;
    result = test_addShowEdge(names,len, edges, elen);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boulder   Denver\n[GRADER] Adding edge between: Boulder   Veil\n[GRADER] Adding edge between: NYC   Boston\nNYC --> Boston \nBoston --> NYC \nBoulder --> Denver Veil \nDenver --> Boulder \nAspen --> \nVeil --> Boulder \n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
    
    
    string names1[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len1 = 6;
    string edges1[] = {"Boston-->Denver", "Denver-->Aspen", "Veil-->Aspen" };
    int elen1 = 3;
    result = test_addShowEdge(names1,len1, edges1, elen1);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\nNYC --> \nBoston --> Denver \nBoulder --> \nDenver --> Boston Aspen \nAspen --> Denver Veil \nVeil --> Aspen \n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
    
    string names2[] = {"NYC", "Boston", "Boulder"};
    int len2 = 3;
    string edges2[] = {"Boulder-->NYC", "NYC-->Boston", "Boston-->Boulder" };
    int elen2 = 3;
    result = test_addShowEdge(names2,len2, edges2, elen2);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding edge between: Boulder   NYC\n[GRADER] Adding edge between: NYC   Boston\n[GRADER] Adding edge between: Boston   Boulder\nNYC --> Boulder Boston \nBoston --> NYC Boulder \nBoulder --> NYC Boston \n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
    
    
    string names3[] = {"NYC", "Boston", "Boulder"};
    int len3 = 3;
    string edges3[] = {"NYC-->Boulder"};
    int elen3 = 1;
    result = test_addShowEdge(names3,len3, edges3, elen3);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding edge between: NYC   Boulder\nNYC --> Boulder \nBoston --> \nBoulder --> NYC \n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
}



TEST_F(test_x, test_bfs){
    string result;
    string expected;
    
    string names[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len = 6;
    string edges[] = {"Boston-->Denver", "Denver-->Aspen", "Veil-->Aspen", "Aspen-->Boulder", "Boulder-->Denver"};
    int elen = 5;
    result = test_bfs(names,len, edges, elen, "NYC");
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \nStarting vertex (root): NYC-> \n\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(10);
    
    
    result = test_bfs(names,len, edges, elen, "Boston");
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \nStarting vertex (root): Boston-> Denver(1) Aspen(2) Boulder(2) Veil(3) \n\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(10);
    
    result = test_bfs(names,len, edges, elen, "Denver");
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \nStarting vertex (root): Denver-> Boston(1) Aspen(1) Boulder(1) Veil(2) \n\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
    
    result = test_bfs(names,len, edges, elen, "Veil");
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \nStarting vertex (root): Veil-> Aspen(1) Denver(2) Boulder(2) Boston(3) \n\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);

    result = test_bfs(names,len, edges, elen, "Boulder");
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \nStarting vertex (root): Boulder-> Aspen(1) Denver(1) Veil(2) Boston(2) \n\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
    
}


TEST_F(test_x, findReachableCities){
    string result;
    string expected;
    
    string names[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len = 6;
    string edges[] = {"Boston-->Denver", "Denver-->Aspen", "Veil-->Aspen", "Aspen-->Boulder", "Boulder-->Denver"};
    int elen = 5;
    string source = "NYC";
    result = test_findReachableCities(names,len, edges, elen, source, 1);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \n[GRADER] No reachable cities.\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(10);

    string names1[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len1 = 6;
    string edges1[] = {"Boston-->Denver", "Denver-->Aspen", "Veil-->Aspen", "Aspen-->Boulder", "Boulder-->Denver"};
    int elen1 = 5;
    string source1 = "Boulder";
    result = test_findReachableCities(names1,len1, edges1, elen1, source1, 1);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \n[GRADER] The reachable cities are: Denver,Aspen\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(10);
    

    string names2[] = {"NYC", "Boston", "Boulder", "Denver", "Aspen", "Veil"};
    int len2 = 6;
    string edges2[] = {"Boston-->Denver", "Denver-->Aspen", "Veil-->Aspen", "Aspen-->Boulder", "Boulder-->Denver"};
    int elen2 = 5;
    string source2 = "Denver";
    result = test_findReachableCities(names2,len2, edges2, elen2, source2, 2);
    expected = "[GRADER] Adding Vertex:NYC\n[GRADER] Adding Vertex:Boston\n[GRADER] Adding Vertex:Boulder\n[GRADER] Adding Vertex:Denver\n[GRADER] Adding Vertex:Aspen\n[GRADER] Adding Vertex:Veil\n[GRADER] Adding edge between: Boston   Denver\n[GRADER] Adding edge between: Denver   Aspen\n[GRADER] Adding edge between: Veil   Aspen\n[GRADER] Adding edge between: Aspen   Boulder\n[GRADER] Adding edge between: Boulder   Denver\nNYC --> \nBoston --> Denver \nBoulder --> Aspen Denver \nDenver --> Boston Aspen Boulder \nAspen --> Denver Veil Boulder \nVeil --> Aspen \n[GRADER] The reachable cities are: Veil\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(15);
    
    
}