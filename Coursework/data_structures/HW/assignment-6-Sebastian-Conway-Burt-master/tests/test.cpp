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

TEST_F(test_x, TestAddShow_PrintShow){
    string result;
    string expected;
       
    int len = 0;
    string name[] = {""};
    int year[] = {};
    string showRating[] = {""};
    float userRating[] = {};
    
    result = test_add(name,year, showRating, userRating, len);
    expected = "[GRADER] Printing show inventory:\nTree is Empty. Cannot print\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(6);

    int len1 = 1;
    string name1[] = {"show1"};
    int year1[] = {1992};
    string showRating1[] ={"PG-13"};
    float userRating1[] = {0.9};
    
    result = test_add(name1,year1, showRating1, userRating1, len1);
    expected = "[GRADER] Adding:(show1, 1992, PG-13, 0.9)\n[GRADER] Printing show inventory:\nShow: show1 0.9\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(6);
    
    int len2 = 2;
    string name2[] = {"show1", "show2"};
    int year2[] = {1992, 2002};
    string showRating2[] ={"PG-13", "R"};
    float userRating2[] = {0.9, 0.7};
    
    result = test_add(name2, year2, showRating2, userRating2, len2);
    expected = "[GRADER] Adding:(show1, 1992, PG-13, 0.9)\n[GRADER] Adding:(show2, 2002, R, 0.7)\n[GRADER] Printing show inventory:\nShow: show1 0.9\nShow: show2 0.7\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(6);
    
    int len3 = 2;
    string name3[] = {"show2", "show1"};
    int year3[] = {1992, 2002};
    string showRating3[] ={"PG-13", "R"};
    float userRating3[] = {0.9, 0.7};
    
    result = test_add(name3,year3, showRating3, userRating3, len3);
    expected = "[GRADER] Adding:(show2, 1992, PG-13, 0.9)\n[GRADER] Adding:(show1, 2002, R, 0.7)\n[GRADER] Printing show inventory:\nShow: show2 0.9\nShow: show1 0.7\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(6);
    


    int len4 = 8;
    string name4[] = {"Life", "The Sopranos", "The World at War", "Avatar: The Last Airbender", "Chernobyl","Sherlock", "Human Planet", "The Vietnam War"};
    int year4[] = {2008, 1997, 2001,1991,2012,2005, 2002, 2007}; 
    string showRating4[] ={"PG-13", "R", "PG-13", "R", "PG-13", "R", "PG-13", "R"};
    float userRating4[] = {7.8, 7.7, 7.7, 8.1, 7.9, 7.7, 8, 7.5};
    
    result = test_add(name4, year4, showRating4, userRating4, len4);
    expected = "[GRADER] Adding:(Life, 2008, PG-13, 7.8)\n[GRADER] Adding:(The Sopranos, 1997, R, 7.7)\n[GRADER] Adding:(The World at War, 2001, PG-13, 7.7)\n[GRADER] Adding:(Avatar: The Last Airbender, 1991, R, 8.1)\n[GRADER] Adding:(Chernobyl, 2012, PG-13, 7.9)\n[GRADER] Adding:(Sherlock, 2005, R, 7.7)\n[GRADER] Adding:(Human Planet, 2002, PG-13, 8)\n[GRADER] Adding:(The Vietnam War, 2007, R, 7.5)\n[GRADER] Printing show inventory:\nShow: Life 7.8\nShow: Avatar: The Last Airbender 8.1\nShow: Chernobyl 7.9\nShow: Human Planet 8\nShow: The Sopranos 7.7\nShow: Sherlock 7.7\nShow: The World at War 7.7\nShow: The Vietnam War 7.5\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(6);
}

TEST_F(test_x, TestGetShow){
    string result;
    string expected;
       
    int len = 7;
    string showRating[] ={"PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "PG-13", "PG-13"};
    string title[] = {"Stranger Things", "Downton Abbey", "Young Justice", "The Grand Tour", "Sacred Games","Peep Show", "Justice League"};
    int year[] = {1994,1942,1994,1979,1999,1981,1972};
    float userRating[] = {8.9,8.5,9.3,8.5,8.8,8.5,9.2};
    string key = "Stranger Things";
    
    result = test_find(title, showRating, year, userRating, len, key);
    expected = "[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Finding: Stranger Things\nShow Info:\n==================\nTitle :Stranger Things\nYear :1994\nShow Rating :PG-13\nUser Rating :8.9\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    
    key = "Peep Show";
    result = test_find(title, showRating, year, userRating, len, key);
    expected = "[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Finding: Peep Show\nShow Info:\n==================\nTitle :Peep Show\nYear :1981\nShow Rating :PG-13\nUser Rating :8.5\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    key = "The Sting";
    result = test_find(title, showRating, year, userRating, len, key);
    expected = "[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Finding: The Sting\nShow not found.\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    key = "Sacred Games";
    result = test_find(title, showRating, year, userRating, len, key);
    expected = "[GRADER] Adding:(Stranger Things, 1994, 8.9, PG-13)\n[GRADER] Adding:(Downton Abbey, 1942, 8.5, PG-13)\n[GRADER] Adding:(Young Justice, 1994, 9.3, PG-13)\n[GRADER] Adding:(The Grand Tour, 1979, 8.5, PG-13)\n[GRADER] Adding:(Sacred Games, 1999, 8.8, PG-13)\n[GRADER] Adding:(Peep Show, 1981, 8.5, PG-13)\n[GRADER] Adding:(Justice League, 1972, 9.2, PG-13)\n[GRADER] Finding: Sacred Games\nShow Info:\n==================\nTitle :Sacred Games\nYear :1999\nShow Rating :PG-13\nUser Rating :8.8\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
}


TEST_F(test_x, TestSearchShow){
    string result;
    string expected;
       
    int len = 7;
    string showRating[] ={"G", "PG", "PG", "PG-13", "PG-13", "PG", "PG"};
    string title[] = {"Planet Earth II", "Band of Brothers", "Game of Thrones", "Planet Earth", "Breaking Bad","Our Planet", "The Wire"};
    int year[] = {1957,1986,1995,1975,2014,1954,1994};
    float userRating[] = {9.5, 9.4, 9.4 , 9.4, 9.4, 9.3, 9.3};
    
    result = test_query(title, showRating, year, userRating, 'P', len);
    expected = "[GRADER] Adding:(Planet Earth II, 1957, 9.5, G)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Game of Thrones, 1995, 9.4, PG)\n[GRADER] Adding:(Planet Earth, 1975, 9.4, PG-13)\n[GRADER] Adding:(Breaking Bad, 2014, 9.4, PG-13)\n[GRADER] Adding:(Our Planet, 1954, 9.3, PG)\n[GRADER] Adding:(The Wire, 1994, 9.3, PG)\n[GRADER] Searching shows\nShows that starts with P:\nPlanet Earth II(1957) 9.5\nPlanet Earth(1975) 9.4\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(1.25);
    
    result = test_query(title, showRating, year, userRating, 'B', len);
    expected = "[GRADER] Adding:(Planet Earth II, 1957, 9.5, G)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Game of Thrones, 1995, 9.4, PG)\n[GRADER] Adding:(Planet Earth, 1975, 9.4, PG-13)\n[GRADER] Adding:(Breaking Bad, 2014, 9.4, PG-13)\n[GRADER] Adding:(Our Planet, 1954, 9.3, PG)\n[GRADER] Adding:(The Wire, 1994, 9.3, PG)\n[GRADER] Searching shows\nShows that starts with B:\nBand of Brothers(1986) 9.4\nBreaking Bad(2014) 9.4\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(1.25);
    
    
    result = test_query(title, showRating, year, userRating, 'G', len);
    expected = "[GRADER] Adding:(Planet Earth II, 1957, 9.5, G)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Game of Thrones, 1995, 9.4, PG)\n[GRADER] Adding:(Planet Earth, 1975, 9.4, PG-13)\n[GRADER] Adding:(Breaking Bad, 2014, 9.4, PG-13)\n[GRADER] Adding:(Our Planet, 1954, 9.3, PG)\n[GRADER] Adding:(The Wire, 1994, 9.3, PG)\n[GRADER] Searching shows\nShows that starts with G:\nGame of Thrones(1995) 9.4\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2.5);
    
    result = test_query(title, showRating, year, userRating, 'O', len);
    expected = "[GRADER] Adding:(Planet Earth II, 1957, 9.5, G)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Game of Thrones, 1995, 9.4, PG)\n[GRADER] Adding:(Planet Earth, 1975, 9.4, PG-13)\n[GRADER] Adding:(Breaking Bad, 2014, 9.4, PG-13)\n[GRADER] Adding:(Our Planet, 1954, 9.3, PG)\n[GRADER] Adding:(The Wire, 1994, 9.3, PG)\n[GRADER] Searching shows\nShows that starts with O:\nOur Planet(1954) 9.3\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2.5);
    
    
    result = test_query(title, showRating, year, userRating, 'T', len);
    expected =  "[GRADER] Adding:(Planet Earth II, 1957, 9.5, G)\n[GRADER] Adding:(Band of Brothers, 1986, 9.4, PG)\n[GRADER] Adding:(Game of Thrones, 1995, 9.4, PG)\n[GRADER] Adding:(Planet Earth, 1975, 9.4, PG-13)\n[GRADER] Adding:(Breaking Bad, 2014, 9.4, PG-13)\n[GRADER] Adding:(Our Planet, 1954, 9.3, PG)\n[GRADER] Adding:(The Wire, 1994, 9.3, PG)\n[GRADER] Searching shows\nShows that starts with T:\nThe Wire(1994) 9.3\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2.5);
    
}


TEST_F(test_x, TestDisplayNumShowRating){
    string result;
    string expected;
       
    int len = 7;
    string name[] = {"Pulp Fiction", "Casablanca", "Shawshank Redemption", "Apocalypse Now", "Fight Club","Raiders of the Lost Ark", "The Godfather"};
    int year[] = {1994,1942,1994,1979,1999,1981,1972};
    string showRating[] ={"G", "PG", "PG", "PG-13", "PG-13", "PG", "PG"};
    float userRating[] = {9.5, 9.4, 9.4 , 9.4, 9.4, 9.3, 9.3};

    ShowCatalog* tree = new ShowCatalog();
    for(int i=0;i<len;i++)
    {
        tree->addShowItem(name[i], year[i], showRating[i], userRating[i]);
    }

    string showRatingArg = "PG-13";
    result = test_displayNumShowRating(tree, showRatingArg);
    expected ="Number of PG-13: 2\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(7.5);
    
    showRatingArg = "PG";
    result = test_displayNumShowRating(tree, showRatingArg);
    expected ="Number of PG: 4\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(7.5);

    delete tree;
    
}


TEST_F(test_x, TestLeafNode){
    string result;
    string expected;
       
    int len = 7;
    string name[] = {"Life", "The Sopranos", "The World at War", "Avatar: The Last Airbender", "Chernobyl","Sherlock", "Human Planet"};
    int year[] = {1994,1942,1994,1979,1999,1981,1972};
    string showRating[] ={"PG-13", "R", "PG-13", "R", "PG-13", "R", "PG-13", "R"};
    float userRating[] = {8.9,8.5, 9.3,8.5,8.8,8.5, 9.2};
    result = test_printLeafNodes(name, year, showRating, userRating,  len);
    expected ="[GRADER] Result of movies.printLeafNodes is:\nHuman Planet\nSherlock\nThe World at War\n\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    
    
    int len2 = 0;
    string name2[] = {""};
    int year2[] = {};
    result = test_printLeafNodes(name2, year2, showRating, userRating, len2);
    expected = "[GRADER] Result of movies.printLeafNodes is:\n\n\n[GRADER] No leaked memory (Good).\n";
    
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    
    string name1[] = { "Planet Earth II", "Band of Brothers", "Game of Thrones","Planet Earth",  "Our Planet", "The Wire", "Rick and Morty"};
    int year1[] = {1979,1942,1999,1994,1981,1994,1972};
    string showRating1[] ={"PG-13", "R", "PG-13", "R", "PG-13", "R", "PG-13", "R"};
    float userRating1[] = {8.5,8.9,8.8, 8.9,8.5,9.3, 9.2};
    result = test_printLeafNodes(name1, year1, showRating1, userRating1 , len);
    expected ="[GRADER] Result of movies.printLeafNodes is:\nOur Planet\nRick and Morty\n\n\n[GRADER] No leaked memory (Good).\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(5);
}


TEST_F(test_x, TestMain)
{
    string desired = readFileIntoString("../tests/output/output1.txt");
    string resp = exec("./run_app_1 ../tests/input/shows1.csv ../tests/input/input1.txt");
    ASSERT_EQ(desired, resp);
    add_points_to_grade(6.25);
    
    string desired2 = readFileIntoString("../tests/output/output2.txt");
    string resp2 = exec("./run_app_1 ../tests/input/shows2.csv ../tests/input/input2.txt");
    
    ASSERT_EQ(desired2, resp2);
    add_points_to_grade(6.25);
    
    
    string desired3 = readFileIntoString("../tests/output/output3.txt");
    string resp3 = exec("./run_app_1 ../tests/input/shows1.csv ../tests/input/input3.txt");
    ASSERT_EQ(desired3, resp3);
    add_points_to_grade(6.25);
    
    
    string desired4 = readFileIntoString("../tests/output/output4.txt");
    string resp4 = exec("./run_app_1 ../shows.csv ../tests/input/input4.txt");
    
    ASSERT_EQ(desired4, resp4);
    add_points_to_grade(6.25);
}
