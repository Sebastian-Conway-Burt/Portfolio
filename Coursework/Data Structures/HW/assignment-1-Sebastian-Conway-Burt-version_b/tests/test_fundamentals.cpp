// If you change anything in this file, your changes will be ignored 
// in your homework submission.
// Checkout TEST_F functions below to learn what is being tested.
#include <gtest/gtest.h>
#include "../code/fundamentals.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>

#define EPSILON 10e-6

using namespace std;

class test_fundamentals : public ::testing::Test {
protected:
	// This function runs only once before any TEST_F function
	static void SetUpTestCase(){
		std::ofstream outgrade("./total_grade.txt");
		if(outgrade.is_open()){
			outgrade.clear();
			outgrade << (int)0;
			outgrade.close();
		}

    // Clean up to ensure old output files are not present
    remove("../out_1.csv");
    remove("../out_2.csv");
    remove("../out_3.csv");
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

double test_fundamentals::total_grade = 0;
double test_fundamentals::max_grade = 100;

/////////////////////////////////////////
// Test Helper Functions
/////////////////////////////////////////

bool compare_arrays(float a[], float b[], int size){
  bool pass = true;
  for(int i = 0; i<size; ++i){
    if(a[i] != b[i]){
      pass = false;
      cout << "Actual Output: " << a[i] << "\nExpected Output: " << b[i] << endl;
    }
  }
  return pass;
}

bool compare_structs(stockInfo &a, stockInfo &b){
  bool pass = false;
  if(a.company_name == b.company_name 
    && a.stock_prices_arr[0] == b.stock_prices_arr[0] 
    && a.stock_prices_arr[1] == b.stock_prices_arr[1] 
    && a.stock_prices_arr[2] == b.stock_prices_arr[2]
    && a.stock_prices_arr[3] == b.stock_prices_arr[3]
    && a.stock_prices_arr[4] == b.stock_prices_arr[4]
    && a.average == b.average
  ){
    pass = true;
  }
  else{
    cout << "The actual and the expected stockInfo structs are not the same." << endl;
  }
  return pass;
}

bool isFloatingPointNumber(string s) {
  int sz = s.size(), digCount = 0;
  bool dot = false;

  for (int i = 0; i < (int) s.size(); i++) {
    if (isdigit(s[i])) digCount++;
    else {
      if (s[i] == '.') dot = true;
    }
  }
  bool ans = digCount == (sz - 1) && dot == true;
  return ans;
}

bool compareFloats(double a, double b) {
  bool ans = abs(a - b) <= EPSILON;
  return ans;
}

bool sameFloatingPointNumbers(string a, string b) {
  bool same = isFloatingPointNumber(a) && isFloatingPointNumber(b) && compareFloats(stod(a), stod(b));
  return same;
}

vector <string> split (string s, char ch) {
  vector <string> tmp; string w = "";
  for (int i = 0; i < (int) s.size(); i++) {
    if (s[i] == ch) {
	tmp.push_back(w);
	w = "";
    } else {
	w += s[i];
    }
  }
  tmp.push_back(w);
  return tmp;
}

bool sameLineWithFloats (string a, string b) {
  vector <string> la = split(a, ',');
  vector <string> lb = split(b, ','); 

  int sza = la.size(), szb = lb.size();
  if (sza != szb) return false;

  for (int i = 0; i < sza; i++) {
    if (isFloatingPointNumber(la[i])) {
      if (!isFloatingPointNumber(lb[i]) || !sameFloatingPointNumbers(la[i], lb[i])) {
        return false;
      }
    } else {
      if (la[i] != lb[i]) return false;
    }
  }
  return true;
}

bool compare_files(string fileA, string fileB){
  std::string lineA, lineB;
  ifstream A, B;
  A.open(fileA);
  if(!A){throw std::runtime_error("could not open file: " + fileA);}
  B.open(fileB);
  if(!B){throw std::runtime_error("could not open file: " + fileB);}

  bool pass = true;
  while(getline(A,lineA) && getline(B,lineB)){
    if(lineA!=lineB && !sameLineWithFloats(lineA, lineB)){
      pass = false;
      cout << "Actual Output: " << lineA << "\nExpected Output: " << lineB << endl;
    }
  }
  return pass;
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

/////////////////////////////////////////
// Tests start here
/////////////////////////////////////////

// Test the insertStocksInfo function [20 POINTS]
TEST_F(test_fundamentals, TestInsertStocksInfo){

  // Check if adding a single element correctly writes a 
  // struct within the array of structs

  stockInfo stocks[10];
  double prices[4][5];
  int k = 10;
  for (int i = 0 ;i < 4; i++){
    for(int j = 0; j < 5; j++){
       prices[i][j] = k;
    }
    k++;
  }
  
  insertStockInfo(stocks, "Apple", prices[0], 0); 

  stockInfo apple;
  apple.company_name = "Apple";
  for(int i = 0; i < 5; i++)
    apple.stock_prices_arr[i] = 10.0;

  apple.average = 10.0;
  bool same = compare_structs(stocks[0], apple);
  ASSERT_TRUE(same);
  add_points_to_grade(10);

  // Write 3 more records to the array of structs
  // Ensure that overall averages are being calculated correctly
  int idx = 0;
  insertStockInfo(stocks, "Tesla", prices[1], idx++);
  insertStockInfo(stocks, "Microsoft", prices[2], idx++);
  insertStockInfo(stocks, "Meta", prices[3], idx++);
  same = false;
  double averages [] = {11.0, 12.0, 13.0};
  for (int i = 1; i < idx; i++) {
    ASSERT_EQ(stocks[i].average, averages[i]);
    add_points_to_grade(5);
  }
}

string desired_result_1("Activision Blizzard Inc,75.05,66.44,74.41,76.6,77.97,79.83\nAdobe Inc,400.852,278.27,340.16,364.18,455,566.65\nASML Holding NV ADRs,100,50,75,100,125,150\nAdvanced Micro Devices Inc,92.254,64.46,66,75.19,110.48,145.14\n");
string desired_result_2("Intuitive Surgical Inc,264.542,189.4,201.07,269.59,304,358.65\nJD.com Inc ADRs,60.282,49.38,57.97,61.25,63.91,68.9\n");
string desired_result_3("Workday Inc,194.6,140.31,152.68,170.7,237.35,271.96\nXcel Energy Inc,69.04,64.72,67.66,69.81,70.79,72.22\nZoom Video Communications Inc. - Class A,110.738,69.01,73.91,108.77,117.56,184.44\nZscaler Inc,198.404,114.72,149.65,165.05,240,322.6\n");

// Test the entire program [80 POINTS]
// The program should print the results to both standard output and to a csv file
// - each case tests for both. 
TEST_F(test_fundamentals, TestApp_2){
  string resp_1 = exec("./run_app ../stock_1.csv ../out_1.csv");
  ASSERT_EQ(resp_1,desired_result_1);
  add_points_to_grade(12);
  bool same = compare_files("../out_1.csv","../tests/expected_1.csv");
  ASSERT_TRUE(same);
  add_points_to_grade(16);

  string resp_2 = exec("./run_app ../stock_2.csv ../out_2.csv");
  ASSERT_EQ(resp_2,desired_result_2);
  add_points_to_grade(12);
  same = compare_files("../out_2.csv","../tests/expected_2.csv");
  ASSERT_TRUE(same);
  add_points_to_grade(14);

  string resp_3 = exec("./run_app ../stock_3.csv ../out_3.csv");
  ASSERT_EQ(resp_3,desired_result_3);
  add_points_to_grade(12);
  same = compare_files("../out_3.csv","../tests/expected_3.csv");
  ASSERT_TRUE(same);
  add_points_to_grade(14);
}
