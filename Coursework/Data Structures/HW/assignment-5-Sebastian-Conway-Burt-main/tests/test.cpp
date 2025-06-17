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

//Grade Points: 4
TEST_F(test_x, TestPush)
{
    string result, expected;
    int arr[] = {1,2,3,4};
    int len = 4;
    result = test_push(arr,  len);
    expected= "Print stack from top to bottom:\n4\n3\n2\n1\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);

    int arr2[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
    len = 21;
    result = test_push(arr2, len);
    expected= "Stack overflow: \nPrint stack from top to bottom:\n20\n19\n18\n17\n16\n15\n14\n13\n12\n11\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
}

//Grade Points 4
TEST_F(test_x, TestPop)
{
    string result, expected;
    int arr[] = {1,2,3,4};
    int len = 4;
    result = test_pop(arr,  len);
    expected= "Print stack from top to bottom:\n4\n3\n2\n1\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
}


//Grade Points 4
TEST_F(test_x, TestPopNoElements)
{
    string result, expected;
    int arr[] = {1,2,3,4};
    int len = 4;
    result = test_pop_no_elements(arr,  len);
    expected= "Print stack from top to bottom:\n4\n3\n2\n1\nStack empty, cannot pop an item!\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
}

//Grade Points 8
TEST_F(test_x, TestAll)
{
    string result, expected;
    int arr[] = {1,2,3,4};
    int len = 4;
    result = test_all(arr,  len);
    expected= "push: 1\npush: 2\npoping thrice \nStack empty, cannot pop an item!\npush: 3\npush: 4\nPrint stack from top to bottom:\n4\n3\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(8);
}


//Grade Points 4
TEST_F(test_x, TestEnQ)
{
    string result, expected;
    
    int len = 4;
    CustomerOrder** orders = createSmallCustomerOrderArray();
    
    expected= "Print queue from front to back:\n";
    for(int x = 0; x < len; x++){
        CustomerOrder* order = orders[x];
        expected += order->name + " " + to_string(order->number_of_pancakes) + " " + to_string(order->type_of_pancakes) +"\n";
    }
    result = test_enQ(orders,  len);
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    len = 21;
    CustomerOrder** ordersLarge = createLargeCustomerOrderArray();
    expected= "Print queue from front to back:\n";
    //Will print every name except for extra if correctly done.
    for(int x = 0; x < len; x++){
        CustomerOrder* order = ordersLarge[x];
        expected += order->name + " " + to_string(order->number_of_pancakes) + " " + to_string(order->type_of_pancakes) +"\n";
    }
    result = test_enQ(ordersLarge,  len);

    ASSERT_EQ(expected, result);
    deleteCustomerOrders(orders, 4);
    deleteCustomerOrders(ordersLarge, 21);

    add_points_to_grade(2);
}

//Grade Points 4
TEST_F(test_x, TestDeQ)
{
    // string result, expected;
    int len = 4;
    CustomerOrder** orders = createSmallCustomerOrderArray();
    string result = test_deQ(orders,  len, 3);
    string expected = "dequeuing 3 times\nPrint queue from front to back:\nPreston 1 3\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    
    len = 4;
    result = test_deQ(orders, len, 5);
    expected= "dequeuing 5 times\nQueue empty, cannot dequeue!\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    deleteCustomerOrders(orders, 4);
}

//Grade Points 6
TEST_F(test_x, TestPeekQ)
{
    string result, expected;
    CustomerOrder** orders = createSmallCustomerOrderArray();
    int len = 0;
    int dq = 0;
    result = test_peekQ(orders,  len, dq);
    expected= "enqueuing 0 elements\ndequeuing 0 times\npeeking in the queue\nQueue empty, cannot peek!\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    
    len = 4;
    result = test_peekQ(orders,  len, 3);
    expected= "enqueuing 4 elements\ndequeuing 3 times\npeeking in the queue\nFront of the queue: Preston 1 3\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    
    len = 4;
    result = test_peekQ(orders,  len, 5);
    expected= "enqueuing 4 elements\ndequeuing 5 times\nQueue empty, cannot dequeue!\npeeking in the queue\nQueue empty, cannot peek!\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    deleteCustomerOrders(orders, 4);
}

//Grade Points 20
TEST_F(test_x, TestAllQ)
{
    //TODO
    //enQ2 deQ2
    string result, expected;
    CustomerOrder** orders = createSmallCustomerOrderArray();
    string seq1[] = {"E","E","D","D", "P"};
    int len1 = 5;
    result = test_allQ(orders, 4, seq1, len1);
    expected = "Enqueue(Jon)\nEnqueue(James)\nDequeue\nDequeue\nPeek\nQueue empty, cannot peek!\nQueue is empty? true\nQueue size: 0\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
    
    //enQ20 deQ10 enQ7
    string seq2[] = {"E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","D","D","D","D","D","D","D","D","D","D","E","E","E","E","E","E","E"};
    int len2 = 37;
    result = test_allQ(orders, 4, seq2, len2);
    expected = "Enqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nQueue is empty? false\nQueue size: 17\nQueue Front: Adam 3 5\nQueue End: Jon 4 1\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
    
     //enQ20 deQ10 enQ10
    string seq3[] = {"E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","D","D","D","D","D","D","D","D","D","D","E","E","E","E","E","E","E","E","E","E"};
    int len3 = 40;
    result = test_allQ(orders, 4, seq3, len3);
    expected = "Enqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nQueue is empty? false\nQueue size: 20\nQueue Front: Adam 3 5\nQueue End: Preston 1 3\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
    
    //enQ20 deQ10 enQ10 deQ20
    string seq4[] = {"E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","D","D","D","D","D","D","D","D","D","D","E","E","E","E","E","E","E","E","E","E","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D","D"};
    int len4 = 60;
    result = test_allQ(orders, 4, seq4, len4);
    expected = "Enqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nQueue is empty? true\nQueue size: 0\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
    
     //enQ20 deQ9 enQ4 deQ7
    string seq5[] = {"E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","D","D","D","D","D","D","D","D","D","E","E","E","E","D","D","D","D","D","D","D"};
    int len5 = 40;
    result = test_allQ(orders, 4, seq5, len5);
    expected = "Enqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nEnqueue(James)\nEnqueue(Adam)\nEnqueue(Preston)\nEnqueue(Jon)\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nDequeue\nQueue is empty? false\nQueue size: 8\nQueue Front: Jon 4 1\nQueue End: Jon 4 1\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(4);
    deleteCustomerOrders(orders, 4);
}

// Test points 6
TEST_F(test_x, CreateOrders)
{
    string output, expected;
    CustomerOrder** orders = createSmallCustomerOrderArray();
    output = test_create_customers(orders, 4);
    expected = "Size: 4\n";
    ASSERT_EQ(expected,output);
    add_points_to_grade(1);

    CustomerOrder** orders2 = createLargeCustomerOrderArray();
    output = test_create_customers(orders2, 21);
    expected = "Size: 20\n";
    ASSERT_EQ(expected, output);
    add_points_to_grade(1);

    CustomerOrder** orders3 = new CustomerOrder*[1];
    CustomerOrder *temp_order = new CustomerOrder;
    temp_order->name = "ThisWontWork";
    temp_order->number_of_pancakes = 10000;
    temp_order->type_of_pancakes = 10;
    orders3[0] = temp_order;
    output = test_create_customers(orders3, 1);
    expected = "Size: 0\n";
    ASSERT_EQ(expected,output);

    deleteCustomerOrders(orders, 4);
    deleteCustomerOrders(orders2, 21);
    deleteCustomerOrders(orders3, 1);
}

// Test points 7
TEST_F(test_x, CookPancakes)
{   
    string result = test_cook_orders(0);
    string expected = "Queue empty, cannot peek!\n0\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);
    result = test_cook_orders(4);
    expected = "Queue empty, cannot peek!\n0\n60\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(2);

    result = test_refund(4);
    expected = "Money refunded = 4\n56\n";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
}

//Test Points 3
TEST_F(test_x, CloseShop){
    CustomerOrder** orders2 = createLargeCustomerOrderArray();
    string result = test_close_shop(orders2, 21);
    //20 cause stack can only hold 20
    string expected = "Size: 20\nQueue Size: 0\nIs register empty?: True";
    ASSERT_EQ(expected, result);
    add_points_to_grade(3);
    deleteCustomerOrders(orders2, 21);
}

// Test points 10
TEST_F(test_x, PrintOrders)
{
    PancakeHouse* pancakeHouse = new PancakeHouse;   
    testing::internal::CaptureStdout();  
    pancakeHouse->printOrders();  
    std::string result = testing::internal::GetCapturedStdout();
    std::string expectedOutput = "Queue empty, cannot peek!\n";
    ASSERT_EQ(expectedOutput, result);
    add_points_to_grade(3);


    CustomerOrder order1 = {"John Doe", 3, 1};
    CustomerOrder order2 = {"Alice Smith", 2, 2};
    CustomerOrder order3 = {"Bob Johnson", 1, 4};  
    pancakeHouse->addOrder(order1.name, order1.number_of_pancakes, order1.type_of_pancakes);
    pancakeHouse->addOrder(order2.name, order2.number_of_pancakes, order2.type_of_pancakes);
    pancakeHouse->addOrder(order3.name, order3.number_of_pancakes, order3.type_of_pancakes);
    testing::internal::CaptureStdout();  
    pancakeHouse->printOrders();  
    result = testing::internal::GetCapturedStdout();
    expectedOutput =
    "Customer name: John Doe\nNumber of pancakes ordered: 3\nType of pancakes: 1\n"
    "Customer name: Alice Smith\nNumber of pancakes ordered: 2\nType of pancakes: 2\n"
    "Customer name: Bob Johnson\nNumber of pancakes ordered: 1\nType of pancakes: 4\n";  
    
    ASSERT_EQ(expectedOutput, result);
    add_points_to_grade(7);
}

// Test points 20
TEST_F(test_x, TestMain)
{
    //CHECK ALL OF ABOVE FUNCTIONS AND CHECK THAT PROFIT/REFUNDS ARE PROCESSED CORRECTLY
    string desired;
    string resp;
    desired = readFileIntoString("../tests/output_1.txt");
    resp = exec("./run_app ../tests/input_1.txt");
    ASSERT_EQ(desired, resp);
    add_points_to_grade(10);
    
    desired = readFileIntoString("../tests/output_2.txt");
    resp = exec("./run_app ../tests/input_2.txt");
    ASSERT_EQ(desired, resp);
    add_points_to_grade(10);
}
