#ifndef TEST_H__
#define TEST_H__
#include <string>
#include <iostream>
#include <fstream>
#include <array>

#include "../code_1/PancakeHouse.hpp"
#include "../code_1/Register.hpp"
#include "../code_1/ShopQueue.hpp"

using namespace std;

// TODO add detailed explanation on what the function should do
std::string exec(const char* cmd);
std::string readFileIntoString(std::string fileName);

string test_push(int* arr, int length);
string test_pop(int* arr, int length);
string test_pop_no_elements(int* arr, int length);
string test_peek(int* arr, int length);
string test_all(int* arr, int length);


string test_consQ();
string test_enQ(CustomerOrder *arr[], int length);
string test_deQ(CustomerOrder *arr[], int length, int dequeue);
string test_peekQ(CustomerOrder *arr[], int length, int dequeue);
string test_allQ(CustomerOrder *show[], int names_length, string* op, int length);

string test_create_customers(CustomerOrder** orders, int length);
string test_cook_orders(int num_customers);
string test_refund(int num_customers);
string test_close_shop(CustomerOrder** orders, int length);

CustomerOrder* createCustomerOrder(string name, int num_pancakes, int type_of_pancakes);
CustomerOrder** createSmallCustomerOrderArray();
CustomerOrder** createLargeCustomerOrderArray();
void deleteCustomerOrders(CustomerOrder** orders, int length);


// string test_evaluate(char* arr, int length);

#endif // TEST_H__