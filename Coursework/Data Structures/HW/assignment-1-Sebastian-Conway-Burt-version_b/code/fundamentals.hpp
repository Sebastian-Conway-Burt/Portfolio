#ifndef FUNDAMENTALS_2_H_
#define FUNDAMENTALS_2_H_

#include <string>
#include <iostream>
#include <fstream>

using namespace std;

struct stockInfo { 
    string company_name; 
    double stock_prices_arr[5];
    double average; 
};

void insertStockInfo(stockInfo stocks[], string company_name, double stock_prices_arr[], int index); 

void displaySortedStocks(stockInfo stock, ofstream& file);

#endif