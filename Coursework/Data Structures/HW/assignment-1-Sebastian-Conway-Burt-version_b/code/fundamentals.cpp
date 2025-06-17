#include "fundamentals.hpp"

void insertStockInfo(stockInfo stocks[], string company_name, double stock_prices_arr[], int index) {
    stocks[index].company_name = company_name;
    double total = 0.0;
    for (int i = 0; i < 5; ++i) {
        stocks[index].stock_prices_arr[i] = stock_prices_arr[i];
        total += stock_prices_arr[i];
    }
    stocks[index].average = total / 5;
}

void displaySortedStocks(stockInfo stock, ofstream& file) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4 - i; j++) {
            if (stock.stock_prices_arr[j] > stock.stock_prices_arr[j + 1]) {
                double temp = stock.stock_prices_arr[j];
                stock.stock_prices_arr[j] = stock.stock_prices_arr[j + 1];
                stock.stock_prices_arr[j + 1] = temp;
            }
        }
    }
    cout << stock.company_name << "," << stock.average;
    file << stock.company_name << "," << stock.average;
    for (int i = 0; i < 5; i++) {
        cout << "," << stock.stock_prices_arr[i];
        file << "," << stock.stock_prices_arr[i];
    }
    cout << endl;
    file << endl;
}
