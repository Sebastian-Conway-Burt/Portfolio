#include <iostream>
#include <fstream>
#include "../code/fundamentals.hpp"
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {

    ifstream inFile(argv[1]);
    ofstream outFile(argv[2]);
    if (!inFile.is_open()) {
        cout << "Could not open file " << argv[1] << endl;
        return -1;
    }

    stockInfo stocks[6];
    string line;
    int index = 0;

    while (getline(inFile, line) && index < 6) {
        stringstream ss(line);
        string token, company_name;
        double stock_prices[5];

        getline(ss, company_name, ',');
        for (int i = 0; i < 5; ++i) {
            getline(ss, token, ',');
            stock_prices[i] = stod(token);
        }

        insertStockInfo(stocks, company_name, stock_prices, index);
        index++;
    }

    inFile.close();

    for (int i = 0; i < index; ++i) {
        displaySortedStocks(stocks[i], outFile);
    }

    outFile.close();

    return 0;
}
