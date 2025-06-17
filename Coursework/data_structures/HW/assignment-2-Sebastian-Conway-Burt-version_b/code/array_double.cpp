#include "array_double.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>


void parseFile(ifstream& input, string queryParams[], AirlinePassenger *&passengers, int &arrCapacity, int &recordIdx, int &doublingCounter) {
    string line;
    while (getline(input, line)) {
        stringstream ss(line);
        string name, airline, location, ageStr;
        getline(ss, name, ',');
        getline(ss, airline, ',');
        getline(ss, location, ',');
        getline(ss, ageStr);
        int age = stoi(ageStr);

        if (isPassengerQueried({name, airline, location, age}, queryParams[0], queryParams[1], stoi(queryParams[2]), stoi(queryParams[3]))) {
            addPassenger(passengers, {name, airline, location, age}, arrCapacity, recordIdx, doublingCounter);
        }
    }
}
bool isPassengerQueried(AirlinePassenger passenger, string queryLocation, string queryAirline, int startAge, int endAge) {
    // Check if the passenger's location matches the queryLocation
    if (passenger.location != queryLocation) {
        return false;
    }
    // Check if the passenger's airline matches the queryAirline
    if (passenger.airline != queryAirline) {
        return false;
    }
    // Check if the passenger's age is within the query age range (inclusive)
    if (passenger.age < startAge || passenger.age > endAge) {
        return false;
    }
    // If all conditions are met, return true
    return true;
}


void resizeArr(AirlinePassenger *&passengers, int *arraySize) {
    AirlinePassenger *newArray = new AirlinePassenger[*arraySize * 2];
    for (int i = 0; i < *arraySize; ++i) {
        newArray[i] = passengers[i];
    }
    delete[] passengers;
    passengers = newArray;
    *arraySize *= 2;
}

void addPassenger(AirlinePassenger *&passengers, AirlinePassenger airlinePassenger, int &arrCapacity, int &recordIdx, int &doublingCounter) {
    if (recordIdx == arrCapacity) {
        resizeArr(passengers, &arrCapacity); // Pass the address of arrCapacity
        doublingCounter++;
    }
    passengers[recordIdx++] = airlinePassenger;
}

void sortPassengers(AirlinePassenger *passengers, int length) {
    bool swapped;
    do {
        swapped = false;
        for (int i = 1; i < length; i++) {
            if (passengers[i-1].age < passengers[i].age || 
               (passengers[i-1].age == passengers[i].age && passengers[i-1].name > passengers[i].name)) {
                swap(passengers[i-1], passengers[i]);
                swapped = true;
            }
        }
    } while (swapped);
}

void printQueriedPassengers(AirlinePassenger *passengers, int numOfRecords) {
    sortPassengers(passengers, numOfRecords);
    cout << "Queried Passengers\n---------------------------------------" << endl;
    for (int i = 0; i < numOfRecords; i++) {
        cout << passengers[i].name << " " << passengers[i].age << endl;
    }
}
