#include "../code_1/ShowCatalog.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
traceNode *traceHead = NULL;
bool verbose = false;
int numAllocs = 0;

static const char *USAGE = " [filename]\n\
\tread shows from filename and build an interactive bst";

//
// DO NOT MODIFY THIS FILE.
//

int main(int argc, char *argv[])
{
	ifstream dataFile;
	string dataLine;

	if (argc > 2)
	{
		freopen(argv[2], "r", stdin);
	}

	ShowCatalog *shows = new ShowCatalog();

	if (argc < 2)
	{
		cerr << argv[0] << USAGE << endl;
		return 1;
	}

	dataFile.open(argv[1]);

	if (!dataFile.is_open())
	{
		cerr << "Could not open " << argv[1] << " for reading!" << endl;
		return 2;
	}

	while (getline(dataFile, dataLine))
	{
        string title;
        int year;
        string showRating;
        float userRating;

		string datum;
		stringstream dataStream(dataLine);

        // title
		getline(dataStream, title, ',');

        // releaseYear
		getline(dataStream, datum, ',');
		year = stoi(datum);

		// ranking
		getline(dataStream, datum, ',');
		showRating = datum;

		// rating
		getline(dataStream, datum, ',');
		userRating = stof(datum);

		shows->addShowItem(title,year,showRating,userRating);
	}

	bool quit = false;
	while (!quit)
	{
		int option;
		string inputLine;

		cout << "======Main Menu======" << endl;
		cout << "1. Delete a show" << endl;
		cout << "2. Print show catalog" << endl;
		cout << "3. Right rotate the tree" << endl;
        cout << "4. Left rotate the tree" << endl;
		cout << "5. Quit" << endl;

		getline(cin, inputLine);

		if (inputLine.length() != 1 || inputLine[0] < '0' || inputLine[1] > '4')
		{
			cerr << "Invalid option : " << inputLine << endl;
			continue;
		}

		option = stoi(inputLine);

		switch (option)
		{
			case 1:
			{
				std::string _title;
				cout << "Enter a show title:" << endl;
				getline(cin, _title);
				shows->removeShow(_title);
				break;
			}
			case 2:
			{
				shows->printShowCatalog();
				break;
			}
			case 3:
			{
				cout << "Enter a show title:" << endl;
				string nodename;
				getline(cin, nodename);
                shows->rightRotate(nodename);
				break;
			}
            case 4:
            {
                cout << "Enter a show title:" << endl;
				string nodename;
				getline(cin, nodename);
                shows->leftRotate(nodename);
				break;
            }
			case 5:
			{
				quit = true;
			}
		}
	}

	cout << "Goodbye!" << endl;

	return 0;
}
