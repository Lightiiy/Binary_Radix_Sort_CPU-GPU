#include <iostream>
#include <ostream>

#include "generate_data.h"
#include <vector>
using namespace std;

int main() {

    const string filename = "basic.txt";

    writeDataToFile(filename, 100, 10);

    optional<vector<int>> opt_data = readDataFromFile(filename);

    if(opt_data) {
        vector<int> test_data = *opt_data;
        cout<<test_data.size()<<endl;
    }
    else {
        cout << "No data found" << endl;
    }



    return 0;
}