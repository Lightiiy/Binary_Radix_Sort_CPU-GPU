#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <optional>
#include <cmath>

using namespace std;

void writeDataToFile(const string& filename, int max_value, int array_size) {
    if (filename.empty()) {
        cerr << "Couldn't write data. Filename is empty." << endl;
        return; // Return an empty optional
    }
    ofstream outfile(filename);
    if(!outfile.is_open()) {
        cerr << "Error opening a file: " << filename << endl;
        return;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, max_value);

    for(int i =0; i< array_size; i++) {
        outfile << dist(gen) << endl;
        if(i % 10 == 0) {
            printf("Generated %d numbers out of %d\n",i, array_size);
        }
    }

    outfile.close();
    cout<< "Successfully wrote data to file"<< filename << endl;
}

 optional<vector<int>> readDataFromFile(const string& filename) {

    if (filename.empty()) {
        cerr << "Couldn't read data. Filename is empty." << endl;
        return nullopt;
    }
    ifstream infile(filename);

    if(!infile.is_open()) {
        cerr << "Error opening a file: " << filename << endl;
        return nullopt;
    }

    vector<int> file_array;
    int read_value;

    while( infile >> read_value) {
        file_array.push_back(read_value);
    }
    cout << endl;

    infile.close();

    cout<< "Successfully read data from file"<< filename << endl;

    return file_array;
}
