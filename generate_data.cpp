#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <optional>

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
    }

    outfile.close();
    cout<< "Successfully wrote data to file"<< filename << endl;
}

 optional<vector<int>> readDataFromFile(const string& filename) {

    if (filename.empty()) {
        cerr << "Couldn't read data. Filename is empty." << endl;
    }
    cout << "TEST "<< filename << endl;
    ifstream infile(filename);
    cout << "TEST2";

    if(!infile.is_open()) {
        cerr << "Error opening a file: " << filename << endl;
        return nullopt;
    }

    vector<int> file_array;
    int read_value;

    while( infile >> read_value) {
        cout<< "Data from file: "<< read_value << endl;
        file_array.push_back(read_value);
    }

    infile.close();

    cout<< "Successfully read data from file"<< filename << endl;

    return file_array;


}