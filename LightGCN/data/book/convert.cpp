#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
map<int, vector<int>> adj;
void get_full_train()
{
    ofstream output_file("full_train.txt");
    for (const auto &[user_id, items] : adj)
    {
        string line = to_string(user_id);
        for (int item_id : items)
        {
            line.push_back(' ');
            line += to_string(item_id);
        }
        line.push_back('\n');
        output_file << line;
    }
    output_file.close();
}
void get_train_test()
{
    ofstream train_file("train.txt"), test_file("test.txt");
    for (const auto &[user_id, items] : adj)
    {
        int n = items.size(), train_set_num = n - n / 5;
        string line = to_string(user_id);
        for (int i = 0; i < train_set_num; ++i)
        {
            line.push_back(' ');
            line += to_string(items[i]);
        }
        line.push_back('\n');
        train_file << line;
        line = to_string(user_id);
        for (int i = train_set_num; i < n; ++i)
        {
            line.push_back(' ');
            line += to_string(items[i]);
        }
        line.push_back('\n');
        test_file << line;
    }
    train_file.close();
    test_file.close();
}
int main()
{
    ifstream input_file("./user_item.csv");
    string line;
    if (input_file.is_open())
    {
        while (getline(input_file, line))
        {
            int pos = line.find(',');
            adj[stoi(line.substr(0, pos))].push_back(stoi(line.substr(pos + 1)));
        }
        input_file.close();
        get_train_test();
    }
    else
        cerr << "Open error!" << endl;
    return 0;
}