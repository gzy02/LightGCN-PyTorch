#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
vector<unordered_set<int>> generate_user_item_map(const string &train_data_path)
{
    vector<unordered_set<int>> ans;
    ifstream input_file(train_data_path);
    string line;
    if (input_file.is_open())
    {
        while (getline(input_file, line))
        {
            string item_id;
            unordered_set<int> interact;
            int begin = 0, n = line.size();
            while (begin != n && line[begin++] != ' ')
                ;
            while (begin != n)
            {
                char ch = line[begin++];
                if (ch == ' ')
                {
                    interact.insert(stoi(item_id));
                    item_id.clear();
                }
                else
                    item_id.push_back(ch);
            }

            if (item_id.size())
                interact.insert(stoi(item_id));
            ans.push_back(interact);
        }
        input_file.close();
    }
    else
        cerr << "Open error!" << endl;
    return ans;
}
vector<int> get_test_user_list(const string &path)
{
    vector<int> ans;
    ifstream input_file(path);
    if (input_file.is_open())
    {
        string line;
        getline(input_file, line);
        while (getline(input_file, line))
        {
            ans.push_back(stoi(line));
        }
        input_file.close();
    }
    else
        cerr << "Open error!" << endl;
    return ans;
}
int get_item_num(const vector<unordered_set<int>> &user_item_map)
{
    int ans = 0;
    for (const auto &items : user_item_map)
    {
        for (int item : items)
        {
            ans = max(item, ans);
        }
    }
    return ans + 1;
}
vector<int> get_item_degree(const vector<unordered_set<int>> &user_item_map, int item_num)
{
    vector<int> ans(item_num);
    for (const auto &items : user_item_map)
    {
        for (int item_id : items)
        {
            ++ans[item_id];
        }
    }
    return ans;
}
int get_max_user_degree(const vector<unordered_set<int>> &user_item_map)
{
    size_t ans = 0;
    for (const auto &items : user_item_map)
    {
        ans = max(ans, items.size());
    }
    return ans;
}
vector<vector<int>> generate_sim_matrix_count(const vector<unordered_set<int>> &user_item_map, int item_num)
{

    vector<vector<int>> ans(item_num, vector<int>(item_num));
    for (const auto &items : user_item_map)
    {
        for (int item_id1 : items)
        {
            for (int item_id2 : items)
            {
                ++ans[item_id1][item_id2];
            }
            ans[item_id1][item_id1] = 0;
        }
    }
    return ans;
}

inline float similarE_dis(int count, int popular_A, int popular_B, int)
{
    return 1 / (1 + sqrt(popular_A + popular_B - (count << 1)));
}
inline float similarJ_sim(int count, int popular_A, int popular_B, int)
{
    return count / float(popular_A + popular_B - count);
}
inline float similarCOS(int count, int popular_A, int popular_B, int)
{
    return count / sqrt(popular_A * popular_B);
}
inline float similarP_cov(int count, int popular_A, int popular_B, int length)
{
    return (length * count - popular_A * popular_B) /
           sqrt((long)popular_A * popular_B * (length - popular_A) * (length - popular_B));
}

using SimilarityFuncPtr = float (*)(int, int, int, int);

vector<vector<float>> generate_sim_matrix(const vector<vector<int>> &sim_matrix_count, const vector<int> &item_degree, const vector<unordered_set<int>> &user_item_map, int item_num, SimilarityFuncPtr simFunc)
{
    int user_num = user_item_map.size();
    vector<vector<float>> ans(item_num, vector<float>(item_num));
    for (int i = 0; i < item_num; ++i)
    {
        for (int j = i + 1; j < item_num; ++j)
        {
            ans[j][i] = ans[i][j] = simFunc(sim_matrix_count[i][j], item_degree[i], item_degree[j], user_num);
        }
    }
    return ans;
}
// 使用vector<pair>形式返回按sim_value从大到小排序k个的index-sim_value键值对
vector<pair<int, float>> getTopk(const vector<float> &sim_matrix, int k)
{
    // 创建pair数组，存储sim_value和index的pair
    vector<pair<float, int>> sim_values(sim_matrix.size());
    for (int i = 0; i < sim_matrix.size(); i++)
    {
        sim_values[i] = {sim_matrix[i], i};
    }

    // 使用partial_sort返回前k个top matches
    partial_sort(sim_values.begin(), sim_values.begin() + k, sim_values.end(), greater<pair<float, int>>());

    // 获取前k个top matches，转换为index-sim_value键值对形式
    vector<pair<int, float>> top_k(k);
    for (int i = 0; i < k; i++)
    {
        int index = sim_values[i].second;
        top_k[i] = {index, sim_matrix[index]};
    }

    return top_k;
}

vector<vector<pair<int, float>>> getTopkSimMatrix(const vector<vector<float>> &sim_matrix, int k)
{
    int item_num = sim_matrix.size();
    vector<vector<pair<int, float>>> ans(item_num);
    for (int i = 0; i < item_num; ++i)
    {
        ans[i] = getTopk(sim_matrix[i], k);
    }
    return ans;
}

vector<int> getTopkByValue(const unordered_map<int, float> &items, int k)
{

    // 将 items 中的 key-value 对存储在 vector 中
    vector<pair<int, float>> kv_pairs(items.begin(), items.end());
    if (k > items.size())
    {
        sort(kv_pairs.begin(), kv_pairs.end(), [](auto &a, auto &b)
             { return a.second > b.second; });
    }
    else
    {
        partial_sort(kv_pairs.begin(), kv_pairs.begin() + k, kv_pairs.end(), [](auto &a, auto &b)
                     { return a.second > b.second; });
    }

    // 返回前 k 个 key，即为 top-k value 对应的 item_id
    int n = min(k, (int)items.size());
    vector<int> ans(n);
    for (int i = 0; i < n; ++i)
    {
        ans[i] = kv_pairs[i].first;
    }

    return ans;
}
unordered_set<int> set_intersection(const unordered_set<int> &s1, const unordered_set<int> &s2)
{
    unordered_set<int> ans;
    for (auto x : s1)
    {
        if (s2.find(x) != s2.cend())
        {
            ans.insert(x);
        }
    }
    return ans;
}

int main()
{
    // 对每个用户看过的item, 找到最相似的n个item, 最终推荐topk个给用户
    const int topk = 10;
    const unordered_map<string, SimilarityFuncPtr> sim_map{{"E_dis", similarE_dis}, {"J_sim", similarJ_sim}, {"COS", similarCOS}, {"P_cov", similarP_cov}};

    string train_data_path = "./book/full_train.txt";
    string test_data_path = "./book/test_dataset.csv";
    vector<std::unordered_set<int>> train_data_map = generate_user_item_map(train_data_path);
    auto test_user_list = get_test_user_list(test_data_path);
    int user_num = train_data_map.size();
    cout << "user_num: " << user_num << endl;
    int item_num = get_item_num(train_data_map);
    cout << "item_num: " << item_num << endl;

    auto sim_matrix_count = generate_sim_matrix_count(train_data_map, item_num);
    auto item_degree = get_item_degree(train_data_map, item_num);
    int max_user_degree = get_max_user_degree(train_data_map);
    cout << "max_user_degree: " << max_user_degree << endl;
    for (const auto &[sim_para, func] : sim_map)
    {
        cout << sim_para << endl;
        auto sim_matrix = generate_sim_matrix(sim_matrix_count, item_degree, train_data_map, item_num, func);
        int N = 50;
        while (N)
        {
            cout << "N = " << N << endl;
            string submit_path = "./book/submit/" + sim_para + "/" + sim_para + "_" + to_string(N) + ".csv";
            ofstream output_file(submit_path);
            output_file << "user_id,item_id\n";
            auto topkSimMatrix = getTopkSimMatrix(sim_matrix, N);
            --N;

            for (int user_id : test_user_list)
            {
                unordered_map<int, float> rec_items;
                const unordered_set<int> &interact_items = train_data_map[user_id];
                for (int item_id : interact_items)
                {
                    for (const auto &[sim_item_id, sim_value] : topkSimMatrix[item_id])
                    {
                        if (interact_items.find(sim_item_id) == interact_items.cend())
                        {
                            rec_items[sim_item_id] += sim_value;
                        }
                    }
                }
                vector<int> rec_list = getTopkByValue(rec_items, topk); // 得到待推荐item列表
                for (int item_id : rec_list)
                {
                    output_file << to_string(user_id) + "," + to_string(item_id) + "\n";
                }
            }
            output_file.close();
        }
    }
    return 0;
}