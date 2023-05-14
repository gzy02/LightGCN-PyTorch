/*
<%
cfg['compiler_args'] = ['-std=c++17', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
namespace py = pybind11;

int global_seed;
class RamdomItem
{
private:
    mt19937 gen;
    uniform_int_distribution<int> dis;

public:
    RamdomItem(int num, int seed = global_seed) : gen(seed), dis(0, num - 1)
    {
    }

    int pickIndex()
    {
        return dis(gen);
    }
};

class RamdomWeight
{
private:
    mt19937 gen;
    uniform_int_distribution<int> dis;
    vector<int> pre;

public:
    RamdomWeight(vector<int> &w, int seed = global_seed) : gen(global_seed), dis(1, accumulate(w.begin(), w.end(), 0))
    {
        partial_sum(w.begin(), w.end(), back_inserter(pre));
    }

    int pickIndex()
    {
        int x = dis(gen);
        return lower_bound(pre.begin(), pre.end(), x) - pre.begin();
    }
};
vector<tuple<int, int, int>> sample_randomBCE(unordered_map<int, unordered_set<int>> user_item_map, int item_num, int fake_num)
{
    vector<tuple<int, int, int>> result;
    RamdomItem randItem(item_num);

    for (const auto &[user, items] : user_item_map)
    {
        int neg_item_id, need = items.size() * fake_num;
        vector<int> neg_item_list(need);
        while (need--)
        {
            do
            {
                neg_item_id = randItem.pickIndex();
            } while (items.find(neg_item_id) != items.cend());
            result.emplace_back(user, neg_item_id, 0);
        }
        for (int item_id : items)
        {
            result.emplace_back(user, item_id, 1);
        }
    }

    return result;
}
vector<tuple<int, int, int>> sample_randomBPR(unordered_map<int, unordered_set<int>> user_item_map, int item_num)
{
    vector<tuple<int, int, int>> result;
    RamdomItem randItem(item_num);

    for (const auto &[user, items] : user_item_map)
    {
        int neg_item_id, need = items.size();
        vector<int> neg_item_list(need);
        while (need--)
        {
            do
            {
                neg_item_id = randItem.pickIndex();
            } while (items.find(neg_item_id) != items.cend());
            neg_item_list[need] = neg_item_id;
        }
        for (int item_id : items)
        {
            result.emplace_back(user, item_id, neg_item_list[++need]);
        }
    }

    return result;
}
vector<tuple<int, int, int>> sample_weightBCE(unordered_map<int, unordered_set<int>> user_item_map, vector<int> item_weights, int fake_num)
{
    vector<tuple<int, int, int>> result;
    RamdomWeight randw(item_weights);

    for (const auto &[user, items] : user_item_map)
    {
        int neg_item_id, need = items.size() * fake_num;
        vector<int> neg_item_list(need);
        while (need--)
        {
            do
            {
                neg_item_id = randw.pickIndex();
            } while (items.find(neg_item_id) != items.cend());
            result.emplace_back(user, neg_item_id, 0);
        }
        for (int item_id : items)
        {
            result.emplace_back(user, item_id, 1);
        }
    }
    return result;
}
vector<tuple<int, int, int>> sample_weightBPR(unordered_map<int, unordered_set<int>> user_item_map, vector<int> item_weights)
{
    // bpr采样三元组
    vector<tuple<int, int, int>> result;
    RamdomWeight randw(item_weights);

    for (const auto &[user, items] : user_item_map)
    {
        int neg_item_id, need = items.size();
        vector<int> neg_item_list(need);
        while (need--)
        {
            do
            {
                neg_item_id = randw.pickIndex();
            } while (items.find(neg_item_id) != items.cend());
            neg_item_list[need] = neg_item_id;
        }
        for (int item_id : items)
        {
            result.emplace_back(user, item_id, neg_item_list[++need]);
        }
    }
    return result;
}
void set_seed(unsigned int seed)
{
    global_seed = seed;
}
PYBIND11_MODULE(my_sample, m)
{
    m.doc() = "sample module";
    m.def("set_seed", &set_seed, "generate int between [0 end]", "end");
    // 绑定函数 按权重采样
    m.def("sample_weightBPR", &sample_weightBPR, "a function that accepts a dict and a list, returns a list of tuples");
    // 绑定函数 随机采样
    m.def("sample_randomBPR", &sample_randomBPR, "a function that accepts a dict and an int and returns a list of tuples");
    m.def("sample_weightBCE", &sample_weightBCE, "a function that accepts a dict, a list and an int, returns a list of tuples");
    // 绑定函数 随机采样
    m.def("sample_randomBCE", &sample_randomBCE, "a function that accepts a dict, an int and an int and returns a list of tuples");
}
