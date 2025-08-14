#include <bits/stdc++.h>
#include <atcoder/all>

#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")

using namespace std;
using namespace atcoder;
using ll = long long;
using ull = unsigned long long;
using vi = vector<int>;
using vvi = vector<vector<int>>;
using vl = vector<ll>;
using vvl = vector<vector<ll>>;
using vs = vector<string>;
using vvs = vector<vector<string>>;
using vc = vector<char>;
using vvc = vector<vector<char>>;
using vb = vector<bool>;
using vvb = vector<vector<bool>>;
const ll mod = 1000000007;
// const ll mod = 998244353;
const ll INF = 1e16;
const string al = "abcdefghijklmnopqrstuvwxyz";
const string au = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
#define rep(i, x) for(ll i=0; i<x; i++)
#define reps(i, a, x) for(ll i=a; i<x; i++)
#define eb emplace_back
#define ef emplace_front
#define all(a) a.begin(),a.end()
#define rall(a) a.rbegin(),a.rend()
#define sum(a) accumulate(all(a),0LL)
#define len(x) (ll)(x).size()
#define pq(T) priority_queue<T,vector<T>,greater<T>>

template <typename T>
inline bool chmax(T &a, T b) {
  return ((a < b) ? (a = b, true) : (false));
}
template <typename T>
inline bool chmin(T &a, T b) {
  return ((a > b) ? (a = b, true) : (false));
}

ll pow(ll x, ll n, ll r=INF) {
  ll res = 1;
  while (n > 0) {
    if (n & 1) res = res * x % r;
    x = x * x % r;
    n >>= 1;
  }
  return res;
}

int main(){

  return 0;
}
