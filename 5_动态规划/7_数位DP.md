# 数位DP

## 主要解决的问题: 

​		在一段区间 [L, R] 上：

1. 满足某些条件的数字个数
2. 将 x∈[L,R] 代到一个函数 f(x) 中，一个数字 x 的 f(x) 值为一次贡献的量，求总的贡献

时间复杂度一般是O(lgL)



## DP设计

DP状态设计: 

`dp[pos, lim]`       

​	`pos` 为当前的数位 N-1 ~ 0 

​	`lim` 表示 [N-1位, pos+1位] 是否顶到上界

​	pos 到 -1 的时候可以 return 1，使得个位的枚举有效

DP状态转移:

```cpp
dp[pos][lim]: 
dp[pos][0] = 10 * dp[pos - 1][0]
dp[pos][1] = digits[i] * dp[pos - 1][0] + dp[pos - 1][1]

```

前导零会对结果产生影响时，加一维 zero

可能需要带上前缀的某种状态 state，此状态可能影响当前位的枚举，也可能影响当前位枚举的值对答案的贡献







# 满足某些条件的数字个数

#### [902. 最大为 N 的数字组合](https://leetcode-cn.com/problems/numbers-at-most-n-given-digit-set/)

给定集合D, 问可用D写出的<=N的数字的数目

D = {1, 3, 5} 可以写出 111, 135135, 1, 35, 555

A:

```cpp
int getdp(int pos, int lim, const vector<int>& digits, const set<int>& num_set, vector<vector<int>>& dp)
{
    if(pos == -1) return 1;
    if(dp[pos][lim] != -1)
        return dp[pos][lim];
    dp[pos][lim] = 0;
    int up = lim ? digits[pos] : 9; // 当前要枚举到的上界
    for(int i: num_set) // 枚举当前位所有可能数字
    {
        if(i > up)
            break;
        dp[pos][lim] += getdp(pos - 1, lim && i == up, digits, num_set, dp); // 本位被限制且选顶到上界的数字,下一位才被限制
    }
    return dp
        [pos][lim];
}
```





# 求贡献f(x)

#### [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

1 <= n < 2^31^

A:

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        
        long res = 0, base = 1;
        
        while(base <= n) {
            long a = n/base/10;
            long b = n%base;
            long cur = (n/base)%10;
			
            if(cur==0) {
                res += a*base; // 只有"前面没拉满"
            } else if(cur==1) {
                res += (a*base)+(b+1); // "前面没拉满" + "前面拉满"
            } else {
                res += (a+1)*base; // "前面没拉满" + "前面拉满"
            }

            base = base*10;
        }
        
        return res;
    }
};
```

