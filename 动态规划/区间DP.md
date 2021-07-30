# 区间DP

dp[i, j] 表示 [i...j] 上问题的解



# 回文相关问题

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

找到S中最长的回文子串

A1: O(n * n)

f[i, j] = f[i+1, j-1] && (si==sj)



A2: (中心拓展算法) O(n * n)

f[i, j] <- f[i+1, j-1] <- f[i+2, j-2] <- .... <- 某一边界情况 (i == j || i+1 == j)



A3: (manacher马拉车算法) O(n)

先加'#', 使都变成找奇数串.

已知 j 的臂长m, 求 i 的臂长n. j为右臂在最右边的中心.

若j<i<j+m, 则 n=min(j+m-i, (2j-i)的臂长 )



#### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

A: f[i, j] 表示 s的[i...j]中最长的回文子序列的长度

若si==sj, 则f[i, j] = f[i+1, j-1] + 2

否则, f[i, j] = max( f[i, j-1], f[i-1, j] )



#### [730. 统计不同回文子序列](https://leetcode-cn.com/problems/count-different-palindromic-subsequences/)

"bccb"有6个. b, c, bcb, bb, cc, bccb

f[i, j] - [i...j]原问题的解

(1)若si==sj,

​	若[i+1, j-1]中没有si, f[i, j] = 2f[i+1, j-1]+2

​	若[i+1, j-1]中有一个si, f[i, j] = 2f[i+1, j-1]+1

​	若[i+1, j-1]中>=2个si, f[i, j] = 2f[i+1, j-1]-f[l+1, r-1], (l,r为[i+1,j-1]中第一/倒一等于si的索引)

(2)否则,

​	f[i, j] = f[i+1, j] + f[i, j-1] - f[i+1, j-1]





# 区间DP其他问题

#### [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)

有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。



逆向思维,戳气球 <==> n个气球, 一个一个放

dp[i, j] 填满开区间(i, j) 能得到的最多硬币数

dp[i, j] = max{ nums[i] * nums[k] * nums[j] + dp[i, k] + dp[k, j] } ,  i<k<j

```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> rec(n + 2, vector<int>(n + 2));
        vector<int> val(n + 2);
        val[0] = val[n + 1] = 1;
        for (int i = 1; i <= n; i++) {
            val[i] = nums[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 2; j <= n + 1; j++) {
                for (int k = i + 1; k < j; k++) {
                    int sum = val[i] * val[k] * val[j];
                    sum += rec[i][k] + rec[k][j];
                    rec[i][j] = max(rec[i][j], sum);
                }
            }
        }
        return rec[0][n + 1];
    }
};

```



#### [664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)

每次选定[L, R, ch] [L, R] = ch

dp[i, j] 打印 [i....j] 所需的次数

若k=i, 则 dp[i, j] = dp[k+1, j] + 1

dp[i, j] = min{ dp[i, k-1] + dp[k+1, j] },  i<k<=j && s[i]==s[k]



#### [546. 移除盒子](https://leetcode-cn.com/problems/remove-boxes/)

每轮可以del掉连续的相同颜色盒子k个, 得分 k*k

f[l, r, k] 表示移除区间[l...r]元素加上区间右边等于ar的k个元素的最大得分

f[l, r, k] = max{ f[l, r-1, 0]+(k+1) * (k+1) ,  max{ f[l, i, k+1]+f[i+1, r-1, 0],    ai==ar&&l<=i<r }   }

ans =  f[1, n, 0]



#### [1000. 合并石头的最低成本](https://leetcode-cn.com/problems/minimum-cost-to-merge-stones/)

将连续k堆和为一堆, 成本为k堆石头总和.

A1:

dp[i, j, k] 合并[i...j]堆为k堆的最低成本

dp[i, j, 1] = dp[i, j, k] + sum[i, j]

dp[i, j, m] = min{ dp[i, j, 1] + dp[p+1, j, m-1] }, i<=p<=j, 2<=m<=k

无法合并时, dp[i, j, k] = MAX

dp[i, i, 1] = 0

ans = dp[1, n, 1]

A2: 

dp[i, j] 尽可能多的合并[i...j]的最低成本

dp[i, j] = min{ dp[i, p]+dp[p+1, j] }, i<=p<j

如果可以继续合并, dp[i, j] += sum[i, j]



#### [486. 预测赢家](https://leetcode-cn.com/problems/predict-the-winner/)

2个人玩, 每次以首/尾取走得分

dp[i, j] [i...j]先手比后手多得的最大分

dp[i, j] = max{ nums[i]-dp[i+1, j], nums[j]-dp[i, j-1] }

#### [471. 编码最短长度的字符串](https://leetcode-cn.com/problems/encode-string-with-shortest-length)

"aaaaa" --> "5[a]"

f[i, j] [i...j] 最短程度的字符串编码

if  s[i, j] = (j-i+1)/(k-i+1)*s[i, k]

​	f[i, j] = (j-i+1)/(k-i+1) [f[i, k]]

else

​	f[i, j] = min { f[i, j], f[i, k] + f[k+1, j] },  i<=k<=(j-1)

判断s是否为重复子串拼成?

​	p = (s+s).find(s, 1)

​	若p!=n, 则有重复, 重复了n/p次; 否则, 没有重复.



# 总结

dp[i, j] [i...j]区间的原问题的解

dp[i, j] = max/min {dp[i, j], dp[i, k]+dp[k+1, j]+cost }

两种编码

(1)

```cpp
for (int i = n; i >= 1; i--) {
	for (int j = i+1; j <= n; j++) {
        for (int k = i; k <= j-1; k++) {
            ......
        }
    }
}
```

(2)更常用

```cpp
for (int len = 2; len <= n; len++) {
	for (int i = 1; i+len-1 <= n; i++) {
        int j = i+len-1;
        for (int k = i; k <= j-1; k++) {
            ......
        }
    }
}
```



