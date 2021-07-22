# 区间DP

dp[i, j] 表示 [i...j] 上问题的解



# 回文相关问题

最长回文子串

找到S中最长的回文子串

A1: O(n * n)

f[i, j] = f[i+1, j-1] && (si==sj)



A2: (中心拓展算法) O(n * n)

f[i, j] <- f[i+1, j-1] <- f[i+2, j-2] <- .... <- 某一边界情况 (i == j || i+1 == j)



A3: (manacher马拉车算法) O(n)

先加'#', 使都变成找奇数串.

已知 j 的臂长m, 求 i 的臂长n. j为右臂在最右边的中心.

若j<i<j+m, 则 n=min(j+m-i, (2j-i)的臂长 )



Q: 最长回文子序列

A: f[i, j] 表示 s的[i...j]中最长的回文子序列的长度

若si==sj, 则f[i, j] = f[i+1, j-1] + 2

否则, f[i, j] = max( f[i, j-1], f[i-1, j] )



Q: 统计不同回文子字符串

"bccb"有6个. b, c, bcb, bb, cc, bccb

f[i, j] - [i...j]原问题的解

(1)若si==sj,

​	若[i+1, j-1]中没有si, f[i, j] = 2f[i+1, j-1]+2

​	若[i+1, j-1]中有一个si, f[i, j] = 2f[i+1, j-1]+1

​	若[i+1, j-1]中>=2个si, f[i, j] = 2f[i+1, j-1]-f[l+1, r-1], (l,r为[i+1,j-1]中第一个/倒一等于si的索引)

(2)否则,

​	f[i, j] = f[i+1, j] + f[i, j-1] - f[i+1, j-1]





# 区间DP其他问题

戳气球

n个气球, 一个一个戳, 戳第 i 个气球, 得到nums[i] * nums[i+1] * nums[i-1] 个硬币

逆向思维,戳气球 <==> n个气球, 一个一个放

dp[i, j] 填满开区间(i, j) 能得到的最多硬币数

dp[i, j] = max{ nums[i] * nums[k] * nums[j] + dp[i, k] + dp[k, j] } ,  i<k<j



奇怪的打印机

每次选定[L, R, ch] [L, R] = ch

dp[i, j] 打印 [i....j] 所需的次数

若k=i, 则 dp[i, j] = dp[k+1, j] + 1

dp[i, j] = min{ dp[i, k-1] + dp[k+1, j] },  i<k<=j && s[i]==s[k]



移除盒子

每轮可以del掉连续的相同颜色盒子k个, 得分 k*k

f[l, r, k] 表示移除区间[l...r]元素加上区间右边等于ar的k个元素的最大得分

f[l, r, k] = max{ f[l, r-1, 0]+(k+1) * (k+1) ,  max{ f[l, i, k+1]+f[i+1, r-1, 0],    ai==ar&&l<=i<r }   }

ans =  f[1, n, 0]



合并石头的最低成本

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



Q: 预测赢家 

2个人玩, 每次以首/尾取走得分

dp[i, j] [i...j]先手比后手多得的最大分

dp[i, j] = max{ nums[i]-dp[i+1, j], nums[j]-dp[i, j-1] }



Q: 编码最短长度的字符串

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



