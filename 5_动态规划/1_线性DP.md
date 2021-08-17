# 状态

```
状态定义：	dp[n] := [0..n] 上问题的解
状态转移：	dp[n] = f(dp[n-1], ..., dp[0])
```

# 单串问题

状态一般定义为 dp[i] := 考虑[0..i]上，原问题的解，其中 i 位置的处理，根据不同的问题，主要有两种方式：

- 第一种是 i 位置必须取，此时状态可以进一步描述为 dp[i] := 考虑[0..i]上，且取 i，原问题的解；
- 第二种是 i 位置可以取可以不取

## 最经典单串 LIS 系列

#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。



A1 : DP O(N^2^)

定义 dp[i] 为考虑前 i 个元素，以第 i 个数字结尾的最长上升子序列的长度.注意 nums[i] 必须被选中

`dp[i] = max(dp[j]) + 1`,其中0≤j<i且num[j]<num[i]

LIS~length~=max(dp[i]), 其中0 <= i < n

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = (int)nums.size();
        if (n == 0) {
            return 0;
        }
        vector<int> dp(n, 0);
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[j] < nums[i]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

A2：贪心 + 二分查找  O（NlogN）

我们维护一个数组 `d[i]`，表示长度为 i 的最长上升子序列的末尾元素的最小值，关于i递增.

设当前已求出的最长上升子序列的长度为 `len`（初始时为 11），从前往后遍历数组 `nums`，在遍历到 `nums[i]` 时：

如果 `nums[i]>d[len]` ，则直接加入到 d 数组末尾，并更新 `len=len+1`；

否则，在 d 数组中二分查找，找到第一个比 `nums[i]` 小的数 `d[k]` ，并更新 `d[k+1]=nums[i]`。

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = 1, n = (int)nums.size();
        if (n == 0) {
            return 0;
        }
        vector<int> d(n + 1, 0);
        d[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0; // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
        
                d[pos + 1] = nums[i];
            }
            // 这一部分也可以用
            // int pos = lower_bound(d.begin() + 1, d.begin() + len + 1, nums[i]) - d.begin();
            // d[pos] = nums[i], len = max(len, pos);
        }
        return len;
    }
};
```



#### [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)



A1：DP

假设对于以 nums[i] 结尾的序列，我们知道最长序列的长度 length[i]，以及具有该长度的序列的 count[i]。

对于每一个 i<j 和一个 A[i]<A[j]，我们可以将一个 A[j] 附加到以 A[i] 结尾的最长子序列上。

如果这些序列比 length[j] 长，那么我们就知道我们有count[i] 个长度为 length 的序列。如果这些序列的长度与 length[j] 相等，那么我们就知道现在有 count[i] 个额外的序列（即 count[j]+=count[i]）。

```cpp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector <int> len(n + 1, 1), count(n + 1, 1);


        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    if (len[j] + 1 == len[i]) count[i] += count[j];
                    else if (len[j] + 1 > len[i]) len[i] = len[j] + 1, count[i] = count[j];
                }
            }
        }

        int maxlen = 1, cnt = 0;
        for (int i = 0; i < n; i++) {
            if (len[i] < maxlen) continue;
            else if (len[i] == maxlen) cnt += count[i];
            else if (len[i] > maxlen) maxlen = len[i], cnt = count[i];
        }

        return cnt;
    }
};
```



A2: 树状数组

先将`nums`离散化.

树状数组 `A[i]` 表示以数字i结尾, first表示长度, second表示个数

A[1...i] 求和 表示 以小于等于 `i` 的数字结尾

```cpp
class Solution {
    int n;
    vector<pair<int,int>> A;

    int lowerbit(int num){
        return num & -num;
    }

    void update(int id,pair<int,int> val){

        while(id <= n){
            if(A[id].first < val.first){
                A[id].first = val.first;
                A[id].second = val.second;
            }else if(A[id].first == val.first){
                A[id].second += val.second;
            }
            id += lowerbit(id);
        }

    }
	// 查询以 <= id - 1 的数字作为结尾的<最大长度, 最大长度的个数>
    pair<int,int> query(int id){
        int ans = 0;
        int cnt = 1;
        while(id){
            if(A[id].first > ans){
                ans = A[id].first;
                cnt = A[id].second;
            }else if( A[id].first == ans ){
                cnt += A[id].second;
            }
            id -= lowerbit(id);
        }
        return {ans+1,cnt};
    }

public:
    int findNumberOfLIS(vector<int>& nums) {

        vector<pair<int,int>> temp(nums.size());
        for(int i =0;i<nums.size();++i){
            temp[i]= make_pair(nums[i],i);
        }
        sort(temp.begin(),temp.end());
        n=0;
        for(int i =0;i<temp.size();++i){
            if( n==0 || temp[i].first != temp[i-1].first){
                ++n;
            }
            nums[temp[i].second] = n;
        }
        A.resize(n+1,std::pair<int,int>(0,0));
        int cnt = 1, maxTime =0;
        for(int i =0;i<nums.size();++i){
            auto val = query(nums[i]-1);
            // cout << val.first << ":" << val.second << ":" << i <<endl;
            if(val.first > maxTime){
                maxTime = val.first;
                cnt = val.second;
            }else if( val.first == maxTime ){
                cnt += val.second;
            }
            update(nums[i],val);

        }
        return cnt;

    }
};
```



#### [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

A :

排序 + 最长递增子序列

在对信封按 `w` 进行排序以后，我们可以找到 `h` 上最长递增子序列的长度。

```CPP
class Solution {
public:
    static bool cmp (const vector <int> a, const vector <int> b) {
        if (a[0]!=b[0]) return a[0]<b[0];
        return a[1]>b[1];
    }

    int wk(vector <int> & a) {
        int n = a.size();
        vector <int> f;
        for (int i = 0; i < n; i++) {
            int p = lower_bound(f.begin(), f.end(), a[i]) - f.begin();
            if (p == f.size()) {
                f.push_back(a[i]);
            } else {
                f[p] =  a[i];
            }
        }
        return f.size();
    }

    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n = envelopes.size();
        if (n == 0) return 0;
        sort(envelopes.begin(), envelopes.end(), cmp);
        vector <int> f;
        for (int i = 0; i < envelopes.size(); i++) {
                f.push_back(envelopes[i][1]);
            
        } 
        return wk(f);
    }
};
```



## 最大子数组和系列

最大子序和

#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。



A :

```cpp
// 方法1, 动态规划 O(n)
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int pre = 0, maxAns = nums[0];
        for (const auto &x: nums) {
            pre = max(pre + x, x);
            maxAns = max(maxAns, pre);
        }
        return maxAns;
    }
};

// 方法2, 分治,类似线段树 O(n)
class Solution {
public:
    struct Status {
        int lSum, rSum, mSum, iSum;
    };

    Status pushUp(Status l, Status r) {
        int iSum = l.iSum + r.iSum;
        int lSum = max(l.lSum, l.iSum + r.lSum);
        int rSum = max(r.rSum, r.iSum + l.rSum);
        int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
        return (Status) {lSum, rSum, mSum, iSum};
    };

    Status get(vector<int> &a, int l, int r) {
        if (l == r) {
            return (Status) {a[l], a[l], a[l], a[l]};
        }
        int m = (l + r) >> 1;
        Status lSub = get(a, l, m);
        Status rSub = get(a, m + 1, r);
        return pushUp(lSub, rSub);
    }

    int maxSubArray(vector<int>& nums) {
        return get(nums, 0, nums.size() - 1).mSum;
    }
};

```



#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)



```CPP
// maxF[i] 表示以第 i 个元素结尾的乘积最大子数组的乘积
// minF[i] 表示以第 i 个元素结尾的乘积最小子数组的乘积
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        vector <int> maxF(nums), minF(nums);
        for (int i = 1; i < nums.size(); ++i) {
            maxF[i] = max(maxF[i - 1] * nums[i], max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = min(minF[i - 1] * nums[i], min(nums[i], maxF[i - 1] * nums[i]));
        }
        return *max_element(maxF.begin(), maxF.end());
    }
};
```



#### [918. 环形子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-circular-subarray/)

A :

最大和具有两种可能，一种是不使用环的情况，另一种是使用环的情况

不使用环的情况时，直接通过53题的思路，逐步求出整个数组中的最大子序和即可

使用到了环,用整个数组的和 sum减掉最小子序和.

## 打家劫舍系列



#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。





```cpp
// 方法1, 动态规划
// dp[i] 表示 [0...i]且偷第i家的 最高金额
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) {
            return 0;
        }
        int size = nums.size();
        if (size == 1) {
            return nums[0];
        }
        vector<int> dp = vector<int>(size, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < size; i++) {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[size - 1];
    }
};

// 方法2, 利用滚动数组
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) {
            return 0;
        }
        int size = nums.size();
        if (size == 1) {
            return nums[0];
        }
        int first = nums[0], second = max(nums[0], nums[1]);
        for (int i = 2; i < size; i++) {
            int temp = second;
            second = max(first + nums[i], second);
            first = temp;
        }
        return second;
    }
};
```



#### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

环形版打家劫舍

```cpp
// 两种情况
// 1. 不偷最后一家, 别的随便偷
// 2. 不偷第一家, 别的随便偷
class Solution {
public:
    int robRange(vector<int>& nums, int start, int end) {
        int first = nums[start], second = max(nums[start], nums[start + 1]);
        for (int i = start + 2; i <= end; i++) {
            int temp = second;
            second = max(first + nums[i], second);
            first = temp;
        }
        return second;
    }

    int rob(vector<int>& nums) {
        int length = nums.size();
        if (length == 1) {
            return nums[0];
        } else if (length == 2) {
            return max(nums[0], nums[1]);
        }
        return max(robRange(nums, 0, length - 2), robRange(nums, 1, length - 1));
    }
};
```



#### [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/)

给你一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

- `1 <= nums[i] <= 104`

 A :

先计算 `sum[i]` 表示 数字 i 出现的个数.

然后普通的DP就ok

```cpp
class Solution {
private:
    int rob(vector<int> &nums) { // 利用滚动数组节省空间
        int size = nums.size();
        int first = nums[0], second = max(nums[0], nums[1]);
        for (int i = 2; i < size; i++) {
            int temp = second;
            second = max(first + nums[i], second);
            first = temp;
        }
        return second;
    }

public:
    int deleteAndEarn(vector<int> &nums) {
        int maxVal = 0;
        for (int val : nums) {
            maxVal = max(maxVal, val);
        }
        vector<int> sum(maxVal + 1);
        for (int val : nums) {
            sum[val] += val;
        }
        return rob(sum);
    }
};
```



#### [1388. 3n 块披萨](https://leetcode-cn.com/problems/pizza-with-3n-slices/)

给你一个披萨，它由 3n 块不同大小的部分组成，现在你和你的朋友们需要按照如下规则来分披萨：

- 你挑选 任意 一块披萨。
- Alice 将会挑选你所选择的披萨逆时针方向的下一块披萨。
- ob 将会挑选你所选择的披萨顺时针方向的下一块披萨。
- 重复上述过程直到没有披萨剩下。

每一块披萨的大小按顺时针方向由循环数组 slices 表示。

请你返回你可以获得的披萨大小总和的最大值。



A :

本题可以转化成如下问题：

> 给一个长度为 3n 的环状序列，你可以在其中选择 n 个数，并且任意两个数不能相邻，求这 n 个数的最大值。

```cpp
// 方法1, DP O(n*n)
// 用 dp[i][j] 表示在前 i 个数中选择了 j 个不相邻的数的最大和
class Solution {
public:
    int calculate(const vector<int>& slices) {
        int n = slices.size();
        int choose = (n + 1) / 3;
        vector<vector<int>> dp(n + 1, vector<int>(choose + 1));
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= choose; ++j) {
                dp[i][j] = max(dp[i - 1][j], (i - 2 >= 0 ? dp[i - 2][j - 1] : 0) + slices[i - 1]);
            }
        }
        return dp[n][choose];
    }

    int maxSizeSlices(vector<int>& slices) {
        vector<int> v1(slices.begin() + 1, slices.end());
        vector<int> v2(slices.begin(), slices.end() - 1);
        int ans1 = calculate(v1);
        int ans2 = calculate(v2);
        return max(ans1, ans2);
    }
};

// 方法2, 贪心 O(nlogn)
// 每次选择了最大的数之后，将这个数的值修改成其与相邻的两个数的差值
// 假设我们在序列 ⋯,x,y,z,⋯ 中选取了 y，那么我们在删去 x 和 z 的同时，将 y 的值改为 y' = x + z - y
// 意味着可以反悔, y+y′ = y+(x+z−y) = x+z
class Solution {
public:
    int maxSizeSlices(vector<int>& slices) {
        int n = slices.size();
        // 使用数组模拟双向链表
        vector<int> linkL(n);
        vector<int> linkR(n);
        for (int i = 0; i < n; ++i) {
            linkL[i] = (i == 0 ? n - 1 : i - 1);
            linkR[i] = (i == n - 1 ? 0 : i + 1);
        }
        // 将初始的元素放入优先队列中
        vector<int> valid(n, 1);
        priority_queue<pair<int, int>> q;
        for (int i = 0; i < n; ++i) {
            q.emplace(slices[i], i);
        }
        
        int ans = 0;
        for (int i = 0; i < n / 3; ++i) {
            // 从优先队列中取出元素时要判断其是否已被删除
            while (!valid[q.top().second]) {
                q.pop();
            }
            int pos = q.top().second;
            q.pop();
            ans += slices[pos];
            // 更新当前位置的值
            slices[pos] = slices[linkL[pos]] + slices[linkR[pos]] - slices[pos];
            q.emplace(slices[pos], pos);
            // 删去左右两侧的值
            valid[linkL[pos]] = valid[linkR[pos]] = 0;
            // 修改双向链表
            linkR[linkL[linkL[pos]]] = pos;
            linkL[linkR[linkR[pos]]] = pos;
            linkL[pos] = linkL[linkL[pos]];
            linkR[pos] = linkR[linkR[pos]];
        }
        return ans;
    }
};
```



## 其它单串 dp[i] 问题

#### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

A :

```cpp
// 方法1, DP
//  dp[i] 表示以下标 i 字符结尾的最长有效括号的长度
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0, n = s.length();
        vector<int> dp(n, 0);
        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = max(maxans, dp[i]);
            }
        }
        return maxans;
    }
};

// 方法2, 栈模拟
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0;
        stack<int> stk;
        stk.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                stk.push(i);
            } else {
                stk.pop();
                if (stk.empty()) {
                    stk.push(i);
                } else {
                    maxans = max(maxans, i - stk.top());
                }
            }
        }
        return maxans;
    }
};
// 方法3, 不需要额外的空间
// 利用两个计数器 left 和 right , 表示 '(' 和 ')' 数量
class Solution {
public:
    int longestValidParentheses(string s) {
        int left = 0, right = 0, maxlength = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = (int)s.length() - 1; i >= 0; i--) {
            if (s[i] == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = max(maxlength, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }
        return maxlength;
    }
};
```



#### [801. 使序列递增的最小交换次数](https://leetcode-cn.com/problems/minimum-swaps-to-make-sequences-increasing/)

我们有两个长度相等且不为空的整型数组 A 和 B 。

我们可以交换 A[i] 和 B[i] 的元素。注意这两个元素在各自的序列中应该处于相同的位置。

在交换过一些元素之后，数组 A 和 B 都应该是严格递增的

A :

用 `n1` 表示数组 A 和 B 满足前 `i - 1` 个元素分别严格递增，并且 `A[i - 1]` 和 `B[i - 1]` 未被交换的最小交换次数，用 `s1` 表示 `A[i - 1]` 和 `B[i - 1]` 被交换的最小交换次数。当我们知道了 `n1` 和 `s1` 的值之后，我们需要通过转移得到 `n2` 和 `s2`（和之前的定义相同，只不过考虑的是 `A[i]` 和 `B[i]` 这两个元素）的值。

```java
class Solution {
    public int minSwap(int[] A, int[] B) {
        // n: natural, s: swapped
        int n1 = 0, s1 = 1;
        for (int i = 1; i < A.length; ++i) {
            int n2 = Integer.MAX_VALUE, s2 = Integer.MAX_VALUE;
            if (A[i-1] < A[i] && B[i-1] < B[i]) {
                n2 = Math.min(n2, n1);
                s2 = Math.min(s2, s1 + 1);
            }
            if (A[i-1] < B[i] && B[i-1] < A[i]) {
                n2 = Math.min(n2, s1);
                s2 = Math.min(s2, n1 + 1);
            }
            n1 = n2;
            s1 = s2;
        }
        return Math.min(n1, s1);
    }
}
```



#### [871. 最低加油次数](https://leetcode-cn.com/problems/minimum-number-of-refueling-stops/)

> 汽车从起点出发驶向目的地，该目的地位于出发位置东面 target 英里处。
>
> 沿途有加油站，每个 station[i] 代表一个加油站，它位于出发位置东面 `station[i][0]` 英里处，并且有 `station[i][1]` 升汽油。
>
> 假设汽车油箱的容量是无限的，其中最初有 startFuel 升燃料。它每行驶 1 英里就会用掉 1 升汽油。
>
> 当汽车到达加油站时，它可能停下来加油，将所有汽油从加油站转移到汽车中。
>
> 为了到达目的地，汽车所必要的最低加油次数是多少？如果无法到达目的地，则返回 -1 。
>
> 注意：如果汽车到达加油站时剩余燃料为 0，它仍然可以在那里加油。如果汽车到达目的地时剩余燃料为 0，仍然认为它已经到达目的地。
>

A :

```java
// 方法1, 动态规划 O(N * N)
// dp[i] 为加 i 次油能走的最远距离
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        int N = stations.length;
        long[] dp = new long[N + 1];
        dp[0] = startFuel;
        for (int i = 0; i < N; ++i)
            for (int t = i; t >= 0; --t) // 注意倒序
                if (dp[t] >= stations[i][0])
                    dp[t+1] = Math.max(dp[t+1], dp[t] + (long) stations[i][1]);

        for (int i = 0; i <= N; ++i)
            if (dp[i] >= target) return i;
        return -1;
    }
}

// 方法2, 优先队列贪心 O(NlogN)
// 定义 pq（优先队列）为一个保存了驶过加油站油量的最大堆，定义当前油量为 tank。
// 如果当前油量不够抵达下一个加油站，必须得从之前的加油站中找一个来加油，贪心选择最大油量储备的加油站就好了。
class Solution {
    public int minRefuelStops(int target, int tank, int[][] stations) {
        // pq is a maxheap of gas station capacities
        PriorityQueue<Integer> pq = new PriorityQueue(Collections.reverseOrder());
        int ans = 0, prev = 0;
        for (int[] station: stations) {
            int location = station[0];
            int capacity = station[1];
            tank -= location - prev;
            while (!pq.isEmpty() && tank < 0) {  // must refuel in past
                tank += pq.poll();
                ans++;
            }

            if (tank < 0) return -1;
            pq.offer(capacity);
            prev = location;
        }

        // Repeat body for station = (target, inf)
        {
            tank -= target - prev;
            while (!pq.isEmpty() && tank < 0) {
                tank += pq.poll();
                ans++;
            }
            if (tank < 0) return -1;
        }

        return ans;
    }
}
```



#### [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

> 给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。
>
> 已知存在楼层 f ，满足 0 <= f <= n ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。
>
> 每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 x 扔下（满足 1 <= x <= n）。如果鸡蛋碎了，你就不能再次使用它。如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中 重复使用 这枚鸡蛋。
>
> 请你计算并返回要确定 f 确切的值 的 最小操作次数 是多少？
>

A : 

 `k`为鸡蛋数，`n`为楼层数。



**方法1,  DP + 二分 O(knlogn)**

![image-20210816175517943](../../../../Desktop/pictures/image-20210816175517943.png)



![image-20210816175830483](../../../../Desktop/pictures/image-20210816175830483.png)



```cpp
class Solution {
    unordered_map<int, int> memo;
    int dp(int k, int n) {
        if (memo.find(n * 100 + k) == memo.end()) {
            int ans;
            if (n == 0) {
                ans = 0;
            } else if (k == 1) {
                ans = n;
            } else {
                int lo = 1, hi = n;
                while (lo + 1 < hi) {
                    int x = (lo + hi) / 2;
                    int t1 = dp(k - 1, x - 1);
                    int t2 = dp(k, n - x);

                    if (t1 < t2) {
                        lo = x;
                    } else if (t1 > t2) {
                        hi = x;
                    } else {
                        lo = hi = x;
                    }
                }

                ans = 1 + min(max(dp(k - 1, lo - 1), dp(k, n - lo)),
                                   max(dp(k - 1, hi - 1), dp(k, n - hi)));
            }

            memo[n * 100 + k] = ans;
        }

        return memo[n * 100 + k];
    }
public:
    int superEggDrop(int k, int n) {
        return dp(k, n);
    }
};

```

**DP + 决策单调性**

![image-20210816175859277](../../../../Desktop/pictures/image-20210816175859277.png)



我们固定 k 时，随着 n 的增加，dp(k, n) 对应的最优解的坐标 x ~opt~  单调递增，这样一来每个 dp(k, n)  的均摊时间复杂度为 O(1)。

```cpp
class Solution {
public:
    int superEggDrop(int k, int n) {
        int dp[n + 1];
        for (int i = 0; i <= n; ++i) {
            dp[i] = i;
        }

        for (int j = 2; j <= k; ++j) {
            int dp2[n + 1];
            int x = 1; 
            dp2[0] = 0;
            for (int m = 1; m <= n; ++m) {
                while (x < m && max(dp[x - 1], dp2[m - x]) >= max(dp[x], dp2[m - x - 1])) {
                    x++;
                }
                dp2[m] = 1 + max(dp[x - 1], dp2[m - x]);
            }
            for (int m = 1; m <= n; ++m) {
                dp[m] = dp2[m];
            }
        }
        return dp[n];
    }
};
```

**数学 ( 逆向思维 )**

如果我们可以做 t 次操作，而且有 k 个鸡蛋，那么我们能找到答案的最高的 n 是多少？我们设 f(t, k)为在上述条件下的 n。如果我们求出了所有的 f(t,k)，那么只需要找出最小的满足f(t,k) ≥ n 的 t。

*f*(*t*,*k*)=1+*f*(*t*−1,*k*−1)+*f*(*t*−1,*k*)

```cpp
class Solution {
public:
    int superEggDrop(int k, int n) {
        if (n == 1) {
            return 1;
        }
        vector<vector<int>> f(n + 1, vector<int>(k + 1));
        for (int i = 1; i <= k; ++i) {
            f[1][i] = 1;
        }
        int ans = -1;
        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= k; ++j) {
                f[i][j] = 1 + f[i - 1][j - 1] + f[i - 1][j];
            }
            if (f[i][k] >= n) {
                ans = i;
                break;
            }
        }
        return ans;
    }
};
```





# 双串问题

dp[i, j] - s1考虑[0...i], s2考虑[0...j]时原问题的解

dp[i, j] = f( dp[i-1, j], dp[i-1, j-1], dp[i, j-1] )

## 最经典双串 LCS 系列

#### [1143. 最长公共子序列 LCS](https://leetcode-cn.com/problems/longest-common-subsequence/)

> 给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

```cpp
// dp[i, j] s1考虑[0...i], s2考虑[0...j]时LCS
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.length(), n = text2.length();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; i++) {
            char c1 = text1.at(i - 1);
            for (int j = 1; j <= n; j++) {
                char c2 = text2.at(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
}; 
```





## 字符串匹配系列

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

> 给定一个字符串 (`s`) 和一个字符模式 (`p`) ，实现一个支持 `'?'` 和 `'*'` 的通配符匹配。
>
> ```
> '?' 可以匹配任何单个字符。
> '*' 可以匹配任意字符串（包括空字符串）。
> ```
>
> 两个字符串**完全匹配**才算匹配成功。

![image-20210816180708478](../../../../Desktop/pictures/image-20210816180708478.png)

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p[i - 1] == '*') {
                dp[0][i] = true;
            }
            else {
                break;
            }
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 1] | dp[i - 1][j];
                }
                else if (p[j - 1] == '?' || s[i - 1] == p[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }
};
```



#### [剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

> 模式中的字符`'.'`表示任意一个字符，而`'*'`表示它前面的字符可以出现任意次（含0次）。
>
> 匹配是指字符串的所有字符匹配整个模式。例如，字符串`"aaa"`与模式`"a.a"`和`"ab*ac*a"`匹配，但与`"aa.a"`和`"ab*a"`均不匹配。
>



用 `f[i][j]` 表示 `s` 的前 `i` 个字符与 `p` 中的前 `j` 个字符是否能够匹配。

![image-20210809212207483](../../../../Desktop/pictures/image-20210809212207483.png)

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i, int j) {
            if (i == 0) {
                return false;
            }
            if (p[j - 1] == '.') {
                return true;
            }
            return s[i - 1] == p[j - 1];
        };

        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    f[i][j] |= f[i][j - 2];
                    if (matches(i, j - 1)) {
                        f[i][j] |= f[i - 1][j];
                    }
                }
                else {
                    if (matches(i, j)) {
                        f[i][j] |= f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }
};
```



## 例题

#### [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

> 给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
>
> 两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：
>
> - s = s1 + s2 + ... + sn
>
> - t = t1 + t2 + ... + tm
> - |n - m| <= 1 
> - 交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...



![image-20210816181310958](../../../../Desktop/pictures/image-20210816181310958.png)

```cpp
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        auto f = vector < vector <int> > (s1.size() + 1, vector <int> (s2.size() + 1, false));

        int n = s1.size(), m = s2.size(), t = s3.size();

        if (n + m != t) {
            return false;
        }

        f[0][0] = true;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                int p = i + j - 1;
                if (i > 0) {
                    f[i][j] |= (f[i - 1][j] && s1[i - 1] == s3[p]);
                }
                if (j > 0) {
                    f[i][j] |= (f[i][j - 1] && s2[j - 1] == s3[p]);
                }
            }
        }

        return f[n][m];
    }
};
```



不同的子序列

计算s的子序列中t出现的个数

f[i, j] - s[i...] 的子序列中 t[j...]出现的个数

f[i, j] = dp[i+1, j-1] + dp[i+1, j] ,  s[i] == t[j]

f[i, j] = dp[i+1, j] , otherwise





#### [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/)

> 使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
>
> - 如果字符串的长度为 1 ，算法停止
> - 如果字符串的长度 > 1 ，执行下述步骤：
>   - 在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
>   - 随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
>   - 在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。
>
> 给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。



![image-20210816181608589](../../../../Desktop/pictures/image-20210816181608589.png)



![image-20210816181616876](../../../../Desktop/pictures/image-20210816181616876.png)



![image-20210816181629087](../../../../Desktop/pictures/image-20210816181629087.png)

```cpp
class Solution {
private:
    // 记忆化搜索存储状态的数组
    // -1 表示 false，1 表示 true，0 表示未计算
    int memo[30][30][31];
    string s1, s2;

public:
    bool checkIfSimilar(int i1, int i2, int length) {
        unordered_map<int, int> freq;
        for (int i = i1; i < i1 + length; ++i) {
            ++freq[s1[i]];
        }
        for (int i = i2; i < i2 + length; ++i) {
            --freq[s2[i]];
        }
        if (any_of(freq.begin(), freq.end(), [](const auto& entry) {return entry.second != 0;})) {
            return false;
        }
        return true;
    }

    // 第一个字符串从 i1 开始，第二个字符串从 i2 开始，子串的长度为 length，是否和谐
    bool dfs(int i1, int i2, int length) {
        if (memo[i1][i2][length]) {
            return memo[i1][i2][length] == 1;
        }

        // 判断两个子串是否相等
        if (s1.substr(i1, length) == s2.substr(i2, length)) {
            memo[i1][i2][length] = 1;
            return true;
        }

        // 判断是否存在字符 c 在两个子串中出现的次数不同
        if (!checkIfSimilar(i1, i2, length)) {
            memo[i1][i2][length] = -1;
            return false;
        }
        
        // 枚举分割位置
        for (int i = 1; i < length; ++i) {
            // 不交换的情况
            if (dfs(i1, i2, i) && dfs(i1 + i, i2 + i, length - i)) {
                memo[i1][i2][length] = 1;
                return true;
            }
            // 交换的情况
            if (dfs(i1, i2 + length - i, i) && dfs(i1 + i, i2, length - i)) {
                memo[i1][i2][length] = 1;
                return true;
            }
        }

        memo[i1][i2][length] = -1;
        return false;
    }

    bool isScramble(string s1, string s2) {
        memset(memo, 0, sizeof(memo));
        this->s1 = s1;
        this->s2 = s2;
        return dfs(0, 0, s1.size());
    }
};
```



# 矩阵问题

## 矩阵 dp[i] [j]  

#### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

> 给定一个包含非负整数的 `*m x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
>
> **说明：**每次只能向下或者向右移动一步。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.size() == 0 || grid[0].size() == 0) {
            return 0;
        }
        int rows = grid.size(), columns = grid[0].size();
        auto dp = vector < vector <int> > (rows, vector <int> (columns));
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < columns; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < columns; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][columns - 1];
    }
};
```





#### [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

> 骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。
>
> 有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。
>
> 为了尽快到达公主，骑士决定每次只向右或向下移动一步。
>
> 编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。

```cpp
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int n = dungeon.size(), m = dungeon[0].size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, INT_MAX));
        dp[n][m - 1] = dp[n - 1][m] = 1;
        for (int i = n - 1; i >= 0; --i) {
            for (int j = m - 1; j >= 0; --j) {
                int minn = min(dp[i + 1][j], dp[i][j + 1]);
                dp[i][j] = max(minn - dungeon[i][j], 1);
            }
        }
        return dp[0][0];
    }
};
```



#### [面试题 17.24. 最大子矩阵](https://leetcode-cn.com/problems/max-submatrix-lcci/)

选定子矩阵上边界L, 下边界R, 压缩二维到一维. 然后求最大子序和.



#### [363. 矩形区域不超过 K 的最大数值和](https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/)

选定子矩阵上边界L, 下边界R, 压缩二维到一维. 然后求不超过K的最大子序和.





## 矩阵 dp[i] [j] [k] 

#### [85. 最大矩形 只含1](https://leetcode-cn.com/problems/maximal-rectangle/)

> 给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。



```cpp
// 方法1, DP O(m * m * n)
// left[i][j] 为矩阵第 i 行第 j 列元素的左边连续 1 的数量。
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }

        int ret = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) { //选定矩阵右下角
                if (matrix[i][j] == '0') {
                    continue;
                }
                int width = left[i][j];
                int area = width;
                for (int k = i - 1; k >= 0; k--) {  // 选定上边界
                    width = min(width, left[k][j]);
                    area = max(area, (i - k + 1) * width);
                }
                ret = max(ret, area);
            }
        }
        return ret;
    }
};

// 方法2, 单调栈 O(m * n)
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].size();
        vector<vector<int>> left(m, vector<int>(n, 0));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0: left[i][j - 1]) + 1;
                }
            }
        }

        int ret = 0;
        for (int j = 0; j < n; j++) { // 对于每一列，使用基于柱状图的方法 (确定右边界)
            vector<int> up(m, 0), down(m, 0);

            stack<int> stk;
            for (int i = 0; i < m; i++) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                up[i] = stk.empty() ? -1 : stk.top();
                stk.push(i);
            }
            stk = stack<int>();
            for (int i = m - 1; i >= 0; i--) {
                while (!stk.empty() && left[stk.top()][j] >= left[i][j]) {
                    stk.pop();
                }
                down[i] = stk.empty() ? m : stk.top();
                stk.push(i);
            }

            for (int i = 0; i < m; i++) {
                int height = (down[i] - 1)  - (up[i] + 1) + 1;
                int area = height * left[i][j];
                ret = max(ret, area);
            }
        }
        return ret;
    }
};
```



# 无串线性问题

#### [650. 只有两个键的键盘](https://leetcode-cn.com/problems/2-keys-keyboard/)

> 最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：
>
> Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。
>
> Paste（粘贴）：粘贴 上一次 复制的字符。
>
> 给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数。



如操作 `CPPCPPPPCP` 可以分为 `[CPP][CPPPP][CP]` 三组。

假设每组的长度为 g_1, g_2, ...。完成第一组操作后，字符串有 g_1 个 A，完成第二组操作后字符串有 g_1 * g_2 个 A。当完成所有操作时，共有 `g_1 * g_2 * ... * g_n` 个 `'A'`。

可以证明素数分解时最好.

```cpp
class Solution {
public:
    int minSteps(int n) {

        vector <int> f(n+10, 0);

        for (int i = 1;  i <= n; i++) f[i] = i;
        f[1] = 0, f[2] = 2;


        for (int i = 3; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) 
                if (i%j==0)
                    f[i] = min(f[j]+f[i/j], f[i]);
        }

        return f[n];
    }
};
```





# 例题

#### [5828. K 次调整数组大小浪费的最小总空间](https://leetcode-cn.com/problems/minimum-total-space-wasted-with-k-resizing-operations/)

> 给你一个下标从 0 开始的整数数组 nums ，其中 nums[i] 是 i 时刻数组中的元素数目。除此以外，你还有一个整数 k ，表示你可以 调整 数组大小的 最多 次数（每次都可以调整成 任意 大小）。
>
> t 时刻数组的大小 sizet 必须大于等于 nums[t] ，因为数组需要有足够的空间容纳所有元素。t 时刻 浪费的空间 为 sizet - nums[t] ，总 浪费空间为满足 0 <= t < nums.length 的每一个时刻 t 浪费的空间 之和 。
>
> 在调整数组大小不超过 k 次的前提下，请你返回 最小总浪费空间 。
>
> 数组最开始时可以为 **任意大小** ，且 **不计入** 调整大小的操作次数。
>



将数组nums[]分成k+1段, 新数组的每一段为 这一段的最大值 × 区间长度.

`dp[i][j]` [0....i] 个数, 分成j 段的最大和.

`maxa[i][j]` :   `i` 到 `j` 之间的最大值. 

```cpp
class Solution {
public:
    int minSpaceWastedKResizing(vector<int>& nums, int k) {
        k++;
        int n = nums.size();
        int INF = 1e9, ans = INF;
        vector < vector <int> > f( n+2, vector <int> (n+2, INF));
        vector < vector <int> > maxa( n+2, vector <int> (n+2));

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                maxa[i][j] = (i == j) ? nums[i] : max(maxa[i][j-1], nums[j]);
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= k; j++) {
                for (int i1 = -1; i1 < i; i1++)
                    f[i][j] = min( f[i][j], getf(i1, j-1, f) + maxa[i1+1][i] * (i-i1) );
                if (i == n-1) 
                    ans = min(ans, f[n-1][j]);
            }
        }
        for (auto x : nums) ans -= x;
        return ans;
        
    }

    int getf(int i, int j, vector < vector <int> > & f) {
        if (i<0) return 0;
        return f[i][j];
    }
};
```

