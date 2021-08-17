# 马拉车Manacher算法

给定一个长度为 $n$ 的字符串 $s$，请找到所有对 $(i, j)$ 使得子串 $s[i \dots j]$ 为一个回文串。当 $t = t_{\text{rev}}$ 时，字符串 $t$ 是一个回文串（$t_{\text{rev}}$ 是 $t$ 的反转字符串）。





给定一个长度为 $n$ 的字符串 $s$，我们在其 $n + 1$ 个空中插入分隔符 $\#$，从而构造一个长度为 $2n + 1$ 的字符串 $s'$。举例来说，对于字符串 $s = \mathtt{abababc}$，其对应的 $s' = \mathtt{\#a\#b\#a\#b\#a\#b\#c\#}$。

这样, 所有的回文串就都是奇数回文串.



`d[i]` : 以位置 i 为中心的最长回文串的半径长度

`r` : 回文串的右边界的最大值, `l` 为 `r` 对应的那个回文串的左边界

`ans` : 最长回文子串的长度.

```cpp
string changes(string s)
{
    string t;
    t += '#';
    for (auto ch : s) {
        t += ch, t += '#';
    }
    return t;
}

int manacher(string s) {
	int ans = 1, n = s.length();
    vector<int> d(n + 2);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
      int k = (i > r) ? 1 : min(d[l + r - i], r - i + 1);
      while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
        k++;
      }
      d[i] = k;
      if (i + k - 1 > r) {
        l = i - (k - 1);
        r = i + (k - 1);
      }
      ans = max(ans, d[i] - 1);
    }
    return ans;
}
```



# 例题

#### [5220. 两个回文子字符串长度的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/)

给你一个下标从 **0** 开始的字符串 `s` ，你需要找到两个 **不重叠**的回文子字符串，它们的长度都必须为 **奇数** ，使得它们长度的乘积最大。

A :

要的就是奇数回文串, 直接用马拉车算法, 不用加 `#` 

d1[i] : 以位置 i 为中心的最长回文串的半径长度

先得到d1[i], 然后得到 `f[i]` : 以 i 结尾的奇数回文串的最大长度, 然后得到 `f[i]` : 奇数回文串的右边界在 [0...i] 的最大长度. 

类比的得到 `g[i]` :  奇数回文串的左边界在 [i...n-1] 的最大长度. 

```cpp
class Solution {
public:
    long long maxProduct(string s) {
        int n = s.length();
        
        vector <int> f(n+2, 1), g(n+2, 1);
        
        //int ans = 1;
        vector<int> d(n + 2);
        for (int i = 0, l = 0, r = -1; i < n; i++) {
         	int k = (i > r) ? 1 : min(d[l + r - i], r - i + 1);
          	while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
                k++;
            }
            d[i] = k;
            if (i + k - 1 > r) {
                l = i - (k - 1);
                r = i + (k - 1);
            }
           // ans = max(ans, d[i] - 1);
        }
        //return ans;
        
        for (int i = 0; i < n; i++) f[i+d[i]-1] = max(2*d[i]-1, f[i+d[i]-1]);
        for (int i = n-2; i >= 0; i--) f[i] = max(f[i+1]-2, f[i]);
        
        for (int i = 0; i < n; i++) g[i-d[i]+1] = max(2*d[i]-1, g[i-d[i]+1]);
        for (int i = 1; i < n; i++) g[i] = max(g[i-1]-2, g[i]);
        
        for (int i = 1; i < n; i++) f[i] = max(f[i], f[i-1]);
        for (int i = n-2; i >= 0; i--) g[i] = max(g[i], g[i+1]);
        
        long long ans = 1;
        
        for (int i = 0; i < n-1; i++) ans = max(ans, (long long)f[i] * g[i+1]);
         
        return ans;
    }
};
```

