# 马拉车Manacher算法

给定一个长度为 $n$ 的字符串 $s$，请找到所有对 $(i, j)$ 使得子串 $s[i \dots j]$ 为一个回文串。当 $t = t_{\text{rev}}$ 时，字符串 $t$ 是一个回文串（$t_{\text{rev}}$ 是 $t$ 的反转字符串）。





给定一个长度为 $n$ 的字符串 $s$，我们在其 $n + 1$ 个空中插入分隔符 $\#$，从而构造一个长度为 $2n + 1$ 的字符串 $s'$。举例来说，对于字符串 $s = \mathtt{abababc}$，其对应的 $s' = \mathtt{\#a\#b\#a\#b\#a\#b\#c\#}$。

这样, 所有的回文串就都是奇数回文串.



d1[i] : 以位置 i 为中心的最长回文串的半径长度

```cpp
vector<int> d1(n);
for (int i = 0, l = 0, r = -1; i < n; i++) {
  int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
  while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) {
    k++;
  }
  d1[i] = k--;
  if (i + k > r) {
    l = i - k;
    r = i + k;
  }
}
```