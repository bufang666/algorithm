# 字典树Trie

字典树，英文名 trie。顾名思义，就是一个像字典一样的树。



# 实现

```cpp
struct Trie {
	int nex[100000][26], cnt;
	bool exist[100000];  // 该结点结尾的字符串是否存在

	void insert(char *s, int l) {  // 插入字符串
		int p = 0;
		for (int i = 0; i < l; i++) {
			int c = s[i] - 'a';
			if (!nex[p][c]) nex[p][c] = ++cnt;  // 如果没有，就添加结点
			p = nex[p][c];
		}
		exist[p] = 1;
	}
	bool find(char *s, int l) {  // 查找字符串
		int p = 0;
		for (int i = 0; i < l; i++) {
			int c = s[i] - 'a';
			if (!nex[p][c]) return 0;
			p = nex[p][c];
		}
		return exist[p];
	}
} t;
```



# 应用



## 检索字符串

Q : [于是他错误的点名开始了](https://www.luogu.com.cn/problem/P2580)

​	给你 $n$ 个名字串，然后进行 $m$ 次点名，每次你需要回答“名字不存在”、“第一次点到这个名字”、“已经点过这个名字”之一。

A:

对所有名字建 trie，再在 trie 中查询字符串是否存在、是否已经点过名，第一次点名时标记为点过名。

```cpp
#include <bits/stdc++.h>

using namespace std;

const int maxn = 5e5 + 100;

struct Trie {
	int nex[maxn][26], cnt = 0;
	bool exist[maxn];  // 该结点结尾的字符串是否存在
	bool v[maxn];
	
	Trie() {
		cnt = 0;
	}

	void insert1(string s, int l) {  // 插入字符串
		int p = 0;
		for (int i = 0; i < l; i++) {
			int c = s[i] - 'a';
			if (!nex[p][c]) nex[p][c] = ++cnt;  // 如果没有，就添加结点
			p = nex[p][c];
		}
		exist[p] = 1, v[p] = 0;
		return ;
	}
	int find(string s, int l) {  // 查找字符串
		int p = 0;
		for (int i = 0; i < l; i++) {
			int c = s[i] - 'a';
			if (!nex[p][c]) return 2;
			p = nex[p][c];
		}
		
		if (exist[p] && !v[p]) {
			v[p] = 1;
			return 1;
		} else if (exist[p] && v[p]) 
			return 0;
		else 
			return 2;
	}
} t;

int main() {
	int n, m;
	string s;
	
	cin >> n;
	for (int i = 0; i < n; i++) 
		cin >> s, t.insert1(s, (int)s.length());
	cin >> m;
	for (int i = 0; i < m; i++) {
		cin >> s;
		int x = t.find(s, (int)s.length());
		if (x == 1) cout << "OK" << endl;
		else if (x == 2) cout << "WRONG" << endl;
		else cout << "REPEAT" << endl;
	}
	return 0;
}
```



## AC自动机

trie 是 AC自动机的一部分



## 异或问题





