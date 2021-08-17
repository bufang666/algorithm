# 树形DP

树形 DP，即在树上进行的 DP。由于树固有的递归性质，树形 DP 一般都是递归进行的。



# 例题

#### [洛谷 P1352 没有上司的舞会](https://www.luogu.com.cn/problem/P1352)

某大学有 $n$ 个职员，编号为 $1 \sim N$。他们之间有从属关系，也就是说他们的关系就像一棵以校长为根的树，父结点就是子结点的直接上司。现在有个周年庆宴会，宴会每邀请来一个职员都会增加一定的快乐指数 $a_i$，但是呢，如果某个职员的上司来参加舞会了，那么这个职员就无论如何也不肯来参加舞会了。所以，请你编程计算，邀请哪些职员可以使快乐指数最大，求最大的快乐指数。

A : 

定义 $f(i,0/1)$ 代表以 $i$ 为根的子树的最优解（第二维的值为 0 代表 $i$ 不参加舞会的情况，1 代表 $i$ 参加舞会的情况）。

显然，我们可以推出下面两个状态转移方程（其中下面的 $x$ 都是 $i$ 的儿子）：

- $f(i,0) = \sum\max \{f(x,1),f(x,0)\}$（上司不参加舞会时，下属可以参加，也可以不参加）
- $f(i,1) = \sum{f(x,0)} + a_i$（上司参加舞会时，下属都不会参加）

我们可以通过 DFS，在返回上一层时更新当前结点的最优解。



```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#define N 6005
using namespace std;
int n;
bool visited[N];
vector<int> T[N];
int dp[N][2]; // 从结点i往下所获最大value
// dp[i][0]表示i不参加时子树的maxValue，dp[i][1]表示i参加时子树的maxValue


void dfs(int k) { // 给定结点k，求以k为root的最大收益
    if (visited[k]) return ;
    visited[k] = true;
    for (int i = 0; i < T[k].size(); ++i) {
        int v = T[k][i];
        dfs(v);
        dp[k][0] += max(dp[v][0],dp[v][1]);
        dp[k][1] += dp[v][0];
    }
}


int main() {
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> dp[i][1];
    }
    bool tmp[N] = {0};
    for (int i = 1; i <= n-1; ++i) {
        int l,k;
        cin >> l >> k;
        T[k].push_back(l);
        tmp[l] = 1; // 是孩子结点，标记为1
    }
    for (int i = 1; i <= n; ++i) {
        if (tmp[i] == 0) { // 寻找唯一的根节点，找到之后dfs
            dfs(i);
            cout << max(dp[i][0],dp[i][1]) << "\n";
            break;
        }
    }
    return 0;
}
```



# 树上背包

#### [洛谷 P2014 CTSC1997 选课](https://www.luogu.com.cn/problem/P2014)

现在有 $n$ 门课程，第 $i$ 门课程的学分为 $a_i$，每门课程有零门或一门先修课，有先修课的课程需要先学完其先修课，才能学习该课程。

 一位学生要学习 $m$​ 门课程，求其能获得的最多学分数。  $n,m \leq 300$



A : 

根据 " 每门课最多只有一门先修课 " 建树, 从而所有课程组成了一个森林的结构。

可以新增一门 $0$ 学分的课程（设这个课程的编号为 $0$），作为所有无先修课课程的先修课，这样我们就将森林变成了一棵以 $0$ 号课程为根的树。

设 $f(u,i,j)$ 表示以 $u$ 号点为根的子树中，已经遍历了 $u$ 号点的前 $i$ 棵子树，选了 $j$ 门课程的最大学分。\

转移的过程结合了树形 DP 和背包 DP 的特点，我们枚举 $u$ 点的每个子结点 $v$，同时枚举以 $v$ 为根的子树选了几门课程，将子树的结果合并到 $u$ 上。

记点 $x$​ 的儿子个数为 $s_x$​，以 $x$​ 为根的子树大小为 $\textit{sizex}$​，很容易写出下面的转移方程：

$$
f(u,i,j)=\max_{v,k \leq j,k \leq \textit{sizex}} f(u,i-1,j-k)+f(v,s_v,k)
$$

注意上面转移方程中的几个限制条件，这些限制条件确保了一些无意义的状态不会被访问到。

$f$ 的第二维可以很轻松地用滚动数组的方式省略掉，注意这时需要倒序枚举 $j$ 的值。

时间复杂度为 $O(nm)$ 

```cpp
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace std;
int f[305][305], s[305], n, m;
vector<int> e[305];
int dfs(int u) {
  int p = 1;
  f[u][1] = s[u];
  for (auto v : e[u]) {
    int siz = dfs(v);
    // 注意下面两重循环的上界和下界
    // 只考虑已经合并过的子树，以及选的课程数超过 m+1 的状态没有意义
    // f[u][i+j] 依赖的是 f[u][i], 所以i从大到小枚举可以起到省略第二维的作用
    for (int i = min(p, m + 1); i; i--)
      for (int j = 1; j <= siz && i + j <= m + 1; j++)
        f[u][i + j] = max(f[u][i + j], f[u][i] + f[v][j]);
    p += siz;
  }
  return p;
}
int main() {
  scanf("%d%d", &n, &m);
  for (int i = 1; i <= n; i++) {
    int k;
    scanf("%d%d", &k, &s[i]);
    e[k].push_back(i);
  }
  dfs(0);
  printf("%d", f[0][m + 1]);
  return 0;
}
```



# 换根DP

树形 DP 中的换根 DP 问题又被称为二次扫描.

通常需要两次 DFS，第一次 DFS 预处理诸如深度，点权和之类的信息，在第二次 DFS 开始运行换根动态规划。



####  [[POI2008\]STA-Station](https://www.luogu.com.cn/problem/P3478)

给定一个 $n$ 个点的树，请求出一个结点，使得以这个结点为根时，所有结点的深度之和最大。



A : 

先用第一次 DFS 预处理，我们得到了以某个结点为根时其子树中的结点总数。

考虑状态转移，这里就是体现＂换根＂的地方了。令 $f_u$ 为以 $u$ 为根时，所有结点的深度之和。

$f_v\leftarrow f_u$ 可以体现换根，即以 $u$ 为根转移到以 $v$ 为根。显然在换根的转移过程中，以 $v$ 为根或以 $u$ 为根会导致其子树中的结点的深度产生改变。具体表现为：

- 所有在 $v$ 的子树上的结点深度都减少了一，那么总深度和就减少了 $s_v$；

- 所有不在 $v$ 的子树上的结点深度都增加了一，那么总深度和就增加了 $n-s_v$；

根据这两个条件就可以推出状态转移方程 $f_v = f_u - s_v + n - s_v=f_u + n - 2 \times s_v$。

于是在第二次 DFS 遍历整棵树并状态转移 $f_v=f_u + n - 2 \times s_v$，那么就能求出以每个结点为根时的深度和了。最后只需要遍历一次所有根结点深度和就可以求出答案。

```cpp
#include <bits/stdc++.h>

using namespace std;

int head[1000010 << 1], tot;
long long n, size[1000010], dep[1000010];
long long f[1000010];

struct node {
	int to, next;
} e[1000010 << 1];

void add(int u, int v) {
	e[++tot] = node {v, head[u]};
	head[u] = tot;
}

void dfs(int u, int fa) {
	size[u] = 1;
	dep[u] = dep[fa] + 1;
	for (int i = head[u]; i; i = e[i].next) {
		int v = e[i].to;
		if (v != fa) {
			dfs(v, u);
			size[u] += size[v];
		}
	}
}

void get_ans(int u, int fa) {
	for (int i = head[u]; i; i = e[i].next) {
		int v = e[i].to;
		if (v != fa) {
			f[v] = f[u] - size[v] * 2 + n;
			get_ans(v, u);
		}
	}
}

int main() {
	scanf("%lld", &n);
	int u, v;
	for (int i = 1; i <= n - 1; i++) {
		scanf("%d %d", &u, &v);
		add(u, v);
		add(v, u);
	}
	dfs(1, 1);
	for (int i = 1; i <= n; i++) f[1] += dep[i];
	get_ans(1, 1);
	long long int ans = -1;
	int id;
	for (int i = 1; i <= n; i++) {
		if (f[i] > ans) {
			ans = f[i];
			id = i;
		}
	}
	printf("%d\n", id);
	return 0;
}
```