# DFS



# 栈

dfs和栈都是"递归结构", "后进先出",所以dfs可以有两种写法.

递归式

```cpp
void dfs(int u) {
    vis[u] = 1;
    for (auto v : g[u]) {
        if (!vis[v]) {
			dfs(v);
        }
    }
}
```



非递归式

```cpp
void dfs(int start) {
    stack <int> S;
    S.push(start), vis[start] = 1;
    while (!s.empty()) {
        int u = s.top();
        bool ff = 1;
        for (auto v : g[u]) {
            if (!vis[v]) {
                vis[v] = 1;
                s.push(v);
                ff = 0;
                break;
            }
        }
        if (ff) s.pop();
    }
}
```



# DFS应用



## 获得图(树)的一些属性





## 计算无向图的连通分量





## 检测图中是否存在环





## 二分图检测





## 拓扑排序





## 回溯算法获得一个问题所有的解





