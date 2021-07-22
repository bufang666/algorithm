# BFS



# 队列

BFS是"齐头并进", 层层扩张, 先进先出, 所以用队列.

```cpp
queue <int> q;
vis[st] = 1, q.push(st);
while (!q.empty()) {
    int u = q.pop_front();
    for (auto v : g[u]) {
        if (!vis[v]) {
            vis[v] = 1, q.push(v);
        }
    }
}
```



# 应用

## 拓扑排序

## 搜索问题

