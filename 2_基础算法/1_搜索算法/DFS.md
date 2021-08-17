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





# 例题



#### [5827. 检查操作是否合法](https://leetcode-cn.com/problems/check-if-move-is-legal/)

给你一个下标从 0 开始的 8 x 8 网格 board ，其中 board[r][c] 表示游戏棋盘上的格子 (r, c) 。棋盘上空格用 '.' 表示，白色格子用 'W' 表示，黑色格子用 'B' 表示。

游戏中每次操作步骤为：选择一个空格子，将它变成你正在执行的颜色（要么白色，要么黑色）。但是，合法 操作必须满足：涂色后这个格子是 好线段的一个端点 （好线段可以是水平的，竖直的或者是对角线）。

好线段 指的是一个包含 三个或者更多格子（包含端点格子）的线段，线段两个端点格子为 同一种颜色 ，且中间剩余格子的颜色都为 另一种颜色 （线段上不能有任何空格子）。

A : 

八个方向判断. 斟酌细节.

```cpp
class Solution {
public:
    int dx[10] = {0, 1, 1, 1, 0, -1, -1, -1};
    int dy[10] = {1, 1, 0, -1, -1, -1, 0, 1};

    bool ok(vector<vector<char>>& board, int rMove, int cMove, int dir, char color) {
        int r = rMove + dx[dir], c = cMove + dy[dir];
        int cnt = 0;
        while (r>=0 && r<8 && c>=0 && c<8) {
            cnt++;
            if (board[r][c] == color) return cnt >= 2;
            else if (board[r][c] == '.') return 0;
            r += dx[dir], c += dy[dir];
        }
        return 0;
    }

    bool checkMove(vector<vector<char>>& board, int rMove, int cMove, char color) {
        if (board[rMove][cMove] != '.') return 0;
        for (int i = 0; i < 8; i++) {
            if (ok(board, rMove, cMove, i, color)) 
                return 1;
        }
        return 0;
    }
};
```



#### [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

时间复杂度 O(3^K^MN)

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
private:
    int rows, cols;
    bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
        if(i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';
        bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || 
                      dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }
};
```

