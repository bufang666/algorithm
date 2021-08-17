# STL算法



# find() 与 find_end()

- `find`：顺序查找。`find(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。
- `find_end`：逆序查找。`find_end(v.begin(), v.end(), value)`。



# reverse()

- `reverse`：翻转数组、字符串。`reverse(v.begin(), v.end())` 或 `reverse(a + begin, a + end)`。



# unique()

- `unique`：去除容器中相邻的重复元素。`unique(ForwardIterator first, ForwardIterator last)`，返回值为指向 **去重后** 容器结尾的迭代器，原容器大小不变。与 `sort` 结合使用可以实现完整容器去重。



# sort()

- `sort`：排序。`sort(v.begin(), v.end(), cmp)` 或 `sort(a + begin, a + end, cmp)`，其中 `end` 是排序的数组最后一个元素的后一位，`cmp` 为自定义的比较函数。
- `stable_sort`：稳定排序，用法同 `sort()`。





# nth_element()

- `nth_element`：按指定范围进行分类，即找出序列中第 $n$ 大的元素，使其左边均为小于它的数，右边均为大于它的数。`nth_element(v.begin(), v.begin() + mid, v.end(), cmp)` 或 `nth_element(a + begin, a + begin + mid, a + end, cmp)`。





# binary_search()

- `binary_search`：二分查找。`binary_search(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。



# merge() 与 inplace_merge()

- `merge`：将两个（已排序的）序列 **有序合并** 到第三个序列的 **插入迭代器** 上。`merge(v1.begin(), v1.end(), v2.begin(), v2.end() ,back_inserter(v3))`。
- `inplace_merge`：将两个（已按小于运算符排序的）：`[first,middle), [middle,last)` 范围 **原地合并为一个有序序列**。`inplace_merge(v.begin(), v.begin() + middle, v.end())``\



# lower_bound() 与 upper_bound()

- `lower_bound`：在一个有序序列中进行二分查找，返回指向第一个 **大于等于**  $x$ 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`lower_bound(v.begin(),v.end(),x)`。
- `upper_bound`：在一个有序序列中进行二分查找，返回指向第一个 **大于**  $x$ 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`upper_bound(v.begin(),v.end(),x)`。

Q : 使用 `lower_bound` 与 `upper_bound` 查找有序数组 $a$ 中小于 $x$，等于 $x$，大于 $x$ 元素的分界线。

```cpp
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 7, 7, 9, 9}, x = 5;
int i = lower_bound(a, a + N, x) - a, j = upper_bound(a, a + N, x) - a;
// a[0] ~ a[i - 1] 为小于x的元素， a[i] ~ a[j - 1] 为等于x的元素， a[j] ~ a[N -
// 1] 为大于x的元素
cout << i << " " << j << endl;
```

Q : 使用 `lower_bound` 查找有序数组 $a$ 中最接近 $x$ 的元素。

```cpp
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 8, 8, 9, 9}, x = 6;
// lower_bound将返回a中第一个大于等于x的元素的地址，计算出的i为其下标
int i = lower_bound(a, a + N, x) - a;
// 在以下两种情况下，a[i] (a中第一个大于等于x的元素) 即为答案：
// 1. a中最小的元素都大于等于x；
// 2. a中存在大于等于x的元素，且第一个大于等于x的元素 (a[i])
// 相比于第一个小于x的元素 (a[i - 1]) 更接近x；
// 否则，a[i - 1] (a中第一个小于x的元素) 即为答案
if (i == 0 || (i < N && a[i] - x < x - a[i - 1]))
  cout << a[i];
else
  cout << a[i - 1];
```



# next_permutation



- `next_permutation`：将当前排列更改为 **全排列中的下一个排列**。如果当前排列已经是 **全排列中的最后一个排列**（元素完全从大到小排列），函数返回 `false` 并将排列更改为 **全排列中的第一个排列**（元素完全从小到大排列）；否则，函数返回 `true`。`next_permutation(v.begin(), v.end())` 或 `next_permutation(v + begin, v + end)`。

  Q : 使用 `next_permutation` 生成 $1$ 到 $9$ 的全排列。

  ```cpp
  int N = 9, a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  do {
    for (int i = 0; i < N; i++) cout << a[i] << " ";
    cout << endl;
  } while (next_permutation(a, a + N));
  ```





# partial_sum

- `partial_sum`：求前缀和。设源容器为 $x$，目标容器为 $y$，则令 $y[i]=x[0]+x[1]+...+x[i]$。`partial_sum(src.begin(), src.end(), back_inserter(dst))`。

  ```cpp
  vector<int> src = {1, 2, 3, 4, 5}, dst;
  // 求解src中元素的前缀和，dst[i] = src[0] + ... + src[i]
  // back_inserter 函数作用在 dst 容器上，提供一个迭代器
  partial_sum(src.begin(), src.end(), back_inserter(dst));
  for (unsigned int i = 0; i < dst.size(); i++) cout << dst[i] << " ";
  ```

  
