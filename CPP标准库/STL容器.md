# STL容器

分为四类, 序列式容器, 关联式容器, 无序关联式容器, 容器适配器.



# 迭代器

```cpp
vector<int> data(10);

for (int i = 0; i < data.size(); i++)
  cout << data[i] << endl;  // 使用下标访问元素

for (vector<int>::iterator iter = data.begin(); iter != data.end(); iter++)
  cout << *iter << endl;  // 使用迭代器访问元素
// 在C++11后可以使用 auto iter = data.begin() 来简化上述代码
```



# 序列式容器

## 简介

- **向量**(`vector`) 后端可高效增加元素的顺序表。
- **数组**(`array`)**C++11**，定长的顺序表，C 风格数组的简单包装。
- **双端队列**(`deque`) 双端都可高效增加元素的顺序表。
- **列表**(`list`) 可以沿双向遍历的链表。
- **单向列表**(`forward_list`) 只能沿一个方向遍历的链表。

## vector

`std::vector` 是 STL 提供的 **内存连续的**、**可变长度** 的数组（亦称列表）数据结构。能够提供线性复杂度的插入和删除，以及常数复杂度的随机访问。

### 构造函数

```cpp
// 1. 创建空vector; 常数复杂度
vector<int> v0;
// 1+. 这句代码可以使得向vector中插入前3个元素时，保证常数时间复杂度
v0.reserve(3);
// 2. 创建一个初始空间为3的vector，其元素的默认值是0; 线性复杂度
vector<int> v1(3);
// 3. 创建一个初始空间为3的vector，其元素的默认值是2; 线性复杂度
vector<int> v2(3, 2);
// 4. 创建一个初始空间为3的vector，其元素的默认值是1，
// 并且使用v2的空间配置器; 线性复杂度
vector<int> v3(3, 1, v2.get_allocator());
// 5. 创建一个v2的拷贝vector v4， 其内容元素和v2一样; 线性复杂度
vector<int> v4(v2);
// 6. 创建一个v4的拷贝vector v5，其内容是{v4[1], v4[2]}; 线性复杂度
vector<int> v5(v4.begin() + 1, v4.begin() + 3);
// 7. 移动v2到新创建的vector v6，不发生拷贝; 常数复杂度; 需要 C++11
vector<int> v6(std::move(v2));  // 或者 v6 = std::move(v2);
```



### 元素访问

```cpp
at()

v.at(pos) 返回容器中下标为 pos 的引用。如果数组越界抛出 std::out_of_range 类型的异常。

operator[]

v[pos] 返回容器中下标为 pos 的引用。不执行越界检查。

front()

v.front() 返回首元素的引用。

back()

v.back() 返回末尾元素的引用。

data()

v.data() 返回指向数组第一个元素的指针。
```



### 迭代器

```cpp
begin()/cbegin()

返回指向首元素的迭代器，其中 *begin = front。

end()/cend()

返回指向数组尾端占位符的迭代器，注意是没有元素的。

rbegin()/rcbegin()

返回指向逆向数组的首元素的逆向迭代器，可以理解为正向容器的末元素。

rend()/rcend()

返回指向逆向数组末元素后一位置的迭代器，对应容器首的前一个位置，没有元素。

以上列出的迭代器中，含有字符 c 的为只读迭代器，
```



### 长度和容量

```cpp
// 长度
empty() 返回一个 bool 值，即 v.begin() == v.end()，true 为空，false 为非空。

size() 返回容器长度（元素数量），即 std::distance(v.begin(), v.end())。

resize() 改变 vector 的长度，多退少补。补充元素可以由参数指定。

max_size() 返回容器的最大可能长度。

// 容量
reserve() 使得 vector 预留一定的内存空间，避免不必要的内存拷贝。

capacity() 返回容器的容量，即不发生拷贝的情况下容器的长度上限。

shrink_to_fit() 使得 vector 的容量与长度一致，多退但不会少。

```



### 元素增删及修改

```cpp
clear() 清除所有元素
insert() 支持在某个迭代器位置插入元素、可以插入多个。复杂度与 pos 距离末尾长度成线性而非常数的
erase() 删除某个迭代器或者区间的元素，返回最后被删除的迭代器。复杂度与 insert 一致。
push_back() 在末尾插入一个元素，均摊复杂度为 常数，最坏为线性复杂度。
pop_back() 删除末尾元素，常数复杂度。
swap() 与另一个容器进行交换，此操作是 常数复杂度 而非线性的。
```



## deque

双端队列.

### 构造函数

```cpp
// 1. 定义一个int类型的空双端队列 v0
deque<int> v0;
// 2. 定义一个int类型的双端队列 v1，并设置初始大小为10; 线性复杂度
deque<int> v1(10);
// 3. 定义一个int类型的双端队列 v2，并初始化为10个1; 线性复杂度
deque<int> v2(10, 1);
// 4. 复制已有的双端队列 v1; 线性复杂度
deque<int> v3(v1);
// 5. 创建一个v2的拷贝deque v4，其内容是v4[0]至v4[2]; 线性复杂度
deque<int> v4(v2.begin(), v2.begin() + 3);
// 6. 移动v2到新创建的deque v5，不发生拷贝; 常数复杂度; 需要 C++11
deque<int> v5(std::move(v2));
```



### 元素访问

```cpp
at() 返回容器中指定位置元素的引用，执行越界检查，常数复杂度。
operator[] 返回容器中指定位置元素的引用。不执行越界检查，常数复杂度。
front() 返回首元素的引用。
back() 返回末尾元素的引用。
```



### 迭代器

与 `vector` 一致。

### 长度

与 `vector` 一致，但是没有 `reserve()` 和 `capacity()` 函数。



### 元素增删及修改

与 `vector` 一致，并额外有向队列头部增加元素的函数。

```cpp
clear() 清除所有元素
insert() 支持在某个迭代器位置插入元素、可以插入多个。复杂度与 pos 与两端距离较小者成线性。
erase() 删除某个迭代器或者区间的元素，返回最后被删除的迭代器。复杂度与 insert 一致。
push_front() 在头部插入一个元素，常数复杂度。
pop_front() 删除头部元素，常数复杂度。
push_back() 在末尾插入一个元素，常数复杂度。
pop_back() 删除末尾元素，常数复杂度。
swap() 与另一个容器进行交换，此操作是 常数复杂度 而非线性的。
```



# 关联式容器

## 简介

- **集合**(`set`) 用以有序地存储 **互异** 元素的容器。其实现是由节点组成的红黑树，每个节点都包含着一个元素，节点之间以某种比较元素大小的谓词进行排列。
- **多重集合**(`multiset`) 用以有序地存储元素的容器。允许存在相等的元素。
- **映射**(`map`) 由 {键，值} 对组成的集合，以某种比较键大小关系的谓词进行排列。
- **多重映射**(`multimap`) 由 {键，值} 对组成的多重集合，亦即允许键有相等情况的映射。

## set

`set` 是关联容器，含有键值类型对象的已排序集，搜索、移除和插入拥有对数复杂度。`set` 内部通常采用红黑树实现。平衡二叉树的特性使得 `set` 非常适合处理需要同时兼顾查找、插入与删除的情况。

和数学中的集合相似，`set` 中不会出现值相同的元素。如果需要有相同元素的集合，需要使用 `multiset`。`multiset` 的使用方法与 `set` 的使用方法基本相同。

### 插入与删除

```cpp
insert(x) 当容器中没有等价元素的时候，将元素 x 插入到 set 中。
erase(x) 删除值为 x 的 所有 元素，返回删除元素的个数。
erase(pos) 删除迭代器为 pos 的元素，要求迭代器必须合法。
erase(first,last) 删除迭代器在 [first, last) 范围内的所有元素。
clear() 清空 set。
```



### 迭代器

```cpp
begin()/cbegin()
返回指向首元素的迭代器，其中 *begin = front。
end()/cend()
返回指向数组尾端占位符的迭代器，注意是没有元素的。
rbegin()/rcbegin()
返回指向逆向数组的首元素的逆向迭代器，可以理解为正向容器的末元素。
rend()/rcend()
返回指向逆向数组末元素后一位置的迭代器，对应容器首的前一个位置，没有元素。
```





### 查找

```cpp
count(x) 返回 set 内键为 x 的元素数量。
find(x) 在 set 内存在键为 x 的元素时会返回该元素的迭代器，否则返回 end()。
lower_bound(x) 返回指向首个不小于给定键的元素的迭代器。如果不存在这样的元素，返回 end()。
upper_bound(x) 返回指向首个大于给定键的元素的迭代器。如果不存在这样的元素，返回 end()。
empty() 返回容器是否为空。
size() 返回容器内元素个数。
```

注:   `set` 自带的 `lower_bound` 和 `upper_bound` 的时间复杂度为 O(logN) 。但使用 `algorithm` 库中的 `lower_bound` 和 `upper_bound` 函数对 `set` 中的元素进行查询，时间复杂度为 O(N)。

注:  `set` 没有提供自带的 `nth_element`。使用 `algorithm` 库中的 `nth_element` 查找第 大的元素时间复杂度为 O(N) 。



## map

`map` 是有序键值对容器，它的元素的键是唯一的。搜索、移除和插入操作拥有对数复杂度。`map` 通常实现为红黑树。

`map` 中不会存在键相同的元素，`multimap` 中允许多个元素拥有同一键。`multimap` 的使用方法与 `map` 的使用方法基本相同。



### 插入与删除

```cpp
可以直接通过下标访问来进行查询或插入操作。例如 mp["Alan"]=100。
通过向 map 中插入一个类型为 pair<Key, T> 的值可以达到插入元素的目的，例如 mp.insert(pair<string,int>("Alan",100));；
erase(key) 函数会删除键为 key 的 所有 元素。返回值为删除元素的数量。
erase(pos): 删除迭代器为 pos 的元素，要求迭代器必须合法。
erase(first,last): 删除迭代器在  范围内的所有元素。
clear() 函数会清空整个容器。
```

注 :  当下标访问操作过于频繁时，容器中会出现大量无意义元素，影响 `map` 的效率。因此一般情况下推荐使用 `find()` 函数来寻找特定键的元素。



### 查询

```cpp
count(x): 返回容器内键为 x 的元素数量。复杂度为 （关于容器大小对数复杂度，加上匹配个数）。
find(x): 若容器内存在键为 x 的元素，会返回该元素的迭代器；否则返回 end()。
lower_bound(x): 返回指向首个不小于给定键的元素的迭代器。
upper_bound(x): 返回指向首个大于给定键的元素的迭代器。若容器内所有元素均小于或等于给定键，返回 end()。
empty(): 返回容器是否为空。
size(): 返回容器内元素个数。
```



# 无序关联式容器

- **无序（多重）集合**(`unordered_set`/`unordered_multiset`)**C++11**，与 `set`/`multiset` 的区别在与元素无序，只关心”元素是否存在“，使用哈希实现。
- **无序（多重）映射**(`unordered_map`/`unordered_multimap`)**C++11**，与 `map`/`multimap` 的区别在与键 (key) 无序，只关心 "键与值的对应关系"，使用哈希实现。

这几种无序关联式容器则采用哈希方式存储元素，内部元素不以任何特定顺序进行排序，所以访问无序关联式容器中的元素时，访问顺序也没有任何保证。



## 制造哈希冲突

在标准库实现里，每个元素的散列值是将值对一个质数取模得到的，更具体地说，是 [这个列表](https://github.com/gcc-mirror/gcc/blob/gcc-8_1_0-release/libstdc++-v3/src/shared/hashtable-aux.cc) 中的质数（g++ 6 及以前版本的编译器，这个质数一般是 $126271$，g++ 7 及之后版本的编译器，这个质数一般是 $107897$）。

因此可以通过向容器中插入这些模数的倍数来达到制造大量哈希冲突的目的。



## 自定义哈希函数

```cpp
struct my_hash {
  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }

  // 针对 std::pair<int, int> 作为主键类型的哈希函数
  size_t operator()(pair<uint64_t, uint64_t> x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x.first + FIXED_RANDOM) ^
           (splitmix64(x.second + FIXED_RANDOM) >> 1);
  }
};

unordered_map<int, int, my_hash> my_map; // 或者 unordered_map<pair<int, int>, int, my_hash> my_pair_map;
```



# 容器适配器

容器适配器其实并不是容器。它们不具有容器的某些特点（如：有迭代器、有 `clear()` 函数……）。

- **栈** `(stack`) 后进先出 (LIFO) 的容器。
- **队列**(`queue`) 先进先出 (FIFO) 的容器。
- **优先队列**(`priority_queue`) 元素的次序是由作用于所存储的值对上的某种谓词决定的的一种队列。

参考"数据结构"专题.



# bitset

`std::bitset` 是标准库中的一个存储 `0/1` 的大小不可变容器。







# pair

`std::pair` 是标准库中定义的一个类模板。用于将两个变量关联在一起，组成一个“对”，而且两个变量的数据类型可以是不同的。



### 初始化

```cpp
pair<int, double> p0(1, 2.0);
// 可以在定义时直接完成 pair 的初始化。

pair<int, double> p1;
p1.first = 1;
p1.second = 2.0;
// 使用先定义，后赋值的方法

pair<int, double> p2 = make_pair(1, 2.0);
// 使用 std::make_pair 函数


auto p3 = make_pair(1, 2.0);
// 使用auto

```



### 访问

```cpp
int i = p0.first;
double d = p0.second;

p1.first++;
```



### 比较

`pair` 已经预先定义了所有的比较运算符，包括 `<`、`>`、`<=`、`>=`、`==`、`!=`。当然，这需要组成 `pair` 的两个变量所属的数据类型定义了 `==` 和/或 `<` 运算符。

其中，`<`、`>`、`<=`、`>=` 四个运算符会先比较两个 `pair` 中的第一个变量，在第一个变量相等的情况下再比较第二个变量。



### 赋值与交换

```cpp
p0 = p1;

swap(p0, p1);
p2.swap(p3);
```

