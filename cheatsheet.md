 Cheatsheet：大多数都是我自己做了题目后感觉会有用而准备的资料，主要注重前几题简单题（因为后面的应该做不出来，但是复习的时候也涵盖了这部分）也些有部分借鉴了群里大佬的资料

**基础概念：**

双指针：
```
    def twoSum(self, price: List[int], target: int) -> List[int]:
        l, r = 0, len(price) - 1
        while l < r:
            s = price[l] + price[r]
            if s == target:
                return [l, r]
            elif s < target:
                l += 1
            else:
                r -= 1
        return []
```

二分法：
```
def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if target > nums[mid]:
                l = mid + 1
            elif target < nums[mid]:
                r = mid - 1
            else:
                return mid
        return -1
```

分割字符串：
（常规）
```
s = input().split(";") 
s = s[:3]
l = []
for i in range(3):
    if i < 3 and "::" in s[i]:
        l.append(s[i].split("::")[1])
    else:
        l.append("0")
print(l)
```

（try-except方法）
```
s = input().split(";")
s = s[:3]
l = []
for i in range(3):
    try:
        l.append(s[i].split("::")[1])
    except:
        l.append("0")
print(l)
```

将整数分为k份（dfs+dp，lru_cache记忆化递归整数划分计数）：
```
from functools import lru_cache

n, k = map(int, input().split())

@lru_cache(maxsize=None)
def count_partitions(n, k, m):
    if n == 0 and k == 0:
        return 1
    if n <= 0 or k <= 0 or m <= 0:
        return 0
    # 选 m 和不选 m 两种情况递归
    return count_partitions(n - m, k - 1, m) + count_partitions(n, k, m - 1)

# 最大数字不能超过 n，初始调用 m 从 n 开始
print(count_partitions(n, k, n))
```

Febonacci：
 （公式）
```
def fraction(self, cont: List[int]) -> List[int]:
        n, m = 0, 1
        for a in cont[::-1]:
            n, m = m, (m * a + n)
        return [m, n]
```

 （dp方法）
```
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[0] = 0  # 初始化
    dp[1] = 1  # 初始化
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]  # 状态转移方程
    return dp[n]
```

 （从后开始的dp（数组元素替换））
```
def replaceElements(self, arr: List[int]) -> List[int]: 
    e = -1
    for i in reversed(range(len(arr))):
        x = arr[i]
        arr[i] = e
        e = max(x, e)
    return arr
```

比较经典的dp（？（最大利润）：
```
def maxProfit(self, prices: List[int]) -> int:
    n = len(prices)
    dp = [0] * n
    minp = prices[0]
    for i in range(1, n):
        minp = min(minp, prices[i])
        dp[i] = max(dp[i - 1], prices[i] - minp)
    return dp[-1]
```

合并区间：
```
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    ans = []
    intervals.sort(key=lambda x: x[0])
    for i in intervals:
        if not ans or ans[-1][1] < i[0]:
            ans.append(i)
        else:
            ans[-1][1] = max(ans[-1][1], i[1])
    return ans
```

矩阵乘加：
```
def mat():
    r, c = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(r)]
    return matrix, r, c

A, rA, cA = mat()
B, rB, cB = mat()
C, rC, cC = mat()

if cA != rB:
    print("Error!")
    exit()

ans = [[0 * cB for _ in range(rA)]]
if cA == rB:
    for i in range(rA):
        for j in range(cB):
            for k in range(cA):
                ans[i][j] += A[i][k] * B[k][j]

if len(ans) != rC or len(ans[0]) != cC:
    print("Error!")
    exit()

res = [[0 * cC for _ in range(rC)]]
if len(ans) == rC and len(ans[0]) == cC:
    for i in range(rC):
        for j in range(cC):
            res[i][j] = ans[i][j] + C[i][j]

for _ in res:
    print(" ".join(map(str, _)))
Stack：
while True:
    try:
        s = input()
        mark = [" "] * len(s)
        stack = []
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            elif ch == ")":
                if stack:
                    stack.pop()
                else:
                    mark[i] = "?"
        if stack:
            for j in stack:
                mark[j] = "$"
        print(s)
        print("".join(mark))
    except EOFError:
        break
```

单调栈：
```
arr = [2, 1, 2, 4, 3]
stack = []
res = [-1] * len(arr)

for i, v in enumerate(arr):
    while stack and arr[stack[-1]] < v:
        idx = stack.pop()
        res[idx] = v  # 找到右边第一个更大的元素
    stack.append(i)

print(res)  # 输出: [4, 2, 4, -1, -1]
```

合法出栈：
```
x = input().strip()

try:
    while True:
        target = input().strip()
        stack = []
        i = 0  # 入栈指针
        j = 0  # 目标序列指针

        while i < len(x):
            stack.append(x[i])
            i += 1

            # 尝试弹栈匹配目标序列
            while stack and j < len(target) and stack[-1] == target[j]:
                stack.pop()
                j += 1

        # 入栈完毕后，如果栈不空，还要继续尝试弹出
        while stack and j < len(target) and stack[-1] == target[j]:
            stack.pop()
            j += 1

        # 判断是否匹配
        if j == len(target) and not stack:
            print("YES")
        else:
            print("NO")

except EOFError:
    pass
```

中序转后序表达式（stack）：
```
def precedence(op):
    if op in ('*', '/'):
        return 2
    elif op in ('+', '-'):
        return 1
    else:
        return 0

def to_postfix(expression):
    import re
    tokens = re.findall(r'\d+\.?\d*|[()+\-*/]', expression)
    stack = []
    output = []

    for token in tokens:
        if re.match(r'\d+\.?\d*', token):
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and precedence(stack[-1]) >= precedence(token):
                output.append(stack.pop())
            stack.append(token)

    while stack:
        output.append(stack.pop())

    return ' '.join(output)

n = int(input())
for _ in range(n):
    expr = input().strip()
    print(to_postfix(expr))
```

实现堆结构（heap）：
```
import heapq

n = int(input())
heap = []

for _ in range(n):
    ops = input().split()
    t = int(ops[0])
    if t == 1:
        # 插入操作
        u = int(ops[1])
        heapq.heappush(heap, u)
    else:
        # 弹出最小元素
        if heap:
            print(heapq.heappop(heap))
```

FBI树（字符串+dfs）：
```
def get_type(s):
    if all(c == '0' for c in s):
        return 'B'
    elif all(c == '1' for c in s):
        return 'I'
    else:
        return 'F'

def build(s):
    if len(s) == 1:  # 字符串长度为1时，直接根据类型返回节点（或对应表示 ）
        return get_type(s)
    mid = len(s) // 2
    left = build(s[:mid])  # 递归构建左子树对应的字符串部分
    right = build(s[mid:])  # 递归构建右子树对应的字符串部分
    # 合并左右子树构建结果和当前节点类型
    return left + right + get_type(s)

N = int(input())
s = input().strip()
result = build(s)
print(result)
```

找最大值（滑动窗口）：
```
def maximumUniqueSubarray(nums):
    seen = set()  # 用于记录当前窗口内的元素，判断是否重复
    left = 0  # 滑动窗口的左边界
    current_sum = 0  # 当前窗口内元素的和
    max_sum = 0  # 记录最大的窗口和

    for right in range(len(nums)):  # 滑动窗口的右边界
        # 如果当前右边界元素在窗口中已存在，移动左边界直到元素不重复
        while nums[right] in seen:  
            seen.remove(nums[left])  
            current_sum -= nums[left]  
            left += 1  
        
        seen.add(nums[right])  # 将当前右边界元素加入窗口
        current_sum += nums[right]  # 更新当前窗口和
        max_sum = max(max_sum, current_sum)  # 更新最大和

    return max_sum
```

链表反转：
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        vals = []
        current_node = head
        while current_node is not None:
            vals.append(current_node.val)
            current_node = current_node.next
        return vals == vals[::-1]
```

二叉树（后序&层序）：
（递归）
```
def maxDepth(self, root: Optional[TreeNode]) -> int: 
    if not root:
        return 0
return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

（bfs）
```
def maxDepth(self, root: TreeNode) -> int: 
    if not root:
        return 0
    queue, res = [root], 0
    while queue:
        tmp = []
        for node in queue:
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        queue = tmp
        res += 1
    return res
```

二叉树直径（深度dfs）：
```
class SomeClass1:
    def __init__(self):
        self.max = 0  # 用于记录二叉树的直径（最长路径长度 ）

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.depth(root)
        return self.max

    def depth(self, root):
        if not root:
            return 0
        l = self.depth(root.left)  # 递归计算左子树深度
        r = self.depth(root.right)  # 递归计算右子树深度
        # 每个结点都判断左子树+右子树的高度和是否大于当前 self.max，更新最大值（直径可能的情况 ）
        self.max = max(self.max, l + r)  
        # 返回当前结点为根的子树的最高高度（供父结点计算使用 ）
        return max(l, r) + 1
```

节点个数：
```
def countNodes(root):
    if not root:
        return 0
    return 1 + countNodes(root.left) + countNodes(root.right)
```

回溯（马士游历（马走日））：
```
import sys
sys.setrecursionlimit(100000)  # 防止递归栈溢出

# 马走日的8个方向
dx = [-2, -2, -1, -1, 1, 1, 2, 2]
dy = [-1, 1, -2, 2, -2, 2, -1, 1]

def knight_tour(n, m, x, y):
    board = [[False for _ in range(m)] for _ in range(n)]  # 标记访问
    total_cells = n * m  # 棋盘总格子数
    count = [0]  # 使用列表修改递归中的值

    def dfs(cx, cy, visited):
        if visited == total_cells:
            count[0] += 1  # 找到一种完整路径
            return
        for i in range(8):
            nx, ny = cx + dx[i], cy + dy[i]
            # 检查是否在棋盘内且未访问
            if 0 <= nx < n and 0 <= ny < m and not board[nx][ny]:  
                board[nx][ny] = True  # 标记访问
                dfs(nx, ny, visited + 1)
                board[nx][ny] = False  # 回溯

    board[x][y] = True  # 从起点出发
    dfs(x, y, 1)
    return count[0]

T = int(input())
for _ in range(T):
    n, m, x, y = map(int, input().split())
    result = knight_tour(n, m, x, y)
    print(result)
```

**一些包&函数&方法：**

前中后序：

![(https://github.com/qianyi21/img/blob/main/image.png?raw=true)](https://raw.githubusercontent.com/qianyi21/img/b85ccdd6476f3be13ca83412b10cbf9dc209bd1a/image.png)
![https://github.com/qianyi21/img/blob/main/image.png?raw=true](https://raw.githubusercontent.com/qianyi21/img/338cbaedb710fa72f69083f16b12fa2aa21c2256/image.png)

zip函数：
```
n = int(input())
res = []
ind = []
for _ in range(n):
    c, m, e = map(int, input().split())
    res.append([c + m + e, c, m, e])
    ind.append(_)  # 原代码此处为空，需确认逻辑，假设是记录循环索引
b = list(zip(res, ind))
b.sort(key=lambda x: [-x[0][0], -x[0][1]])

for i, j in enumerate(b[:5], start=1):
    print(f"#{i}{j[1]+1} {j[0][0]}")
```

counter包：
```
from collections import Counter

n, m = map(int, input().split())
tags = list(map(int, input().split()))
fruit_list = [input() for _ in range(m)]

counter = Counter(fruit_list)
counts = sorted(counter.values(), reverse=True)
tags_sorted = sorted(tags)

# 最小总价（最便宜的价格分配给购买次数最多的水果 ）
min_total = sum(c * p for c, p in zip(counts, tags_sorted))
# 最大总价（最贵的价格分配给购买次数最多的水果 ）
max_total = sum(c * p for c, p in zip(counts, reversed(tags_sorted)))

print(min_total, max_total)
```

inf：
```
prices = list(map(int, input().split()))
min_price = float('inf')
max_profit = 0
for price in prices:
    if price < min_price:
        min_price = price
    else:
        max_profit = max(max_profit, price - min_price)
print(max_profit)
```

进制（stack）： 
```
a = int(input())
stack = []
if a == 0:
    print(0)
else:
    while a > 0:
        stack.append(a % 8)
        a //= 8
    while stack:
        print(stack.pop(), end="")
```

**（有点看不懂的部分）：**

归并排序（逆序对）：
```
import sys
sys.setrecursionlimit(1000000)

def merge_sort(arr):
    def sort_and_count(left, right):
        if right - left <= 1:
            return 0
        mid = (left + right) // 2
        # 递归统计左、右子数组逆序对，并排序
        count = sort_and_count(left, mid) + sort_and_count(mid, right)  
        i, j = left, mid
        tmp = []
        while i < mid and j < right:
            if arr[i] <= arr[j]:
                tmp.append(arr[i])
                i += 1
            else:
                tmp.append(arr[j])
                j += 1
                # 左边剩余元素都比 arr[j] 大，统计逆序对
                count += mid - i  
        # 处理剩余元素
        tmp.extend(arr[i:mid])  
        tmp.extend(arr[j:right])  
        # 合并回原数组
        arr[left:right] = tmp  
        return count

return sort_and_count(0, len(arr))

while True:
    line = sys.stdin.readline()
    if not line:
        break
    n = int(line.strip())
    if n == 0:
        break
    arr = []
    for _ in range(n):
        arr.append(int(sys.stdin.readline()))
    print(merge_sort(arr))
```

（平衡）二叉树搜索树：将有序数组转换为二叉树：
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val  # 节点值
        self.left = left  # 左子树
        self.right = right  # 右子树

class Solution:
    def sortedArrayToBST(self, nums):
        if not nums:  # 数组为空，返回空树
            return None
        mid = len(nums) // 2  # 找到数组中间位置
        # 中间值作为当前根节点的值
        root = TreeNode(nums[mid])  
        # 递归构建左子树（数组左半部分 ）
        root.left = self.sortedArrayToBST(nums[:mid])  
        # 递归构建右子树（数组右半部分 ）
        root.right = self.sortedArrayToBST(nums[mid+1:])  
        return root  # 返回构建好的平衡二叉搜索树根节点
```

MST（兔子与星空，基于并查集（Union-Find）的 Kruskal 算法）：
```
# 并查集：用于判断是否成环
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  # 父节点数组，初始每个节点父节点是自己

    def find(self, x):
        # 路径压缩：递归找根节点，同时扁平化树结构
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False  # 已在同一集合，合并失败（成环）
        self.parent[fx] = fy  # 合并集合
        return True

n = int(input())  # 节点数量
edges = []  # 存储边：(权重, 起点, 终点)

for _ in range(n - 1):  # 最后一个点不用输入（按题目逻辑）
    parts = input().split()
    from_star = ord(parts[0]) - ord('A')  # 起点转数字（A->0, B->1...）
    k = int(parts[1])
    for i in range(k):
        to_star = ord(parts[2 + i * 2]) - ord('A')  # 终点转数字
        weight = int(parts[3 + i * 2])  # 边的权重
        edges.append((weight, from_star, to_star))  # 存边

# Kruskal 算法：按权重排序，用并查集避环
edges.sort()  # 按权重升序排序
uf = UnionFind(n)  # 初始化并查集
mst_weight = 0  # 最小生成树总权重
edge_count = 0  # 已选边数量

for weight, u, v in edges:
    if uf.union(u, v):  # 合并成功（未形成环）
        mst_weight += weight
        edge_count += 1
        if edge_count == n - 1:  # 选够 n-1 条边，生成树已形成
            break

print(mst_weight)
```

根据二叉树前中序序列建图（前序 + 中序 → 后序遍历（递归构建二叉树））：
```
def build_postorder(preorder, inorder):
    if not preorder:
        return ''
    root = preorder[0]  # 前序第一个元素是根节点
    root_index = inorder.index(root)  # 找根在中序的位置

    # 分割左、右子树的前序和中序
    left_inorder = inorder[:root_index]
    left_preorder = preorder[1:1+len(left_inorder)]
    
    right_inorder = inorder[root_index+1:]
    right_preorder = preorder[1+len(left_inorder):]

    # 递归构建左、右子树的后序
    left_post = build_postorder(left_preorder, left_inorder)
    right_post = build_postorder(right_preorder, right_inorder)
    
    # 后序：左 + 右 + 根
    return left_post + right_post + root

import sys

lines = sys.stdin.read().strip().split('\n')
for i in range(0, len(lines), 2):
    preorder = lines[i].strip()
    inorder = lines[i+1].strip()
    print(build_postorder(preorder, inorder))
```

Disjoint set（宗教信仰，并查集统计连通分量）：
```
def find(parent, x):
    # 查找根节点，路径压缩优化
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, a, b):
    # 合并两个集合
    rootA = find(parent, a)
    rootB = find(parent, b)
    if rootA != rootB:
        parent[rootB] = rootA  # 合并集合

case_num = 1  # 记录测试用例编号
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break  # 终止条件
    
    # 初始化并查集：每人自成一个集合
    parent = [i for i in range(n + 1)]
    
    # 处理 m 条合并操作
    for _ in range(m):
        a, b = map(int, input().split())
        union(parent, a, b)  # 合并 a 和 b 所在集合
    
    # 统计不同的根节点数（连通分量数）
    roots = set()
    for i in range(1, n + 1):
        roots.add(find(parent, i))

    print(f"Case {case_num}: {len(roots)}")
    case_num += 1
```

献给安吉尔侬的花（地图寻路，bfs）：
```
from collections import deque

def bfs(maps, start, end, R, C):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    visited = [[False] * C for _ in range(R)]  # 初始化访问标记数组
    queue = deque()
    queue.append((start[0], start[1], 0))  # 队列元素：(x, y, 已走步数)
    visited[start[0]][start[1]] = True  # 标记起点已访问

    while queue:
        x, y, step = queue.popleft()
        if (x, y) == end:  # 到达终点，返回步数
            return step
        for dx, dy in dirs:  # 遍历四个方向
            nx, ny = x + dx, y + dy
            # 检查坐标是否在地图范围内、是否未访问、是否不是障碍物
            if 0 <= nx < R and 0 <= ny < C and not visited[nx][ny] and maps[nx][ny] != "#":
                visited[nx][ny] = True  # 标记为已访问
                queue.append((nx, ny, step + 1))  # 入队并更新步数
    return False  # 无法到达终点

T = int(input())
for _ in range(T):
    R, C = map(int, input().split())
    maps = []
    for _ in range(R):
        row = input().strip()
        maps.append(row)
    
    # 查找起点 S 和终点 E 的坐标
    start = end = None
    for i in range(R):
        for j in range(C):
            if maps[i][j] == "S":
                start = (i, j)
            elif maps[i][j] == "E":
                end = (i, j)
    
    res = bfs(maps, start, end, R, C)
    if res:
        print(res)
    else:
        print("oop!")
```

词梯（bfs）：
```
from collections import deque

def can_link(w1, w2):
    """判断两个单词是否仅差一个字符"""
    diff = 0
    for a, b in zip(w1, w2):
        if a != b:
            diff += 1
            if diff > 1:
                return False
    return diff == 1

n = int(input())
words = [input().strip() for _ in range(n)]

# 预处理：快速查找单词索引
word_set = set(words)
word_index = {word: i for i, word in enumerate(words)}

# 构建图：邻接表表示，graph[i] 存与 words[i] 仅差一个字符的单词索引
graph = [[] for _ in range(n)]
for i in range(n):
    for j in range(i + 1, n):
        if can_link(words[i], words[j]):
            graph[i].append(j)
            graph[j].append(i)

# 读取起点和终点单词
start_word, end_word = input().strip().split()

# 检查起点/终点是否存在
if start_word not in word_index or end_word not in word_index:
    print("NO")
else:
    start = word_index[start_word]
    end = word_index[end_word]

    # BFS 初始化
    queue = deque()
    queue.append(start)
    visited = [False] * n
    visited[start] = True
    prev = [-1] * n  # 记录路径前驱节点
    found = False

    # BFS 遍历找路径
    while queue:
        curr = queue.popleft()
        if curr == end:
            found = True
            break
        # 遍历邻接节点
        for neighbor in graph[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                prev[neighbor] = curr
                queue.append(neighbor)

    if not found:
        print("NO")
    else:
        # 回溯路径
        path = []
        curr = end
        while curr != -1:
            path.append(words[curr])
            curr = prev[curr]
        # 路径是逆序的，反转
        path.reverse()
        print(" ".join(path))
```

找最大连续子数组和（Kadane算法）：
```
arr = list(map(int, input().split(","))) 
n = len(arr)  # 商品数量

# Step 1: 找最大连续子数组和（Kadane算法）
max_ending_here = max_so_far = arr[0]  # 初始化当前最大和和全局最大和为第一个元素
start = end = s = 0  # 用于记录最大子数组的开始和结束位置

for i in range(1, n):
    # 如果当前元素比加上之前的和还大，说明从这里重新开始
    if max_ending_here + arr[i] < arr[i]:
        max_ending_here = arr[i]
        s = i  # 记录当前子数组起点
    else:
        max_ending_here += arr[i]

    # 更新最大和和对应区间
    if max_ending_here > max_so_far:
        max_so_far = max_ending_here
        start = s
        end = i

# Step 2: 在找到的最大连续子数组中尝试去掉一个商品，看能不能得到更大价值
total = sum(arr[start:end+1])  # 最大子数组的总和
remove_one_max = total  # 初始最大价值就是不放回任何商品

for i in range(start, end + 1):
    # 试着去掉第 i 个商品
    # 只有当子数组长度大于1时，才有可能去掉一个商品
    if end - start >= 1:
        remove_one_max = max(remove_one_max, total - arr[i])

# 最大价值（考虑了放回一个商品的情况）
print(remove_one_max)
```

拓扑（海军，基于拓扑排序判断有向图是否有环）：
```
from collections import deque
import sys

input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    graph = [[] for _ in range(N + 1)]  # 邻接表存图
    indegree = [0] * (N + 1)  # 入度数组

    # 读入边，构建图和入度
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        indegree[v] += 1

    # 初始化队列：入度为0的节点
    queue = deque([i for i in range(1, N + 1) if indegree[i] == 0])
    count = 0  # 记录拓扑排序访问的节点数

    # 拓扑排序过程
    while queue:
        node = queue.popleft()
        count += 1
        # 遍历邻接节点，更新入度
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # 判断是否有环：拓扑排序节点数 == N 则无环，否则有环
    print("No" if count == N else "Yes")
```

Dijkstra（走山路（最小体力、路径））：
```
import heapq
import sys
input = sys.stdin.readline

m, n, p = map(int, input().split())
grid = []
for _ in range(m):
    row = input().split()
    grid.append(row)

queries = []
for _ in range(p):
    sx, sy, ex, ey = map(int, input().split())
    queries.append((sx, sy, ex, ey))

# 四个方向
dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def in_bounds(x, y):
    return 0 <= x < m and 0 <= y < n

def dijkstra(sx, sy, ex, ey):
    # 起点或终点是障碍物，直接返回 NO
    if grid[sx][sy] == "#" or grid[ex][ey] == "#":
        return "NO"
    
    # 距离数组初始化，inf 表示不可达
    dist = [[float('inf')] * n for _ in range(m)]
    dist[sx][sy] = 0  # 起点距离为 0
    heap = [(0, sx, sy)]  # 优先队列：(当前成本, x, y)

    while heap:
        cost, x, y = heapq.heappop(heap)
        # 到达终点，返回当前成本
        if (x, y) == (ex, ey):
            return cost
        # 已处理过更优路径，跳过
        if cost > dist[x][y]:
            continue
        # 遍历四个方向
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            # 检查是否在网格内且不是障碍物
            if in_bounds(nx, ny) and grid[nx][ny] != "#":
                # 计算高度差（移动成本）
                height_diff = abs(int(grid[x][y]) - int(grid[nx][ny]))
                new_cost = cost + height_diff
                # 发现更优路径，更新距离并入队
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    heapq.heappush(heap, (new_cost, nx, ny))
    
    # 无法到达终点
    return "NO"

for sx, sy, ex, ey in queries:
    print(dijkstra(sx, sy, ex, ey))
```

Trie（电话号码，基于字典树判断电话号码是否一致）：
```
class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点字典，键是数字字符，值是 TrieNode
        self.is_end = False  # 标记是否为某个号码的结尾

def is_consistent(phone_numbers):
    root = TrieNode()  # 字典树根节点
    # 先排序，保证短号码（可能成为前缀的）先插入
    for number in sorted(phone_numbers):
        node = root
        for digit in number:
            # 如果当前节点已标记为号码结尾，说明存在前缀冲突
            if node.is_end:
                return False
            # 不存在则创建子节点
            if digit not in node.children:
                node.children[digit] = TrieNode()
            # 移动到子节点
            node = node.children[digit]
        # 插入完号码后，检查当前节点是否有子节点（当前号码是其他号码的前缀）
        if node.children:
            return False
        # 标记当前节点为号码结尾
        node.is_end = True
    return True

t = int(input())
for _ in range(t):
    n = int(input())
    # 读取号码，去除空格
    numbers = [input().strip().replace(" ", "") for _ in range(n)]
    # 判断是否一致，输出结果
    print("YES" if is_consistent(numbers) else "NO")
```

