# 刷题记录与思考

## 双指针

### 001-两数之和

考查双指针的使用，采用双指针遍历。

```python
# 20ms，复杂度较高
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i,it in enumerate(nums):
            for j in range(i+1, len(nums)):
                if it+nums[j]==target:
                    return i,j
```

一者遍历，一者通过hash表(字典数据)，hash表所花时间反而更长？

```python
# 24ms
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hash_map = {key:i for i,key in enumerate(nums)}
        for i,it in enumerate(nums):
            j = hash_map.get(target - it)
            if j:
                if j!=i:
                    return i,j

# 28ms                
class Solution(object):
    def twoSum(self, nums, target):
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []             
```

### 15-三数之和

> 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
>
> 注意：答案中不可以包含重复的三元组。

暴力搜索，超时···

```python
class Solution(object):
    def threeSum(self, nums):
        if len(nums)<3:
            return []
        
        result = []
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    if (nums[i] + nums[j] + nums[k])==0:
                        tmp = [nums[i], nums[j], nums[k]]
                        if set(tmp) not in [set(it) for it in result]:
                            result.append(tmp)
        return result
        
nums = [-1,0,1,2,-1,-4]
s = Solution()
print(s.threeSum(nums))  
```

因而考虑**排序+双指针**：

- 特判，对于数组长度 n，如果数组为 null 或者数组长度小于 3，返回 [][]。
- 对数组进行排序。
- 遍历排序后数组：
  - 若 nums[i]>0：因为已经排序好，所以后面**不可能有**三个数加和等于 0，直接返回结果。
  - 对于重复元素：跳过，避免出现重复解
  - 令左指针 L=i+1，右指针 R=n-1，当 L<R 时，执行循环：
    - 当 nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复（指针位置不断变化，直到未出现重复元素），去除重复解。并同时将 L,R 移到下一位置，寻找新的解。
    - 若和大于 0，说明 nums[R]太大，R左移
    - 若和小于 0，说明 nums[L]太小，L右移

```python
class Solution(object):
    def threeSum(self, nums):
        n = len(nums)
        if n<3: return []

        # 先排序，关键！
        nums.sort()     
        result = []
        
        for i in range(n):
            if nums[i]>0:
                return result
            
            if i > 0 and nums[i-1]==nums[i]: # 重复元素跳过
                continue
            
            # 两指针
            L = i + 1
            R = n - 1
            
            while (L < R):
                if (nums[i] + nums[L] + nums[R])==0: # 满足条件
                    result.append([nums[i], nums[L], nums[R]])
                    while (L < R and nums[L] == nums[L + 1]): # 不断移动指针，直到不再重复
                        L +=1
                    while (L < R and nums[R] == nums[R - 1]): # 不断移动指针，直到不再重复
                        R -=1
                    # 移动指针
                    L +=1
                    R -=1
                elif (nums[i] + nums[L] + nums[R])>0:
                    R -=1
                else:
                    L +=1
        return result
        
nums = [-2,0,3,-1,4,0,3,4,1,1,1,-3,-5,4,0]        
s = Solution()
print(s.threeSum(nums))  
```

### 152-乘积最大子数组

> 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

0与任何数的乘积都是0，因此可以将数组看成被0分割的子数组，在各个子数组中查找乘积最大的值。
在一个非0的数组中求最大乘积，需要分析正负性。
- 没有负数或者负数为偶数个，最大乘积就是整个数组的乘积
- 有奇数个负数，如果第i个元素为负数，则[start,i-1]，[i+1,end]这2个区间的乘积都是最大乘积的候选。

通过下面2个指针交替移动算法可以计算所有[start,i-1]和[i+1,end]的乘积。

- right指针向右移动，mul累计left至right指针之间的乘积，直至right遇到0或末尾。
- 向右移动left指针，mul除以被移出子数组的元素。
- 重复以上过程直至left指针移动到末尾。

```python
class Solution:
    def maxProduct(self, nums):
        left, right, n = 0, 0, len(nums)
        mul, product = 1, float('-inf')
        while left < n:
            # 移动right指针直至遇到0，这中间用mul累计乘积，product记录最大的乘积
            while right < n and nums[right] != 0:  
                mul *= nums[right]
                right += 1
                product = max(product, mul)
            # 移动left指针，这中间用mul累计乘积，product记录最大的乘积
            while left + 1 < right:  
                mul /= nums[left]
                left += 1
                product = max(product, mul)
            # 跳过0
            while right < n and nums[right] == 0:  
                product = max(product, 0) # 有可能所有子数组的乘积都小于0，所以0也是候选
                right += 1
            left = right
            mul = 1
        return int(product)
```

优化：

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prefix, suffix, res = 0, 0, -float('inf')
        for i in range(len(nums)):
            prefix = nums[i] if not prefix else nums[i]*prefix
            suffix = nums[-i - 1] if not suffix else nums[-i - 1]*suffix
            res = max(res, prefix, suffix)
        return res
```

### 179-最大数

> 给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
>
> 注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。
>
> ```
> 输入：nums = [3,30,34,5,9]
> 输出："9534330"
> ```

自定义一种排序方式 比较 s1 + s2 和 s2 + s1

```python
# 冒泡排序
class Solution(object):
    def largestNumber(self, nums):
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                # 对比前后组成的数，以大小交换顺序
                if int(str(nums[i])+str(nums[j])) < int(str(nums[j])+str(nums[i])): 
                    nums[i], nums[j] = nums[j], nums[i]
        res = ''.join(str(item) for item in nums)
        return str(int(res))

# 希尔排序
class Solution(object):
    def largestNumber(self, nums):
        n = len(nums)
        step = int(n/2)
        while step > 0:
            for i in range(step, n):
                while i >= step and int(str(nums[i-step])+str(nums[i])) < int(str(nums[i])+str(nums[i-step])):
                    nums[i], nums[i-step] = nums[i-step], nums[i]
                    i -=step
            step=int(step/2)  
            
        res = ''.join(str(item) for item in nums)
        return str(int(res))

# 采用cmp_to_key 函数，可以接受两个参数，将两个参数做处理，比如做和做差，转换成一个参数，就可以应用于key关键字
class Solution(object):
    def largestNumber(self, nums):
         from functools import cmp_to_key
         return str(int(''.join(sorted(map(str, nums), key=cmp_to_key(lambda x,y:int(y+x)-int(x+y))))))

# cmp_to_key的例子
from functools import cmp_to_key 
L=[9,2,23,1,2]

sorted(L,key=cmp_to_key(lambda x,y:y-x))
输出：
[23, 9, 2, 2, 1]
```



## 滑动窗口

### 3-无重复字符的最长子串

> 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

暴力搜索(超时)：

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        L = len(s)
        if L<2:
            return L
        
        dp = 0   
        for i in range(L):
            for j in range(i+1, L+1):
                slice = s[i:j] 
                if len(slice) == len(set(slice)):
                    if len(slice) > dp:
                        dp = len(slice)
        return dp
```

双指针(滑框)

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        L = len(s)
        if L<2: # 长度小于2，直接返回
            return L
        
        dp = 1  # 最长长度
        head = 0 # 第一个指针
        tail = 1 # 第二个指针
        while tail < L:
            if s[tail] not in s[head:tail]: # 如果末端元素不在已有的序列中，将指针指向下一个
                tail +=1
            else: # 如果已在序列中，获得已有的元素位置，并从此处重新开始计算
                head += s[head:tail].index(s[tail])+1
            dp = max(dp, tail-head) # 更新已有的最大值
                
        return dp
```

### 239-滑动窗口最大值

> 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
>
> 返回滑动窗口中的最大值。

暴力搜索(超时)：

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        L = len(nums)
        if L<2:
            return nums
        
        ans = []
        head = 0
        tail = k
        while tail < L+1:
            ans.append(max(nums[head:tail]))
            tail +=1
            head +=1
        return ans
```

通过 **"双端队列"** ，也就是两边都能进能出的队列。

首先就是入队列，每次滑动窗口都把最大值左边小的数踢掉，也就是出队，后面再滑动窗口进行维护，这样相当于就是每一个数走过场。时间复杂度就是O（N*1）

遍历数组，当前元素为 num，索引为 i

- 当队列非空，左边界出界时（滑动窗口向右移导致的），更新左边界
- 当队列非空，将队列中索引对应的元素值比 num 小的移除
- 更新队列
- 当索引 i 大于 k-1，更新输出结果

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        if not nums:
            return []
 
        window, res = [], []
    
        for i, num in enumerate(nums):
            # 窗口滑动时的规律, 即滑动的位置超出了windows范围
            if i>=k and window[0] <= i-k:
                window.pop(0)
                
            # 把最大值左边的数小的清除，即判断新的数据Num,其与windows中值的比较：从最右端开始，Windows中的数据小于Num则弹出，否则停止，说明Windows中存的是从左到右，数值降低的序列。新的数据，需要与原有的序列进行判断
            while window and nums[window[-1]] <= num:
                window.pop()
                
            window.append(i) # # 队首一定是滑动窗口的最大值的索引
            
            if i >= k-1:
                res.append(nums[window[0]])
        return res
```

### 718-最长重复子数组

> 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
>
>  示例：
>
> 输入：A: [1,2,3,2,1]，B: [3,2,1,4,7]
> 输出：3
>
> 解释：长度最长的公共子数组是 [3, 2, 1] 。

以类似卷积的方式进行：

<img src="https://pic.leetcode-cn.com/9ed48b9b51214a8bafffcad17356d438b4c969b4999623247278d23f1e43977f-%E9%94%99%E5%BC%80%E6%AF%94%E8%BE%83.gif" alt="错开比较.gif" style="zoom:45%;" />

```python
class Solution:
    def findLength(self, nums1, nums2):
        self.res = 0
        n1, n2 = len(nums1), len(nums2)

        def getLength(i, j): # 找对齐部分最长
            cur = 0
            while i < n1 and j < n2:
                if nums1[i] == nums2[j]: # 如果相等，则对齐长度增加，否则置0
                    cur += 1
                    self.res = max(self.res, cur)
                else:
                    cur = 0
                i, j = i + 1, j + 1

        # 两个数组分别以自己的头怼另外一个数组一遍
        for j in range(n2): getLength(0, j)
        for i in range(n1): getLength(i, 0)
        return self.res
```

指针型：

```python
class Solution(object):
    def findLength(self, nums1, nums2):
        res = dp = 0
        if nums1 and nums2:
            # chr为ASCII码转换，没懂
            a, b, n = ''.join(map(chr, nums1)), ''.join(map(chr, nums2)), len(nums1)
            while dp + res <n:
                if a[dp:dp+res+1] in b:
                    res +=1
                else:
                    dp +=1                           
        return res
```

## 链表

### 206-反转链表

**双指针迭代**
我们可以申请两个指针：

- 第一个指针叫 pre，最初是指向 null 的。
- 第二个指针 cur 指向 head，然后不断遍历 cur。
- 每次迭代到 cur，都将 cur 的 next 指向 pre，然后 pre 和 cur 前进一位。
- 都迭代完了(cur 变成 null 了)，pre 就是最后一个节点了。

动画演示如下：

<img src="https://pic.leetcode-cn.com/7d8712af4fbb870537607b1dd95d66c248eb178db4319919c32d9304ee85b602-%E8%BF%AD%E4%BB%A3.gif" alt="迭代.gif" style="zoom:80%;" />

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None # 空值
        current = head # 当前所在值
        while current: # 当当前所在值不为空，也即未运行至链表末尾
            temp = current.next # 将next值保存为中间变量
            current.next = pre # 将next指向前一个
            pre = current # 移动前一个为当下的值
            current = temp # 当前值指向之前保存的next值，其实类似于对角线

            # 以上可简化为一行，利用赋值即可 
            # current.next, pre, current = pre, current, current.next
        return pre
```

### 剑指 Offer 22. 链表中倒数第k个节点

> 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
>
> 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
>
> ```python
> 给定一个链表: 1->2->3->4->5, 和 k = 2.
> 返回链表 4->5.
> ```

快慢指针，快的先走k步，然后再一起同步走。
```python
class Solution(object):
    def getKthFromEnd(self, head, k):
        fast, slow = head, head
        for _ in range(k): # 快指针先走k步
            if fast:
                fast = fast.next
            else:
                return None
        while fast:
            slow, fast = slow.next, fast.next
        return slow
```

### 143-重排链表

> 给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
> 将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
>
> 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
>
> ```
> 给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
> ```

思路：重排列表，可以视为正反序列表，岔开插入，而插入长度为列表长度的一半。

```python
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def reorderList(self, head):
        if not head or not head.next: # 列表为空，或者只有一个元素
            return None
        # 快慢指针，用于确定链表的中间
        slow = head
        fast = head    # 行进速度为2step    
        while fast.next and fast.next.next: # 确保前进有值
            fast = fast.next.next
            slow = slow.next           
        
        second = slow.next # 后半段链表head
        node = slow.next.next # 下一个值
        slow.next = None
        second.next = None
        
        while node: # 反转后半段链表
            tmp = node.next
            node.next = second
            second = node
            node = tmp
        
        first = head # 前半段链表
        while second:
            tmp = first.next # 保存前半段链表的下一个值
            first.next = second # 插入后半段链表的值
            tmp2 = second.next # 保存后半段链表的下一个值
            second.next = tmp # 传递前半段链表的下一个值
            first = tmp # 移动至前半段链表的下一个值
            second = tmp2 # 移动至后半段链表的下一个值
            
        return first
L = ListNode(1)
L.next = ListNode(2)
L.next.next = ListNode(3)          
L.next.next.next = ListNode(4) 
L.next.next.next.next = ListNode(5) 
     
s = Solution()
print(s.reorderList(L))  

# 简化版
class Solution(object):
    def reorderList(self, head):
        if not head or not head.next:
            return None
        slow = fast = first = head
        while fast.next and fast.next.next:
            fast, slow = fast.next.next, slow.next           
        
        second, node, slow.next, second.next = slow.next, slow.next.next, None, None
        
        while node:
            node.next, second, node = second, node, node.next
        
        while second:
            first.next, second.next, first, second = second, first.next, first.next, second.next   
        return first
```

### 21-合并两个有序链表

> 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
>
> 输入：l1 = [1,2,4], l2 = [1,3,4]
> 输出：[1,1,2,3,4,4]

```python
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not (l1 and l2): # 判断l1或l2中，存在一个为None
            return l1 if l1 else l2

        if l1.val < l2.val: # 如果l1的当前值小于l2，后面的值接在l1后面，并递归追溯
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else: # 相反情况
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

### 23-合并K个升序链表

> 给你一个链表数组，每个链表都已经按升序排列。
>
> 请你将所有链表合并到一个升序链表中，返回合并后的链表。
>
> 示例 1：
>
> 输入：lists = [[1,4,5],[1,3,4],[2,6]]
> 输出：[1,1,2,3,4,4,5,6]
> 解释：链表数组如下：
> [
>   1->4->5,
>   1->3->4,
>   2->6
> ]
> 将它们合并到一个有序链表中得到。
> 1->1->2->3->4->4->5->6

思路：合并k个，则可以分解为若干个子问题，链表两两合并，如上一题。即为分治思想。

```python
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        n = len(lists)
        if n==0: return None
        if n==1: return lists[0]

        mid = n//2 # 分治，成为两部分链表
        return self.mergeTwo(self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:]))

    def mergeTwo(self, l1, l2):
        if not (l1 and l2):
            return l1 if l1 else l2
        if l1.val < l2.val:
            l1.next = self.mergeTwo(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwo(l1, l2.next)
            return l2
```

### 141-环形链表

> 给定一个链表，判断链表中是否有环。
>
> 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
>
> 如果链表中存在环，则返回 true 。 否则，返回 false 。
>
> <img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png" alt="img" style="zoom:80%;" />

本题核心思路：走 a+nb 步一定处于环的入口位置

假设链表环前有 a个节点，环内有 b个节点

- 利用快慢指针 fast 和 slow，fast 一次走两步，slow 一次走一步
- 当两个指针第一次相遇时，假设 slow 走了 s步，下面计算 fast 走过的步数
  - fast 比 slow多走了 n个环：f = s + nb
  - fast 比 slow多走一倍的步数：f = 2s--> 跟上式联立可得 s = nb
  - 综上计算得，f = 2nb，s = nb

```python
class Solution(object):
    def hasCycle(self, head):
        slow = fast = head
        while True:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        return True
```

### 142-环形链表 II

> 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
>
> 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
>
> 说明：不允许修改给定的链表。
>
> 进阶：你是否可以使用 O(1) 空间解决此题？
>
>
> 示例 1：
>
> <img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png" alt="img" style="zoom:67%;" />
>
> 输入：head = [3,2,0,-4], pos = 1
> 输出：返回索引为 1 的链表节点
> 解释：链表中有一个环，其尾部连接到第二个节点。

本题**核心思路**：走 a+nb 步一定处于环的入口位置

假设链表环前有 a个节点，环内有 b个节点

<img src="https://pic.leetcode-cn.com/1618325270-ZZpHMB-%E9%93%BE%E8%A1%A8.png" alt="链表.png" style="zoom:67%;" />

- 利用快慢指针 fast 和 slow，fast 一次走两步，slow 一次走一步
- 当两个指针第一次相遇时，假设 slow 走了 s步，下面计算 fast 走过的步数
  - fast 比 slow多走了 n个环：f = s + nb
  - fast 比 slow多走一倍的步数：f = 2s--> 跟上式联立可得 s = nb
  - 综上计算得，f = 2nb，s = nb
- 也就是两个指针第一次相遇时，都走过了环的倍数，那么再走 a步就可以到达环的入口
- 让 fast从头再走，slow留在原地，fast和 slow均一次走一步，当两个指针第二次相遇时，fast走了 a步，slow走了 a+nb步

此时 slow就在环的入口处，返回 slow

```python
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head
        while True: # 先判断是否有环
            if not fast or not fast.next:
                return None
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        fast = head
        while slow != fast: # 找到入口
            fast = fast.next
            slow = slow.next
        return slow
```



## 215-数组中的第K个最大元素

一般想法是，先排序，后定位寻找。

```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort() #默认升序排序
        return nums[len(nums)-k]
```

**借助 partition 操作定位到最终排定以后索引为 len - k 的那个元素（特别注意：随机化切分元素）**

以下是注意事项，因为很重要，所以放在前面说：

快速排序虽然快，但是如果实现得不好，在遇到特殊测试用例的时候，时间复杂度会变得很高。如果你使用 partition 的方法完成这道题，时间排名不太理想，可以考虑一下是什么问题，这个问题很常见。

以下的描述基于 “快速排序” 算法知识的学习，如果忘记的朋友们可以翻一翻自己的《数据结构与算法》教材，复习一下，partition 过程、分治思想和 “快速排序” 算法的优化。

分析：我们在学习 “快速排序” 的时候，接触的第 1 个操作就是 partition（切分），简单介绍如下：

partition（切分）操作，使得：

- 对于某个索引 j，nums[j] 已经排定，即 nums[j] 经过 partition（切分）操作以后会放置在它 “最终应该放置的地方”；
- nums[left] 到 nums[j - 1] 中的所有元素都不大于 nums[j]；
- nums[j + 1] 到 nums[right] 中的所有元素都不小于 nums[j]。

<img src="https://pic.leetcode-cn.com/65ec311c3e9792bb17e9c08cabd4a07f251c9cd65a011b6c5ffb54b46d8e5012-image.png" alt="image.png" style="zoom:40%;" />

partition（切分）操作总能排定一个元素，还能够知道这个元素它最终所在的位置，这样每经过一次 partition（切分）操作就能缩小搜索的范围，这样的思想叫做 “减而治之”（是 “分而治之” 思想的特例）。

切分过程可以不借助额外的数组空间，仅通过交换数组元素实现。

```python
# -*- coding: utf-8 -*-
class Solution:
    def findKthLargest(self, nums, k):
        size = len(nums)
        target = size - k
        left = 0
        right = size - 1
        while True:
            index = self.__partition(nums, left, right)
            if index == target:
                return nums[index]
            elif index < target:
                # 下一轮在 [index + 1, right] 里找
                left = index + 1
            else:
                right = index - 1

    #  循环不变量：[left + 1, j] < pivot
    #  (j, i) >= pivot
    def __partition(self, nums, left, right):

        pivot = nums[left]
        j = left
        for i in range(left + 1, right + 1):
            if nums[i] < pivot:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]

        nums[left], nums[j] = nums[j], nums[left]
        return j
```

### 快速排序-分而治之

```python
def quicksort(my_list):
    if len(my_list) < 2:  # 基线条件   列表中只有1个或0个元素
        return my_list
    else:  # 递归条件
        num = my_list[0]  # 将列表的第一个数定义为基准值
        list_a = [i for i in my_list[1:] if i <= num]  # 比基准值小的放在一个列表
        list_b = [i for i in my_list[1:] if i > num]  # 比基准值大的放在一个列表
        return quicksort(list_a) + [num] + quicksort(list_b) # 存在递归排序

print(quicksort([1, 22, 3, 4, 5, 6, 7, 8, 9]))
```

自己实现排序算法，再取第k个：

```python
# -*- coding: utf-8 -*-
class Solution:
    from random import randint
    def findKthLargest(self, nums, k):
        l = self.quicksort(nums)
        print(l)
        return l[len(l)-k]
    
    def quicksort(self, my_list):
        if len(my_list) < 2:  # 基线条件   列表中只有1个或0个元素
            return my_list
        else:  # 递归条件
            n = randint(0, len(my_list)-1)
            num = my_list[n]  # 随意取一个数定义为基准值
            list_a = [i for i in (my_list[:n] + my_list[n+1:]) if i <= num]  # 比基准值小的放在一个列表
            list_b = [i for i in (my_list[:n] + my_list[n+1:]) if i > num]  # 比基准值大的放在一个列表
            return self.quicksort(list_a) + [num] + self.quicksort(list_b) # 存在递归排序
```

不必排序，分而治之，在最大、最小里面找：

```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k > len(nums):
            return
        index = randint(0,len(nums)-1)
        pivot = nums[index]
        small=[i for i in (nums[:index]+nums[index+1:])if i < pivot] # 比pivot都小的放入small
        large=[i for i in (nums[:index]+nums[index+1:])if i >= pivot] # 比pivot都大的放入large

        if k-1 == len(large): # 所有比第k个数大的集合
            return pivot
        elif k-1<len(large): # 如果小于，说明large中元素多了，在large中
            return self.findKthLargest(large, k)
        if k-1 > len(large): # 如果大于，说明large中元素少了，在small中
                return self.findKthLargest(small, k-1-len(large))
```

## 动态规划

### 72-编辑距离

> 给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2` 所使用的最少操作数 。

最直观的方法是暴力检查所有可能的编辑方法，取最短的一个。所有可能的编辑方法达到指数级，但我们不需要进行这么多计算，因为我们只需要找到距离最短的序列而不是所有可能的序列。

**思路和算法**

我们可以对任意一个单词进行三种操作：

- 插入一个字符；

- 删除一个字符；

- 替换一个字符。


题目给定了两个单词，设为 A 和 B，这样我们就能够六种操作方法。

但我们可以发现，如果我们有单词 A 和单词 B：

- 对单词 A **删除**一个字符和对单词 B **插入**一个字符是等价的。例如当单词 A 为 doge，单词 B 为 dog 时，我们既可以删除单词 A 的最后一个字符e，得到相同的 dog，也可以在单词 B 末尾添加一个字符 e，得到相同的 doge；

- 同理，对单词 B **删除**一个字符和对单词 A **插入**一个字符也是等价的；

- 对单词 A **替换**一个字符和对单词 B **替换**一个字符是等价的。例如当单词 A 为 bat，单词 B 为 cat 时，我们修改单词 A 的第一个字母 b -> c，和修改单词 B 的第一个字母 c -> b 是等价的。


这样以来，本质不同的操作实际上只有三种：

- 在单词 A 中插入一个字符；

- 在单词 B 中插入一个字符；

- 修改单词 A 的一个字符。


这样以来，我们就可以把原问题转化为规模较小的子问题。我们用 A = horse，B = ros 作为例子，来看一看是如何把这个问题转化为规模较小的若干子问题的。

- 在**单词 A 中插入**一个字符：如果我们知道 horse 到 ro 的编辑距离为 a，那么显然 horse 到 ros 的编辑距离不会超过 a + 1。这是因为我们可以在 a 次操作后将 horse 和 ro 变为相同的字符串，只需要额外的 1 次操作，在单词 A 的末尾添加字符 s，就能在 a + 1 次操作后将 horse 和 ro 变为相同的字符串；

- 在**单词 B 中插入**一个字符：如果我们知道 hors 到 ros 的编辑距离为 b，那么显然 horse 到 ros 的编辑距离不会超过 b + 1，原因同上；

- **修改单词 A** 的一个字符：如果我们知道 hors 到 ro 的编辑距离为 c，那么显然 horse 到 ros 的编辑距离不会超过 c + 1，原因同上。

那么从 horse 变成 ros 的编辑距离应该为 min(a + 1, b + 1, c + 1)。

**注意：**为什么我们总是在单词 A 和 B 的末尾插入或者修改字符，能不能在其它的地方进行操作呢？答案是可以的，但是我们知道，操作的顺序是不影响最终的结果的。例如对于单词 cat，我们希望在 c 和 a 之间添加字符 d 并且将字符 t 修改为字符 b，那么这两个操作无论为什么顺序，都会得到最终的结果 cdab。

你可能觉得 horse 到 ro 这个问题也很难解决。但是没关系，我们可以继续用上面的方法拆分这个问题，对于这个问题拆分出来的所有子问题，我们也可以继续拆分，直到：

- 字符串 A 为空，如从 转换到 ro，显然编辑距离为字符串 B 的长度，这里是 2；

- 字符串 B 为空，如从 horse 转换到 ，显然编辑距离为字符串 A 的长度，这里是 5。


因此，我们就可以使用动态规划来解决这个问题了。我们用 D[i][j] 表示 A 的前 i 个字母和 B 的前 j 个字母之间的编辑距离。

<img src="https://pic.leetcode-cn.com/426564dbe63a8cdec3de2ebe83ea2a2640bbff41d18c1bac739c9ae4542854af-72_fig1.PNG" alt="72_fig1.PNG" style="zoom:20%;" />

如上所述，当我们获得 D\[i][j-1]，D\[i-1][j] 和 D\[i-1][j-1] 的值之后就可以计算出 D\[i][j]。

- D\[i][j-1] 为 A 的前 i 个字符和 B 的前 j - 1 个字符编辑距离的子问题。即对于 B 的第 j 个字符，我们在 A 的末尾添加了一个相同的字符，那么 D[i][j] 最小可以为 D\[i][j-1] + 1；

- D\[i-1][j] 为 A 的前 i - 1 个字符和 B 的前 j 个字符编辑距离的子问题。即对于 A 的第 i 个字符，我们在 B 的末尾添加了一个相同的字符，那么 D[i][j] 最小可以为 D\[i-1][j] + 1；

- D\[i-1][j-1] 为 A 前 i - 1 个字符和 B 的前 j - 1 个字符编辑距离的子问题。即对于 B 的第 j 个字符，我们修改 A 的第 i 个字符使它们相同，那么 D\[i][j] 最小可以为 D\[i-1][j-1] + 1。特别地，如果 A 的第 i 个字符和 B 的第 j 个字符原本就相同，那么我们实际上不需要进行修改操作。在这种情况下，D\[i][j] 最小可以为 D\[i-1][j-1]。


那么我们可以写出如下的状态转移方程：

若 A 和 B 的最后一个**字母相同**：
$$
\begin{aligned} 

D[i][j] &= \min(D[i][j - 1] + 1, D[i - 1][j]+1, D[i - 1][j - 1])\\ &= 1 + \min(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1] - 1) 

\end{aligned}
$$

若 A 和 B 的最后一个**字母不同**：

$$
D[i][j] = 1 + \min(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1])
$$


所以每一步结果都将基于上一步的计算结果，示意如下：

<img src="https://pic.leetcode-cn.com/3241789f2634b72b917d769a92d4f6e38c341833247391fb1b45eb0441fe5cd2-72_fig2.PNG" alt="72_fig2.PNG" style="zoom:20%;" />

对于边界情况，一个空串和一个非空串的编辑距离为 D\[i][0] = i 和 D\[0][j] = j，D\[i][0] 相当于对 word1 执行 i 次删除操作，D\[0][j] 相当于对 word1执行 j 次插入操作。

**总结：**

dp\[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数

所以，

- 当 word1[i] == word2[j]，dp\[i][j] = dp\[i-1][j-1]；

- 当 word1[i] != word2[j]，dp\[i][j] = min(dp\[i-1][j-1], dp\[i-1][j], dp\[i][j-1]) + 1


其中，dp\[i-1][j-1] 表示**替换**操作，dp\[i-1][j] 表示**删除**操作，dp\[i][j-1] 表示**插入**操作。

```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n1, n2 = len(word1), len(word2)
        
        # 有一个字符串为空串
        if n1 * n2 == 0:
            return n1 + n2
        
        # DP 数组
        dp = [[0] * (n2 + 1) for _ in xrange(n1 + 1)]

        # 边界状态初始化
        for i in xrange(1, n1 + 1):
            dp[i][0] = i
        for j in xrange(1, n2 + 1):
            dp[0][j] = j
            
        # 计算所有 DP 值
        for i in xrange(1, n1 + 1):
            for j in xrange(1, n2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[n1][n2]

# 内存优化
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n1, n2 = len(word1), len(word2)
        dp = [0] * (n2 + 1) #保留一行
        dp[0] = 0
        for j in xrange(1, n2 + 1):
            dp[j] = j
        for i in xrange(1, n1 + 1):
            old_dp_j = dp[0]
            dp[0] = i
            for j in xrange(1, n2 + 1):
                old_dp_j_1, old_dp_j = old_dp_j, dp[j]
                if word1[i - 1] == word2[j - 1]:
                    dp[j] = old_dp_j_1
                else:
                    dp[j] = min(dp[j], dp[j - 1], old_dp_j_1) + 1
        return dp[n2]
```

### 1143-最长公共子序列

> 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
>
> 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
>
> - 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
>
> 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

此题跟上一题类似：

状态定义：

- dp[i][j] 表示text1[0\~i-1]和text2[0\~j-1]的最长公共子序列长度 
- dp\[0][0]等于0，等于dp数组总体往后挪了一个，免去了判断出界 

转移方程： 

- text1[i-1] == text2[j-1] 当前位置匹配上了: dp\[i][j] = dp[i-1]\[j-1]+1 
- text1[i-1] != text2[j-1] 当前位置没匹配上了 ：dp\[i][j] = max(dp\[i-1][j], dp\[i][j-1]); 
- base case: 任何一个字符串为0时都是零，初始化时候就完成了base case是赋值

```python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        n1, n2 = len(text1), len(text2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n1][n2]
```

### 718-最长重复子数组

> 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
>
>  示例：
>
> 输入：A: [1,2,3,2,1]，B: [3,2,1,4,7]
> 输出：3
>
> 解释：长度最长的公共子数组是 [3, 2, 1] 。

此题跟上一题类似，但不同的是，需要连续的数组，要考虑连续性。

状态定义：

- dp[i][j] 表示text1[0\~i-1]和text2[0\~j-1]的最长公共子序列长度 
- dp\[0][0]等于0，等于dp数组总体往后挪了一个，免去了判断出界 

转移方程： 

- text1[i-1] == text2[j-1] 当前位置匹配上了: dp\[i][j] = dp[i-1]\[j-1]+1 
- text1[i-1] != text2[j-1] 当前位置没匹配上了 ：dp\[i][j] = 0（这部分与上面不同，因为一旦不同，需要重新置0） 
- base case: 任何一个字符串为0时都是零，初始化时候就完成了base case是赋值

```python
class Solution(object):
    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        n1, n2 = len(nums1), len(nums2)
        dp = [[0]*(n2+1) for _ in range(n1+1)]
        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if nums1[i-1]==nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
        return max(max(item) for item in dp)

# 例如dp输出
  3 2 1 4 7
1 0 0 1 0 0
2 0 1 0 0 0
3 1 0 0 0 0
2 0 2 0 0 0
1 0 0 3 0 0
```



### 300-最长上升子序列

> 给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**思路与算法**

定义dp[i] 为考虑前 i 个元素，以第 i个数字结尾的最长上升子序列的长度，注意nums[i] 必须被选取。

我们从小到大计算dp 数组的值，在计算dp[i] 之前，我们已经计算出dp[0,…,i−1] 的值，则状态转移方程为：

$$
dp[i]=max(dp[j])+1,其中0≤j<i且num[j]<num[i]
$$


即考虑往dp[0,…,i−1] 中最长的上升子序列后面再加一个 nums[i]。由于 dp[j] 代表nums[0,…,j] 中以nums[j] 结尾的最长上升子序列，所以如果能从 dp[j] 这个状态转移过来，那么nums[i] 必然要大于nums[j]，才能将nums[i] 放在 nums[j] 后面以形成更长的上升子序列。

最后，整个数组的最长上升子序列即所有dp[i] 中的最大值。
$$
\text{LIS}_{\textit{length}}= \max(\textit{dp}[i]), \text{其中} \, 0\leq i < n
$$


```python
# 时间复杂度：O(N^2)
class Solution:
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
# 二分查找：O(NlogN)，没太懂
class Solution(object):
    def lengthOfLIS(self, nums):
        # 定义新的列表
        tails = [0] * len(nums)
        # 定义已经插入的序列的长度
        size = 0
        for x in nums:
            # 此处用二分查找，因为是排好序的序列，所以每次只与中间值比较
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size

# 输出序列，也没太懂
def lis(arr):
    n = len(arr)
    # m用来存储个数
    m = [0]*n
    # 序列从右到左，逐一比较
    for x in range(n-2,-1,-1):
        for y in range(n-1,x,-1):
            if arr[x] < arr[y] and m[x] <= m[y]:
                m[x] += 1
        max_value = max(m)
        result = []
        for i in range(n):
            if m[i] == max_value:
                result.append(arr[i])
                max_value -= 1
    return result
 
arr = [0,1,0,3,2,3]
print(lis(arr))
```



### 53-最大子序和

> 给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

此题与上一题思路一致：

在计算子序和dp[i] 之前，我们已经计算出前i-1个num的子序和dp[i−1] 值，那么对于第i个数num[i]，要考虑原子序和dp[i−1]的值：

- 当dp[i−1]<=0时，不管num[i]为何值（>0，<0，=0），都要丢弃原来的计算值，重新计算，将dp[i]=num[i]时，当前i处为最大
- 当dp[i−1]>0时，当num[i]<0时，则dp[i]=num[i]+dp[i−1]导致dp[i]<dp[i−1]，但是否小于0，未知，放入i+1时考虑；当num[i]>=0时，dp[i]=num[i]+dp[i−1]。综合起来，dp[i]=num[i]+dp[i−1]。
- 两种情况，综合为dp[i] = num[i] + max(dp[i−1], 0)

最后max(dp)

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==1:
            return nums[0]
        dp = []
        dp.append(nums[0])
        for i in range(1, len(nums), 1):
            dp.append(nums[i] + max(dp[i-1],0))
        return max(dp)

nums=[-2,1,-3,4,-1,2,1,-5,4]
s = Solution()
print(s.maxSubArray(nums))

# 优化为
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==1:
            return nums[0]
        for i in range(1, len(nums), 1):
            nums[i] = nums[i] + max(dp[i-1],0)
        return max(nums)
```

### 乘积最大子数组

> 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

对于乘法，负数乘以负数，会变成正数，所以解题时需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。

动态方程:
$$
dp_{min}[i]=\min(dp_{max}[i-1] \times nums[i-1], nums[i], dp_{min}[i-1] \times nums[i-1]) \\
dp_{max}[i]=\max(dp_{max}[i-1] \times nums[i-1], nums[i], dp_{min}[i-1] \times nums[i-1])
$$
返回值max(dp_max)

```python
class Solution(object):
    def maxProduct(self, nums):
        if len(nums)==1:
            return nums[0]
        dp_min = []
        dp_max = []
        dp_min.append(nums[0])
        dp_max.append(nums[0])
        for i in range(1, len(nums), 1):
            dp_min.append(min(dp_min[i-1]*nums[i], dp_max[i-1]*nums[i], nums[i]))
            dp_max.append(max(dp_min[i-1]*nums[i], dp_max[i-1]*nums[i], nums[i]))
        return max(dp_max)
```



### 121-买卖股票的最佳时机

> 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
>
> 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
>
> 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

思路：动态规划

- 记录【今天之前买入的最小值】
- 计算【今天之前最小值买入，今天卖出的获利】，也即【今天卖出的最大获利】
- 比较【每天的最大获利】，取最大值即可

前$i$天的最大收益 = max(前$i-1$天的最大收益，当前价格-前$i-1$天的最低价)

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        import numpy as np
        if len(prices)<1:
            return 0
        
        get_max = 0
        min_p = prices[0]
        for p in prices[1:]:
            min_p = min(p, min_p)
            get_max = max(get_max, p - min_p)
        return get_max
```

**另一种思路：**(超时了)

以当前第i天为界，最大利润为第i天后的最高价-前i天的最低价，遍历最大利润，取最大。

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        import numpy as np
        if len(prices)<1:
            return 0
        
        get_max = 0
        for i in range(len(prices)):
            if i ==0:
                if get_max < max(prices[0:]) - prices[0]:
                    get_max = max(prices[1:]) - prices[0]
            else:
                if get_max < max(prices[i:]) - min(prices[:i]):
                    get_max = max(prices[i:]) - min(prices[:i])
            
        return get_max
```

### 5-最长回文子串

> 给你一个字符串 s，找到 s 中最长的回文子串。

思路：双指针+动态规划

- 布尔类型的dp\[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp\[i][j]为true，否则为false。
- 子串跨越长度: L

判断情况：

- 如果长度L=1，一定为回文，则dp\[i][j]=True
- 如果长度L=2，并且s[i]==s[j]，也是回文，则dp\[i][j]=True
- 如果长度L>2，则需要判断首首尾是否相同s[i]==s[j]，且中间子串也是回文dp\[i+1][j-1]=True，则dp\[i][j]=True

```python
class Solution(object):
    def longestPalindrome(self, s):
        n = len(s)
        if n<2:
            return s
        dp =[[False]*n for _ in range(n)] # 变量dp
        start, max_L = 0, 1 # 起始位与长度
        for right in range(n):
            for left in range(0, right+1):
                L = right - left + 1
                if L == 1:  # 情况1
                    dp[left][right] = True
                elif L ==2: # 情况2
                    dp[left][right] = s[left]==s[right]
                else:       # 情况3
                    dp[left][right] = dp[left+1][right-1] and s[left]==s[right]
                
                if dp[left][right]: # 更新最大长度及起始位
                    if L > max_L:
                        max_L = L
                        start = left
        return s[start:start+max_L]
```

思路：双指针+中心点扩散

确定回文串，即找中心往两边扩散看是不是对称，而中心点有两种情况：一个元素，或两个元素。

```python
class Solution(object):
    def longestPalindrome(self, s):
        def extend(i,j):
            while 0<=i and j< len(s) and s[i]==s[j]: # 扩散的条件
                i, j = i-1, j+1
            return i,j
        
        result = ''
        for i in range(len(s)):
            m, n = extend(i, i) # 一个元素扩散
            result = s[m+1:n] if n-m-1 > len(result) else result
            m, n = extend(i, i+1) # 两个元素扩散(如果两者相同)
            result = s[m+1:n] if n-m-1 > len(result) else result
        return result
```

### 509-斐波那契数

> 斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
>
> F(0) = 0，F(1) = 1
> F(n) = F(n - 1) + F(n - 2)，其中 n > 1
> 给你 n ，请计算 F(n) 。

```python
# 递归是暴力搜索，也即会出现重复搜索即fib(n-1)在下一轮中即为fib(n-2)，出现重复
class Solution(object):
    def fib(self, n):
        if n == 0 or n==1:
            return n
        else:
            return self.fib(n-1) + self.fib(n-2)

# 动态规划
class Solution(object):
    def fib(self, n):
        if n == 0 or n==1:
            return n
        dp = [0 for _ in range(n+1)]
        dp[0], dp[1] = 0, 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

### 1137-第 N 个泰波那契数

> 泰波那契序列 Tn 定义如下： 
>
> T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2
>
> 给你整数 n，请返回第 n 个泰波那契数 Tn 的值。
>
> ```python
> 输入：n = 4
> 输出：4
> 解释：
> T_3 = 0 + 1 + 1 = 2
> T_4 = 1 + 1 + 2 = 4
> ```

动态规划，同上：

```python
class Solution(object):
    def tribonacci(self, n):
        if n < 3:
            return n if n<2 else 1
        dp = [0 for _ in range(n+1)]
        dp[0], dp[1], dp[2] = 0, 1, 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
        return dp[n]
```



### 70- 爬楼梯

> 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
>
> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
>
> 注意：给定 n 是一个正整数。

分析：

- 如果第一次爬的是1个台阶，那么剩下n-1个台阶，爬法是f(n-1)
- 如果第一次爬的是2个台阶，那么剩下n-2个台阶，爬法是f(n-2)
- 可以得出总爬法为: f(n) = f(n-1) + f(n-2)
- 只有一个台阶时f(1) = 1，只有两个台阶的时候 f(2) = 2

即为变相的斐波那契数。

```python
class Solution(object):
    def climbStairs(self, n):
        if n < 2:
            return n
        dp = [0 for _ in range(n+1)]
        dp[1], dp[2] = 1, 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

### 746-使用最小花费爬楼梯

> 数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。
>
> 每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。
>
> 请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。
>
> 示例 1：
>
> 输入：cost = [10, 15, 20]
> 输出：15
> 解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15 。

如上分析，只是增加了cost：

- 如果第一次爬的是1个台阶，那么剩下n-1个台阶，爬法是f(n-1)
- 如果第一次爬的是2个台阶，那么剩下n-2个台阶，爬法是f(n-2)
- 加上损失，爬法为: f(n) = min(cost(n-1)+f(n-1)，cost(n-2) + f(n-2))
- 没有台阶时，自然为0，只有一个台阶时，不动f(1) = 0

```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        n = len(cost)
        if n<3:
            return min(cost)
        
        dp = [0 for _ in range(n+1)]
        for i in range(2, n+1):
            dp[i] = min(cost[i-1]+dp[i-1], cost[i-2]+dp[i-2])
        return dp[n]
```



## 岛屿问题

### 200-岛屿数量

目标是找到矩阵中 “岛屿的数量” ，上下左右相连的 1 都被认为是连续岛屿。

**深度优先dfs方法**： 设目前指针指向一个岛屿中的某一点 (i, j)，寻找包括此点的岛屿边界。

- 从 (i, j) 向此点的上下左右 (i+1,j),(i-1,j),(i,j+1),(i,j-1) 做深度搜索。
- **终止条件**：
  - (i, j) 越过矩阵边界;
  - grid[i][j] == 0，代表此分支已越过岛屿边界。
  - 搜索岛屿的同时，执行 grid[i][j] = '0'，即将岛屿所有节点删除，以免之后重复搜索相同岛屿。

**主循环：**

- 遍历整个矩阵，当遇到 grid[i][j] == '1' 时，从此点开始做深度优先搜索 dfs，岛屿数 count + 1 且在深度优先搜索中删除此岛屿。
- 最终返回岛屿数 count 即可。

```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """    
        count=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1': # 当有陆地时，深度搜索，直到满足边界，搜取完全后加1
                    self.dfs(grid, i, j)
                    count +=1
        return count
                        
    def dfs(self, grid, i, j):
        # 边界条件
        if not (0<=i<len(grid)) or not (0<=j<len(grid[0])) or grid[i][j]=='0': return
        # 将搜素过的标记为'0'
        grid[i][j]='0'
        # 向其他四个方向搜索
        self.dfs(grid, i-1, j)
        self.dfs(grid, i+1, j)
        self.dfs(grid, i, j-1)
        self.dfs(grid, i, j+1)

grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

s = Solution()
print(s.numIslands(grid))
```

**广度优先遍历 BFS**

主循环和思路一类似，不同点是在于搜索某岛屿边界的方法不同。

**bfs 方法：**

- 借用一个队列 queue，判断队列首部节点 (i, j) 是否未越界且为 1：
  - 若是则置零（删除岛屿节点），并将此节点上下左右节点 (i+1,j),(i-1,j),(i,j+1),(i,j-1) 加入队列；
  - 若不是则跳过此节点；
- 循环 pop 队列首节点，直到整个队列为空，此时已经遍历完此岛屿。

```python
class Solution:
    def numIslands(self, grid):
        def bfs(grid, i, j):
            queue = [[i, j]]
            while queue:
                [i, j] = queue.pop(0) # 弹出第一个
                # 判断是否满足加入队列的条件
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
                    grid[i][j] = '0'
                    queue += [[i + 1, j], [i - 1, j], [i, j - 1], [i, j + 1]] # 增加周边搜索
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1': 
                    bfs(grid, i, j)
                    count += 1
        return count
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

s = Solution()
print(s.numIslands(grid))
```

### 695-岛屿的最大面积

> 给定一个包含了一些 0 和 1 的非空二维数组 grid 。
>
> 一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
>
> 找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

在岛屿问题的基础上，添加计数功能。

```python
class Solution(object):
    def maxAreaOfIsland(self, grid):
        max_num=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==1: # 当有陆地时，深度搜索，直到满足边界，搜取完全后加1
                    max_num = max(self.dfs(grid, i, j), max_num)
        return max_num
                        
    def dfs(self, grid, i, j):
        # 边界条件
        if not (0<=i<len(grid)) or not (0<=j<len(grid[0])) or grid[i][j]==0: return 0
        # 将搜素过的标记为'0'
        grid[i][j]=0
        count = 1
        # 向其他四个方向搜索
        count += self.dfs(grid, i-1, j)
        count += self.dfs(grid, i+1, j)
        count += self.dfs(grid, i, j-1)
        count += self.dfs(grid, i, j+1)
        return count
```

## 二叉树

### 103-二叉树的锯齿形层序遍历

> 给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

首先需实现二叉树的层序遍历，后对层序遍历时的层数进行判断：

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def zigzagLevelOrder(self, root):
        def helper(node, level):
            if not node:
                return
            else:
                if level%2==1:
                    sol[level-1].append(node.val) # 类似于堆栈,从右侧推入
                else:
                    sol[level-1].insert(0, node.val) # 类似于堆栈,从左侧推入
                    
                if len(sol) == level:  # 遍历到新层时，只有最左边的结点使得等式成立
                    sol.append([])
                helper(node.left, level+1)
                helper(node.right, level+1)
        sol = [[]]
        helper(root, 1)
        return sol[:-1] 
        
if __name__ == '__main__':
    tree = TreeNode(0)
    tree.left = TreeNode(1)
    tree.right = TreeNode(2)
    tree.left.left = TreeNode(3)
    tree.left.right = TreeNode(4)
    tree.right.left = TreeNode(5)
    tree.right.right = TreeNode(6)
    s = Solution()
    print(s.levelOrder(tree)) 
```

### 102-二叉树的层序遍历

```python
class Solution(object):
    def levelOrder(self, root):
        def helper(node, level):
            if not node:
                return 
            else:
                result[level-1].append(node.val)
                if len(result)==level:
                    result.append([])
                helper(node.left, level+1)
                helper(node.right, level+1)
        result=[[]]
        helper(root, 1)
        return result[:-1]
```

### 94-二叉树的中序遍历

较为简单。

```python
class Solution(object):
    def inorderTraversal(self, root):
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

### 236-二叉树的最近公共祖先

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
>
> - 输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
> - 输出：3
> - 解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

![img](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

两个节点 p,q 分为两种情况：

- p 和 q 在相同子树中
- p 和 q 在不同子树中

从**根节点遍历**，递归向**左右子树**查询节点信息

**递归终止条件**：如果当前节点**为空**或等于 **p 或 q**，则返回**当前节点**

- 递归遍历左右子树，如果左右子树查到节点都不为空，则表明 p 和 q 分别在左右子树中，因此，当前节点即为最近公共祖先；
- 如果左右子树其中一个不为空，则返回非空节点。

```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if root==p or root==q or not root: return root

        left = self.lowestCommonAncestor(root.left, p, q) #遍历左子树
        right = self.lowestCommonAncestor(root.right, p, q) #遍历右子树
        
        if left and right: return root # 左右不空，返回当前节点
        return right if right else left # 代替如下几行
    
        # if not left and right: 
        #     return right
        # elif not right and left:
        #     return left
        # else:
        #     return None
```

### 105-从前序与中序遍历序列构造二叉树

> 根据一棵树的前序遍历与中序遍历构造二叉树。
>
> 注意:你可以假设树中没有重复的元素。
>
> 例如，给出
>
> ```
> 前序遍历 preorder = [3,9,20,15,7]
> 中序遍历 inorder = [9,3,15,20,7]
> ```
>
>
> 返回如下的二叉树：
>
>     	3
>     	/ \
>       9  20
>         /  \
>        15   7

前序遍历：根节点->左子树->右子树**（根->左->右）**，即（root.val）[左子树的结点的数值们，长度为left_len] [右子树的结点的数值们，长度为right_len]

中序遍历：左子树->根节点->右子树**（左->根->右）**，即[左子树的结点的数值们，长度为left_len] （root.val）[右子树的结点的数值们，长度为right_len]

进一步分析：

1. **前序**中左起第一位`1`肯定是根结点，我们可以据此找到**中序**中根结点的位置`rootin`；
2. **中序**中根结点**左边**就是**左子树**结点，右边就是**右子树**结点，即`[左子树结点，根结点，右子树结点]`，我们就可以得出左子树结点个数为`int left = rootin - leftin;`；
3. 前序中结点分布应该是：`[根结点，左子树结点，右子树结点]`；
4. 根据前一步确定的左子树个数，可以确定前序中左子树结点和右子树结点的范围；
5. 如果我们要前序遍历生成二叉树的话，下一层递归应该是：
   - 左子树：`root.left = pre_order(前序左子树范围，中序左子树范围);`
   - 右子树：`root.right = pre_order(前序右子树范围，中序右子树范围);`。
6. 每一层递归都要返回当前根结点`root`；

```python
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def buildTree(self, preorder, inorder):
        if not inorder:
            return None
        root = TreeNode(preorder[0]) # 根节点
        mid = inorder.index(preorder[0]) # 中序中的节点位置
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid]) # 递归左子树
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:]) # 递归右子树
        return root
```

### 106-从中序与后序遍历序列构造二叉树

> 根据一棵树的中序遍历与后序遍历构造二叉树。
>
> 注意:你可以假设树中没有重复的元素。
>
> 例如，给出
>
> ```
> 中序遍历 inorder = [9,3,15,20,7]
> 后序遍历 postorder = [9,15,7,20,3]
> ```
>
> 返回如下的二叉树：
>
>     	3
>     	/ \
>       9  20
>         /  \
>        15   7

中序遍历：左子树->根节点->右子树**（左->根->右）**，即[左子树的结点的数值们，长度为left_len] （root.val）[右子树的结点的数值们，长度为right_len]

后序遍历：左子树->右子树->根节点**（左->右->根）**，即[左子树的结点的数值们，长度为left_len] [右子树的结点的数值们，长度为right_len]（root.val）

进一步分析(类似上一题思路)：

1. **后序**中左起最后一位肯定是根结点，我们可以据此找到**中序**中根结点的位置`rootin`；
2. **中序**中根结点**左边**就是**左子树**结点，右边就是**右子树**结点，即`[左子树结点，根结点，右子树结点]`，我们就可以得出左子树结点个数为`int left = rootin - leftin;`；
3. 后序中结点分布应该是：`[左子树结点，右子树结点，根结点]`；
4. 根据前一步确定的左子树个数，可以确定后序中左子树结点和右子树结点的范围；
5. 如果我们要后序遍历生成二叉树的话，下一层递归应该是：
   - 左子树：`root.left = post_order(中序左子树范围, 后序左子树范围);`
   - 右子树：`root.right = post_order(中序右子树范围, 后序右子树范围);`。
6. 每一层递归都要返回当前根结点`root`；

```python
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def buildTree(self, inorder, postorder):
        if not postorder:
            return None
        root = TreeNode(postorder[-1])
        mid = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[:mid], postorder[:mid]) # mid是否加1，可以根据例子而得
        root.right = self.buildTree(inorder[mid+1:], postorder[mid:-1])
        return root
```

## 69-x 的平方根

> 实现 int sqrt(int x) 函数。
>
> 计算并返回 x 的平方根，其中 x 是非负整数。
>
> 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

二分法：

```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==1:
            return x
        
        left = 0
        right = x

        while (right - left) > 1:
            middle = int((left+right)/2)
            if x/middle < middle:
                right = middle
            else:
                left = middle
        return left
```

牛顿迭代法：
$$
x_{i+1}=\frac{x_i+\frac{C}{x_i}}{2}
$$


```python
class Solution(object):
    def mySqrt(self, x):
        num = x
        while abs(x - num * num) > 0.01:
            num = (num + x/num) / 2.0
        return int(num)
```

## 46-全排列（回溯算法）

> 给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

DFS方法：

```python
class Solution:
    def permute(self, nums):
        ret = []
        path = []

        def dfs(li):
            if len(li) == len(path):
                ret.append(path[:])
            for i in li:
                if i not in path:
                    path.append(i)
                    dfs(li)
                    path.pop()
        dfs(nums)
        return ret
```

回溯法的三个基本要素：

- 路径：已经做出的选择；
- 选择列表：当前可以做出的选择；
- 结束条件：结束一次回溯算法的条件，即遍历到决策树的叶节点；

回溯法解决问题的通用框架为：

```python
# 回溯算法，复杂度较高，因为回溯算法就是暴力穷举，遍历整颗决策树是不可避免的
结果 = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        结果.append(路径)
        return
    for 选择 in 选择列表:    # 核心代码段
        做出选择
        递归执行backtrack
        撤销选择
```

注意：对于回溯算法，不管怎么优化，回溯算法的时间复杂度都不可能低于O(N!)，因为回溯算法本质上是穷举，穷举整颗决策树是不可避免的，这也是回溯算法的缺点，复杂度很高。

主要的过程如下图：

<img src="https://pic.leetcode-cn.com/0bf18f9b86a2542d1f6aa8db6cc45475fce5aa329a07ca02a9357c2ead81eec1-image.png" alt="image.png" style="zoom:30%;" />

```python
class Solution:
    def permute(self, nums):
        ret = []
        path = []

        def backtrace(li):
            if not li: # 如果li为[]，说明没有元素了，则返回保存的所有path，将其加入结果中
                return ret.append(path[:])
            for i, item in enumerate(li):
                path.append(item) # 加入当前的
                backtrace(li[:i]+li[i+1:]) # 递归添加列表中除item之外的元素
                path.pop() # 撤销加入的item，因为与顺序有关，后面还得加入
        backtrace(nums)
        return ret
```

## 20-有效的括号

> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
>
> 有效字符串需满足：
>
> - 左括号必须用相同类型的右括号闭合。
> - 左括号必须以正确的顺序闭合。

可以推断出以下要点：

- 有效括号字符串的长度，一定是偶数！
- 右括号前面，必须是相对应的左括号，才能抵消！
- 右括号前面，不是对应的左括号，那么该字符串，一定不是有效的括号！

<img src="https://pic.leetcode-cn.com/467248403853f33e0dabd80c644893ad22aa6069f261bd6a4c4d62e3d7df2f8c-p1.png" alt="p1.png" style="zoom:50%;" />

思考过程是栈的实现过程。因此我们考虑使用栈，当遇到匹配的**最小括号对**时，我们将这对括号从栈中删除（即出栈），如果最后栈为空，那么它是有效的括号，反之不是。

<img src="https://pic.leetcode-cn.com/baa8829ac398e665eb645dca29eadd631e2b337e05022aa5a678e091471a4913-20.gif" alt="20.gif" style="zoom:50%;" />



```python
# 堆栈过程
class Solution:
    def isValid(self, s):
        dic = {')':'(', ']':'[', '}':'{'} # 根据右侧进行索引、判断
        stack = []
        for i in s:
            if stack and i in dic:
                if stack[-1]==dic[i]: 
                    stack.pop()
                else:
                    return False
            else:
                stack.append(i)
        return not stack

# 简化过程    
class Solution:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s: # 逐步替换，直到为空
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
```

## 数组（二分法）

### 33-搜索旋转排序数组

> 整数数组 nums 按升序排列，数组中的值 互不相同 。
>
> 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
>
> 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

进阶：一个时间复杂度为 O(log n)

```python
# 偷懒法
class Solution(object):
    def search(self, nums, target):
        if target not in nums:
            return -1
        else:
            return nums.index(target)
```

采用二分法（有序数组）

1. 判断 mid 与 left 对应元素的大小关系，从而判断哪边是有序的：选取 left（选 right 也可以），如果 `nums[mid] > nums[left]`，则左边有序；反之右边有序。

   <img src="https://pic.leetcode-cn.com/5e467ce8c9ba116700336ede3b97b659ef0a0141a6a1b4f8882fc8273bd07738.png" alt="img" style="zoom:75%;" />

2. 判断 mid 对应元素与 target 的大小关系。

   1. 假设左边是有序的，我们可以在有序的一边画一条线，这条线可以认为是柱状图的顶点：

      <img src="https://pic.leetcode-cn.com/fb6718b71c09ef5b5a8ea290f3221b48888f2a453b7f34fcc59009c512b8473e.png" alt="img" style="zoom:100%;" />

      如果 `target > nums[mid]`，target 只会存在于这种情况：

      <img src="https://pic.leetcode-cn.com/f3044cfa44c490d19202a4d5c9bfa40bc23b8db28e1023cf47595222e8a9e7c6.png" alt="img" style="zoom:80%;" />

      即 target 一定会存在于右半区间内，所以 `left = mid + 1`；

   2. 如果 `target < nums[mid]`，target 会存在以下两种情况：

      ![img](https://pic.leetcode-cn.com/3dc4d4ac72b1f971117c9d51a1401dd06b91997cdb02de0ded3f1e7d69db5aa0.png)

      target 小于 mid 对应元素的时候，可能只是比 mid 稍小一点，但是仍然比 left 大，这时 target 位于左半区间；

      或者，target 比 mid 对应的元素小很多，甚至比 left 对应的元素都小，这时 target 就位于右半区间了。


```python
class Solution(object):
    def search(self, nums, target):
        if not nums:
            return -1
        
        left = 0
        right = len(nums)-1
        while left <=right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            if nums[mid] <= nums[right]:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

### 4-寻找两个正序数组的中位数

> 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
>
> 输入：nums1 = [1,3], nums2 = [2]
> 输出：2.00000
> 解释：合并数组 = [1,2,3] ，中位数 2

取两个数组中的第k/2个元素进行比较，如果数组1的元素小于数组2的元素，则说明数组1中的前k/2个元素不可能成为第k个元素的候选，所以将数组1中的前k/2个元素去掉，组成新数组和数组2求第k-k/2小的元素，因为我们把前k/2个元素去掉了，所以相应的k值也应该减小。另外就是注意处理一些边界条件问题，比如某一个数组可能为空或者k为1的情况。

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        l1, l2 = len(nums1), len(nums2)
        left, right = (l1+l2+1)//2, (l1+l2+2)//2 # 找到两个数, 对奇偶数均适用
        return (self.findMaxK(nums1,nums2,left) + self.findMaxK(nums1,nums2,right))/2 # 取均值

    def findMaxK(self, nums1, nums2, k):
        len1, len2 = len(nums1), len(nums2)
        if len2==0:
            return nums1[k-1]
        if len1==0:
            return nums2[k-1]
        if k == 1:
            return min(nums1[0], nums2[0])

        i, j = min(k//2, len1)-1, min(k//2,len2)-1
        
        # 数组1的元素小于数组2的元素，说明数组1中的前k/2个元素不可能成为第k个元素的候选
        if nums1[i] <= nums2[j]: 
            # 将数组1中的前k/2个元素去掉，组成新数组和数组2求第k-k/2小的元素
            return self.findMaxK(nums1[i+1:], nums2, k-i-1)
        else:
            return self.findMaxK(nums1, nums2[j+1:], k-j-1)
```

### 88-合并两个有序数组（归并）

> 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
>
> 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。
>
> ```
> 输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
> 输出：[1,2,2,3,5,6]
> ```

归并排序中的合并部分程序：

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        res = []
        left, right = nums1[:m], nums2[:n]
        while len(left) > 0 and len(right) > 0:
            if left[0] <= right[0]:
                res.append(left.pop(0))
            else:
                res.append(right.pop(0))
        return res + left + right
```

双指针:

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        #i为nums1的m开始索引, j为nums2的n开始索引，idx为修改nums1的开始索引，然后从nums1末尾开始往前遍历
        i, j, idx = m-1, n-1, len(nums1)-1
        #按从大到小，从后往前修改nums1的值，每次都赋值为nums1和nums2的当前索引的较大值，然后移动索引
        while i >=0 and j >=0:
             #如果nums1[i] >= nums2[j]，则先赋值为nums1[i]，i索引减少
            if nums1[i] >= nums2[j]:
                nums1[idx] = nums1[i]
                i -= 1
            else:
                #如果nums1[i] <= nums2[j]，则先赋值为nums2[j]，j索引减少
                nums1[idx] = nums2[j]
                j -= 1
            #当前nums1修改索引减少1
            idx -=1
        # nums2没有遍历完n个，则继续遍历，直到n个完成
        while j>=0:
            nums1[idx] = nums2[j]
            j -= 1
            idx -= 1
        return nums1
```

### 51-数组中的逆序对（归并）

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
>
> ```
> 输入: [7,5,6,4]
> 输出: 5
> ```

主要思路是归并排序，归并排序的过程中会将左右两部分合并成一个有序的部分，当满足左侧元素大于右侧时，计算其逆序数即可，最后采用一个全局的变量，对其统计，最后返回即可。相当于对原数组排序后，同时返回一个逆序对的统计值。

```python
class Solution(object):
    def reversePairs(self, nums):
        if len(nums)<=1:
            return 0
        self.count = 0 # 全局统计值
        self.merge_sort(nums)
        return self.count
    
    def merge_sort(self, alist):
        if len(alist)==1:
            return alist
        mid = len(alist)//2
        left = self.merge_sort(alist[:mid]) # 递归进行分治与合并
        right = self.merge_sort(alist[mid:])
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []
        while len(left)>0 and len(right)>0:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
                self.count += len(left) # right[0]小于left[0]，那么right[0]小于整个left值，因为left是从小到大排序的，后面的值left[0:]均大于left[0],这些都是逆数对
        return result + left + right
```





## 48-旋转图像

> 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
>
> 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
>
> <img src="https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg" alt="img" style="zoom:50%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210524203127990.png" alt="image-20210524203127990" style="zoom:75%;" />

```python
class Solution(object):
    def rotate(self, matrix):
        n = len(matrix)
        matrix_new = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n-1-i]=matrix[i][j]
        matrix[:] = matrix_new
        return matrix
```

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210524203503608.png" alt="image-20210524203503608" style="zoom:80%;" />

```python
class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

## 矩形面积(单调栈)

### 84-柱状图中最大的矩形

> 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png)

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

**单调栈**：从栈底元素到栈顶元素呈单调递增或单调递减，栈内序列满足单调性的栈。

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531205123228.png" alt="image-20210531205123228" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531210835401.png" alt="image-20210531210835401" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531210853880.png" alt="image-20210531210853880" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531210918464.png" alt="image-20210531210918464" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531210950793.png" alt="image-20210531210950793" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531211011920.png" alt="image-20210531211011920" style="zoom:80%;" />

<img src="C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20210531211028679.png" alt="image-20210531211028679" style="zoom:80%;" />

```python
class Solution:
    def largestRectangleArea(self, heights):
        res = 0
        heights = [0] + heights +[0]
        size = len(heights)
        stack = [0]
        
        for i in range(1, size):
            while heights[i] < heights[stack[-1]]: # 如果右侧的值，小于栈中末尾的，则停止，计算矩形面积
                height = heights[stack.pop()] # 栈中末尾值
                width = i - stack[-1] -1 # 当前位置，减去其左侧的位置，取宽度
                res = max(res, height*width)
            stack.append(i)
        return res
       
heights = [2, 1, 5, 6, 2, 3]
s = Solution()
print(s.largestRectangleArea(heights))  
```

### 85-最大矩形

> 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
>
> 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
> 输出：6

<img src="https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg" alt="img" style="zoom:80%;" />

在上一题求**柱状图中最大的矩形**的基础上，对矩阵，逐行按照柱状图求解。

```python
class Solution:
    def maximalRectangle(self, matrix):
        if not matrix:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0]*n # 逐行对列统计个数
        res = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "0": # 下层出现为0，需要对原来的统计置0
                    heights[j] = 0
                else:
                    heights[j] +=1 # 统计柱子高度
            res = max(res, self.largestRectangleArea(heights))
        return res
    
    def largestRectangleArea(self, heights): # 柱状图中最大的矩形
        res = 0
        heights = [0] + heights + [0]
        size = len(heights)
        stack = [0]
        
        for i in range(1, size):
            while heights[i] < heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i -stack[-1]-1
                res = max(res, height*width)
            stack.append(i)
        return res
       
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
s = Solution()
print(s.maximalRectangle(matrix)) 
```

**另一解法**：二进制的&运算求矩形高度，>>运算来计算矩形宽度，然后每一行都遍历一遍，得出最大值。

```python
class Solution:
    def maximalRectangle(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        nums = [int(''.join(row), base=2) for row in matrix] #先将每一行变成2进制的数字
        ans, N = 0, len(nums)
        for i in range(N):#遍历每一行，求以这一行为第一行的最大矩形
            j, num = i, nums[i]
            while j < N: #依次与下面的行进行与运算。
                num = num & nums[j]  #num中为1的部分，说明上下两行该位置都是1，相当于求矩形的高，高度为j-i+1
                # print('num=',bin(num))
                if not num: #没有1说明没有涉及第i到第j行的竖直矩形
                    break
                width, curnum = 0, num
                while curnum: 
                    #将cursum与自己右移一位进行&操作。如果有两个1在一起，那么cursum才为1，相当于求矩形宽度
                    width += 1
                    curnum = curnum & (curnum >> 1)
                    # print('curnum',bin(curnum))
                ans = max(ans, width * (j-i+1))
                # print('i','j','width',i,j,width)
                # print('ans=',ans)
                j += 1
        return ans

matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
s = Solution()
print(s.maximalRectangle(matrix))
```

## 二维矩阵

### 74-搜索二维矩阵/240-搜索二维矩阵 II

> 编写一个高效的算法来判断 $m \times n$矩阵中，是否存在一个目标值。该矩阵具有如下特性：
>
> 每行中的整数从左到右按升序排列。
> 每行的第一个整数大于前一行的最后一个整数。
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2020/10/05/mat.jpg" alt="img" style="zoom:67%;" />
>
> 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
> 输出：true
>
> **240题**
>
> 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
>
> 每行的元素从左到右升序排列。
> 每列的元素从上到下升序排列。

**从左下角或右上角开始查找**

遍历的方式，是从左上或有右下角开始，但是并未使用矩阵有序的性质，时间复杂度为*O*(*M*∗*N*)。

如果我们从**右上角**开始遍历：

- 如果要搜索的 target 比当前元素大，那么让行增加；
- 如果要搜索的 target 比当前元素小，那么让列减小；

```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        m, n = len(matrix), len(matrix[0])
        i, j = 0, n-1
        while i < m and j >=0:
            if matrix[i][j] < target:
                i +=1
            elif matrix[i][j] > target:
                j -=1
            else:
                return True
        return False
```

**二分法**

既然是有序数组，那么可以考虑二分法，可将整个数组看成一个数列来处理。

```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        m, n = len(matrix), len(matrix[0])
        left, right = 0, n*m-1
        while left<=right:
            mid = left + (right - left)//2
            value = matrix[mid // n][mid % n]
            if value < target:
                left = mid + 1
            elif value > target:
                right = mid - 1
            else:
                return True
        return False
```

