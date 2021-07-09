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

### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

> 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
>
> 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
>
> 示例 1：
>
> 输入：nums = [2,0,2,1,1,0]
> 输出：[0,0,1,1,2,2]

```python
# 最简单为sort
class Solution(object):
    def sortColors(self, nums):
        return nums.sort()
```

双指针:

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        start, end, i = 0, len(nums)-1, 0
        while i <= end:
            if nums[i] == 0:	# 为0时，与前端交换
                nums[i], nums[start] = nums[start], 0
                start += 1
            elif nums[i] == 2:	# 为2时，与末端交换
                nums[i], nums[end] = nums[end], 2
                end -= 1
                i -= 1
            i += 1
        return nums
```





### [560-和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

> 给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。
>
> 示例 1 :
>
> 输入:nums = [1,1,1], k = 2
> 输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。

前缀和思想：求数列的和时，Sn = a1+a2+a3+...an; 此时Sn就是数列的前n项和。例S5 = a1 + a2 + a3 + a4 + a5; S2 = a1 + a2，即可以通过 S5-S2 得到 a3+a4+a5 的值。


```python
# 暴力搜索
class Solution(object):
    def subarraySum(self, nums, k):
        presum, res, n = [0 for _ in range(len(nums)+1)], 0, len(nums)
        # 前缀和
        for i in range(n):
            presum[i+1] = nums[i] + presum[i]
		# 计数
        for i in range(n+1):
            for j in range(i+1, n+1):
                if presum[j] - presum[i] == k:
                    res +=1
            
        return res

# 字典数存储，并优化
class Solution(object):
    def subarraySum(self, nums, k):
        Sum, res, cul = 0, 0, {}
        cul[0] = 1
        for i in range(len(nums)):
            Sum += nums[i]
            if Sum - k in cul: # 查找k的倍数，如果存在，加1,
                res += cul[Sum-k]
            if Sum not in cul: # 保存前缀和
                cul[Sum] = 0
            cul[Sum] += 1
        return res 
```

为什么我们只要查看是否含有 presum - k ，并获取到presum - k 出现的次数就行呢？见下图，所以我们完全可以通过 presum - k的个数获得 k 的个数

<img src="https://pic.leetcode-cn.com/1610773274-xuuVxS-file_1610773273681" alt="微信截图_20210115194113" style="zoom:50%;" />

### VIP-和等于 k 的最长子数组长度

> 给定一个数组 nums 和一个目标值 k，找到和等于 k 的最长子数组长度。如果不存在任意一个符合要求的子数组，则返回 0。
>
> 示例 1:
> 输入: nums = [1, -1, 5, -2, 3], k = 3
> 输出: 4

在上一题的基础上，存储索引即可。

```python
class Solution(object):
    def maxSubArrayLen(self, nums, k):
        Sum, cul, res = 0, {}, 0
        cul[0] = -1   # 余数为0时，索引为-1，以免第一次遇见可整除时，满足条件
        for i in range(len(nums)):
            Sum += nums[i]
            if Sum - k in cul: # 查找k的倍数，如果存在，加1,
                res = max(res, i - cul[Sum-k])
            if Sum not in cul: # 保存前缀和
                cul[Sum] = i	# 存储索引
        return res
```

### [974-和可被K整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/)

> 给定一个整数数组 A，返回其中元素之和可被 K 整除的（连续、非空）子数组的数目。
>
> 示例：
>
> 输入：A = [4,5,0,-2,-3,1], K = 5
> 输出：7
> 解释：
> 有 7 个子数组满足其元素之和可被 K = 5 整除：
> [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]

(presum[j+1] - presum[i] ) % k是满足条件，可变为presum[j +1] % k = presum[i] % k，余数相同则计算。

```python
class Solution(object):
    def subarraysDivByK(self, nums, k):
        Sum, res, cul = 0, 0, {}
        cul[0] = 1
        for i in range(len(nums)):
            Sum += nums[i] # 前缀和
            key = (Sum % k + k) % k # 获得整除的余数，加k的目的是，当被除数为负数时取模结果为负数，需要纠正
            if key in cul: # 如果存在，加1
                res += cul[key]
            else:
                cul[key] = 0
            
            cul[key] += 1
        return res 
```

### [523-连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)

> 给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：
>
> 子数组大小 至少为 2 ，且
> 子数组元素总和为 k 的倍数。
> 如果存在，返回 true ；否则，返回 false 。
>
> 如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。0 始终视为 k 的一个倍数。
>
> 示例 1：
>
> 输入：nums = [23,2,4,6,7], k = 6
> 输出：true
> 解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。

在上一题的基础上，需要对记录余数的索引，通过索引相减，以满足条件。

```python
class Solution(object):
    def checkSubarraySum(self, nums, k):
        Sum, cul = 0, {}
        cul[0] = -1   # 余数为0时，索引为-1，以免第一次遇见可整除时，满足条件
        for i in range(len(nums)):
            Sum += nums[i]
            key = (Sum % k + k) % k
            if key in cul: # 查找k的倍数，如果存在，加1,
                if  (i - cul[key]) > 1:
                    return True
                else:
                    continue  # 仅仅保存最小的索引，其他的跳过
            else:
                cul[key] = i

        return False
```

### [209-长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

> 给定一个含有 n 个正整数的数组和一个正整数 target 。
>
> 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
>
> 示例 1：
>
> 输入：target = 7, nums = [2,3,1,2,4,3]
> 输出：2
> 解释：子数组 [4,3] 是该条件下的长度最小的子数组。

仍然需要运用上面的前缀和，同时需要运用双指针，前后移动。时间复杂度O(n)。

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        if target > sum(nums):
            return 0
        # 返回长度，头指针，尾指针，列表长度
        res, head, tail, Sum, n = len(nums), 0, 0, 0, len(nums)
        while tail < n:
            # 先递增尾指针，直到值大于target
            while Sum < target and tail < n:
                Sum += nums[tail]
                tail += 1
            # 再递增头指针，以找到最短长度
            while Sum >= target and head >= 0:
                res = min(res, tail - head)
                Sum -= nums[head]
                head += 1
        return res
```

二分法，O(nlog(n))。

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        left, right, res = 0, len(nums), 0
        def helper(size):
            sum_size = 0
            for i in range(len(nums)):
                sum_size += nums[i]
                if i >= size:
                    sum_size -= nums[i-size]
                if sum_size >= target:
                    return True
            return False
        while left<=right:
            mid = (left+right)//2  # 滑动窗口大小
            if helper(mid):  # 如果这个大小的窗口可以那么就缩小
                res = mid
                right = mid-1
            else:  # 否则就增大窗口
                left = mid+1
        return res
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

**动态规划：**

因为是乘积的关系，nums[i] 数值的正负，与前面的状态值是有联系的，具体如下：

- 当 nums[i] > 0 时：
  - 与最大值的乘积依然是最大值
  - 与最小值的乘积依然是最小值
- 当 nums[i] < 0 时：
  - 与最大值的乘积变为最小值
  - 与最小值的乘积变为最大值
- 当 nums[i] = 0 时，这里无论最大最小值，最终结果都是 0，这里其实可以合并在上面任意一种情况。

```python
class Solution(object):
    def maxProduct(self, nums):
        mi = ma = res = nums[0]
        for i in range(1, len(nums)):
            if nums[i] < 0: # 元素值小于0，最大、最小值互换，因为与正数相乘后，最小变最大，最大变最小
                mi, ma = ma, mi
            ma = max(ma * nums[i], nums[i]) # 最大值更新
            mi = min(mi * nums[i], nums[i])
            res = max(res, ma)
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

### [581-最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

> 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
>
> 请你找出符合题意的 最短 子数组，并输出它的长度。
>
> 示例 1：
>
> 输入：nums = [2,6,4,8,10,9,15]
> 输出：5
> 解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。

从左到右循环，记录最大值为 max，若 nums[i] < max, 则表明位置 i 需要调整, 循环结束，记录需要调整的最大位置 i 为 high; 同理，从右到左循环，记录最小值为 min, 若 nums[i] > min, 则表明位置 i 需要调整，循环结束，记录需要调整的最小位置 i 为 low.

```python
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        max_, min_, high, low = nums[0], nums[-1], 0, len(nums)-1
        for i in range(len(nums)):
            max_ = max(nums[i], max_)
            min_ = min(min_, nums[len(nums)-1-i])

            if nums[i] < max_:
                high = i
            
            if nums[len(nums)-1-i] > min_:
                low = len(nums)-1-i

        return high - low +1 if high > low else 0
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
        window, res = [], []
        for i, num in enumerate(nums):
            # 窗口滑动时的规律, 即滑动的位置超出了windows范围
            if window and window[0] <= i-k:
                window.pop(0)
                
            # 把最大值左边的数小的清除，即判断新的数据Num,其与windows中值的比较：从最右端开始，Windows中的数据小于Num则弹出，否则停止，说明Windows中存的是从左到右，数值降低的序列。新的数据，需要与原有的序列进行判断
            while window and nums[window[-1]] <= num:
                window.pop()
                
            window.append(i) # # 队首一定是滑动窗口的最大值的索引
            
            if i >= k-1:
                res.append(nums[window[0]])
        return res

class Solution:
    def maxSlidingWindow(self, nums, k):
        deque = collections.deque()
        ans = []
        for i, num in enumerate(nums):
            if deque and deque[0] <= i-k: # 窗口滑动时的规律, 即滑动的位置超出了windows范围
                deque.popleft()
            while deque and nums[deque[-1]] < num:
                deque.pop()
            deque.append(i)
            if i >= k-1:
                ans.append(nums[deque[0]])
        return ans
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

### [92-反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

> 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg" alt="img" style="zoom:67%;" />
>
>
> 输入：head = [1,2,3,4,5], left = 2, right = 4

头插法：

<img src="https://pic.leetcode-cn.com/1616067008-ygfgJz-0092-reverse-linked-list-ii.gif" alt="0092-reverse-linked-list-ii.gif" style="zoom:80%;" />

根据上面的图示，这里说下其中涉及的参数，以及其中反转过程中的步骤：

其中涉及的参数：

- dummy_node：哑节点，减少判断；
- pre：指向left 的前一个节点，反转过程中不变；
- cur：初始指向需要反转区域的第一个节点，也就是left的位置；
- next：指向cur 的下一个节点，跟随cur 变化。

其中反转过程中的步骤：

- 将cur 的下一个节点指向next 的下一个节点；
- 将 next 的下一个节点指向pre 的下一个节点；
- 将 pre 的下一个节点指向next。

循环上面三个步骤，直至反转结束。

```python
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy = ListNode()
        dummy.next, pre = head, dummy
        # 令 pre 指向 left 位置的前一个节点
        for _ in range(left-1):
            pre = pre.next

        cur = pre.next
        # 通过头插法，实现反转
        for _ in range(left, right):
            nxt = cur.next
            cur.next = nxt.next
            nxt.next = pre.next
            pre.next = nxt
        
        return dummy.next
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

### [19-删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

> 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
>
> 进阶：你能尝试使用一趟扫描实现吗？
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg" alt="img" style="zoom:67%;" />
>
> 输入：head = [1,2,3,4,5], n = 2
> 输出：[1,2,3,5]

与上一题一样，快慢指针。

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next, fast, slow = head, head, dummy
        for _ in range(n):
            if fast:
                fast = fast.next
            else:
                return None

        while fast:
            slow, fast = slow.next, fast.next
        slow.next = slow.next.next
        return dummy.next
```

或者

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        fast, slow = head, head
        for _ in range(n):
            if fast.next:
                fast = fast.next
            else: 
                return head.next # 由于提前了一位, 正常为None

        while fast.next:
            slow, fast = slow.next, fast.next
        slow.next = slow.next.next
        return head
# 以下报错
class Solution(object):
    def removeNthFromEnd(self, head, n):
        fast, slow = head, head
        for _ in range(n):
            if fast:
                fast = fast.next
            else:
                return None

        while fast:  # 此时到达倒数第n个
            slow, fast = slow.next, fast.next
        # 此条件，第n个未删除，删除的是第n+1个，故如上采用fast.next，提前一个
        slow.next = slow.next.next
        return head
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

### [25-K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)(有点难)

> 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。k 是一个正整数，它的值小于或等于链表的长度。
>
> 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
>
> 进阶：
>
> 你可以设计一个只使用常数额外空间的算法来解决此问题吗？
> 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
>
> **示例 1：**
>
> ![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)
>
> ```
> 输入：head = [1,2,3,4,5], k = 2
> 输出：[2,1,4,3,5]
> ```

尾插法。

直接举个例子：`k = 3`。

```python
pre
tail    head
dummy    1     2     3     4     5
# 我们用tail 移到要翻转的部分最后一个元素
pre     head       tail
dummy    1     2     3     4     5
	       cur
# 我们尾插法的意思就是,依次把cur移到tail后面
pre          tail  head
dummy    2     3    1     4     5
	       cur
# 依次类推
pre     tail      head
dummy    3     2    1     4     5
		cur
....
```

```python
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = ListNode()
        dummy.next, pre, tail = head, dummy, dummy # 进行复制变量名
        
        while True:
            count = k
            while count and tail: # 移动k个
                count -= 1
                tail = tail.next
            
            if not tail: break # 如果是末尾，直接跳出
            head = pre.next # 重新复制变量名 [0,1,2,3,4,5]
            while pre.next != tail:
                cur = pre.next # 获取下一个元素 cur=[1,2,3,4,5], cur.val=1 | pre:[0,2,3,4,5], cur:[2,3,4,5],cur.val=2
                # pre与cur.next连接起来,此时cur(孤单)掉了出来
                pre.next = cur.next # pre:[0,2,3,4,5] | pre:[0,3,4,5]
                cur.next = tail.next # 和剩余的链表连接起来, tail:[3,4,5], cur:[1,4,5] | tail:[3,1,4,5], cur:[2,1,4,5]
                tail.next = cur #插在tail后面, tai:[3,1,4,5] | tai:[3,2,1,4,5]
            # 改变 pre tail 的值
            pre = head #[1,4,5]
            tail = head

        return dummy.next
```

**递归整体想法**

- 如果长度l小于k那直接返回；否则, 记链表由长度为k和$l_1 $的两个链表组成,$l=k+l_1$
- 翻转结果=【直接翻转前段长度为k的链表】 + 【k个一组翻转第二段长度为$l_1 $的链表】
- 前者迭代翻转(记得保留翻转前的头节点) 后者递归调用函数

```python
class Solution(object):
    def reverseKGroup(self, head, k):
        # 如果长度>k, 找第k+1个节点; 如果长度小于k, 中途返回head 
        h, i = head, 1
        for i in range(k): 
            if not h: return head
            h = h.next # 走k个节点
        # 翻转前k个节点组成的链表 保留头节点 以便连接后面的翻转链表
        tail_reverse = head
        pre, cur = None, head
        # 反转链表子程序
        while cur!=h: 
            temp=cur.next
            cur.next=pre
            pre, cur=cur, temp
            # 换成一行: cur.next, pre, cur = pre, cur, cur.next
        head_reverse=pre
        # 连接两端
        tail_reverse.next = self.reverseKGroup(h, k)
        return head_reverse
```

或者：

```python
class Solution:
    def reverseKGroup(self, head, k):
        cur = head
        count = 0
        while cur and count!= k:
            cur = cur.next
            count += 1
        if count == k:
            cur = self.reverseKGroup(cur, k)
            while count:
                head.next, cur, head= cur, head, head.next
                count -= 1
            head = cur   
        return head
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

**编辑距离**：

状态定义：dp\[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数

转移方程：

- 当 word1[i] == word2[j]，dp\[i][j] = dp\[i-1][j-1]；

- 当 word1[i] != word2[j]，dp\[i][j] = min(dp\[i-1][j-1], dp\[i-1][j], dp\[i][j-1]) + 1

**最长公共子序列：**

状态定义：

- dp[i][j] 表示text1[0\~i-1]和text2[0\~j-1]的最长公共子序列长度 
- dp\[0][0]等于0，等于dp数组总体往后挪了一个，免去了判断出界 

转移方程： 

- text1[i-1] == text2[j-1] 当前位置匹配上了: dp\[i][j] = dp[i-1]\[j-1]+1 
- text1[i-1] != text2[j-1] 当前位置没匹配上了 ：dp\[i][j] = max(dp\[i-1][j], dp\[i][j-1]); 
- base case: 任何一个字符串为0时都是零，初始化时候就完成了base case是赋值

**最长重复子数组：**

状态定义：

- dp[i][j] 表示text1[0\~i-1]和text2[0\~j-1]的最长公共子序列长度 
- dp\[0][0]等于0，等于dp数组总体往后挪了一个，免去了判断出界 

转移方程： 

- text1[i-1] == text2[j-1] 当前位置匹配上了: dp\[i][j] = dp[i-1]\[j-1]+1 
- text1[i-1] != text2[j-1] 当前位置没匹配上了 ：dp\[i][j] = 0，一旦为0，需要重新开始 
- base case: 任何一个字符串为0时都是零，初始化时候就完成了base case是赋值

**最长上升子序列：**

状态定义：dp[i] 为考虑前 i 个元素，以第 i个数字结尾的最长上升子序列的长度，注意nums[i] 必须被选取

转移方程：dp[i]=max(dp[j])+1,其中0≤j<i且num[j]<num[i]

**最大子序和：**

在计算子序和dp[i] 之前，我们已经计算出前i-1个num的子序和dp[i−1] 值，那么对于第i个数num[i]，要考虑原子序和dp[i−1]的值：

- 当dp[i−1]<=0时，不管num[i]为何值（>0，<0，=0），都要丢弃原来的计算值，重新计算，将dp[i]=num[i]时，当前i处为最大
- 当dp[i−1]>0时，当num[i]<0时，则dp[i]=num[i]+dp[i−1]导致dp[i]<dp[i−1]，但是否小于0，未知，放入i+1时考虑；当num[i]>=0时，dp[i]=num[i]+dp[i−1]。综合起来，dp[i]=num[i]+dp[i−1]。
- 两种情况，综合为dp[i] = num[i] + max(dp[i−1], 0)

**乘积最大子数组：**

最小的可变大，最大的可变小（正负数）。nums[i]为负时，存在最大最小互换。

动态方程:
$$
dp_{min}[i]=\min(dp_{max}[i-1] \times nums[i], nums[i], dp_{min}[i-1] \times nums[i]) \\
dp_{max}[i]=\max(dp_{max}[i-1] \times nums[i], nums[i], dp_{min}[i-1] \times nums[i])
$$
**买卖股票的最佳时机：**

前$i$天的最大收益 = max(前$i-1$天的最大收益，当前价格-前$i-1$天的最低价)

**最长回文子串：**

双指针+动态规划（双指针+中心点扩散）

- 布尔类型的dp\[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp\[i][j]为true，否则为false。
- 子串跨越长度: L

判断情况：

- 如果长度L=1，一定为回文，则dp\[i][j]=True
- 如果长度L=2，并且s[i]==s[j]，也是回文，则dp\[i][j]=True
- 如果长度L>2，则需要判断首首尾是否相同s[i]==s[j]，且中间子串也是回文dp\[i+1][j-1]=True，则dp\[i][j]=True

**斐波那契数/泰波那契数:**

不用递归，直接计算，用数组存储。递归存在重复计算，时间较慢。

- F(0) = 0，F(1) = 1，F(n) = F(n - 1) + F(n - 2)
- T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2

**爬楼梯：**（斐波那契数的变形）

- 如果第一次爬的是1个台阶，那么剩下n-1个台阶，爬法是f(n-1)
- 如果第一次爬的是2个台阶，那么剩下n-2个台阶，爬法是f(n-2)
- 可以得出总爬法为: f(n) = f(n-1) + f(n-2)
- 只有一个台阶时f(1) = 1，只有两个台阶的时候 f(2) = 2

**使用最小花费爬楼梯：**

增加了cost：

- 如果第一次爬的是1个台阶，那么剩下n-1个台阶，爬法是f(n-1)
- 如果第一次爬的是2个台阶，那么剩下n-2个台阶，爬法是f(n-2)
- 加上损失，爬法为: f(n) = min(cost(n-1)+f(n-1)，cost(n-2) + f(n-2))
- 没有台阶时，自然为0，只有一个台阶时，不动f(1) = 0

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

### 152-乘积最大子数组

> 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

对于乘法，负数乘以负数，会变成正数，所以解题时需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。

动态方程:
$$
dp_{min}[i]=\min(dp_{max}[i-1] \times nums[i], nums[i], dp_{min}[i-1] \times nums[i]) \\
dp_{max}[i]=\max(dp_{max}[i-1] \times nums[i], nums[i], dp_{min}[i-1] \times nums[i])
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

### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
>
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
>
> 问总共有多少条不同的路径？
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png" alt="img" style="zoom:80%;" />
>
>
> 输入：m = 3, n = 7
> 输出：28

机器人一定会走m+n-2步，即从m+n-2中挑出m-1步向下走，即C((m+n-2)，(m-1))。

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0]*(n) for _ in range(m)]
        for i in range(0, m):
            for j in range(0, n):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

### [64-最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

> 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
>
> 说明：每次只能向下或者向右移动一步。
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg" alt="img" style="zoom:67%;" />
>
>
> 输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
> 输出：7
> 解释：因为路径 1→3→1→1→1 的总和最小。

动态规划：dp\[i][j] 代表[0, 0]位置到[i, j]位置的最小值

$dp[i][i] =  \min(dp[i-1][j], dp[i][j-1]) + grid[i][j]$

```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        dp = [[0]*(n) for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

### [174-地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

> 一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。
>
> 骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。
>
> 有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。
>
> 为了尽快到达公主，骑士决定每次只向右或向下移动一步。
>
> 编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。
>
> 例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 右 -> 右 -> 下 -> 下，则**骑士的初始健康点数至少为 7**。
>
> -2 (K)	-3	3
> -5	     -10	1
> 10	    30	-5 (P)
>
>
> 说明:
>
> 骑士的健康点数没有上限。
>
> 任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。

最小耗费的生命值 **表示骑士救完公主刚刚好死了所需要用的生命值**，那么在终点时，已有的健康数-消耗的健康数=0，不死时，最少为1。如果我们要得到起始点的最小的耗费的生命值：

> 从其实起始点的**右边**、起始点的**下边**，两者之中选择一个耗费生命较小的，也就是二者之中的最优选择，然后算上本身这个格子所需要的血量，就是最小的耗费生命值了。

初始思路也是正序dp，此时dp[i][j]代表(0，0)到(i,j)的最低血量；但是实际递推时发现无法处理正健康值的网格，比如下一个网格是正健康值，则从起点到下一个网格的最低初始血量不变，但健康值发生了改变，dp数组无法记录与反映这一变化。

考虑倒序dp，此时dp[i][j]代表(i,j)到终点的最低血量，此时根据: 后一节点到达终点所需的最低血量(已知)、与当前节点的健康值(已知)倒推当前节点到达终点的最低血量(至少等于1):

$dp[i][j]+dungeon[i][j]-\min (dp[i+1][j], dp[i][j+1]) \ge 1$

即可改写为：

$dp[i][j] \ge 1 + \min (dp[i+1][j], dp[i][j+1])-dungeon[i][j]$    ->  $dp[i][j] = \max(1, \min (dp[i+1][j], dp[i][j+1])-dungeon[i][j])$

```python
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        m, n = len(dungeon), len(dungeon[0])
        dp = [[0]*n for _ in range(m)]
        # 终点初始条件
        dp[-1][-1] = -dungeon[-1][-1]+1 if dungeon[-1][-1]<=0 else 1

        # 初始化边界条件
        for i in range(m-2, -1, -1):
            dp[i][-1] = max(1, dp[i+1][-1] - dungeon[i][-1]) # 边界上，如果小于0，取1；大于0，取正值最大
        
        for j in range(n-2, -1, -1):
            dp[-1][j] = max(1, dp[-1][j+1] - dungeon[-1][j])

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]) # 转态转移公式
        return dp[0][0]
```

### [741-摘樱桃](https://leetcode-cn.com/problems/cherry-pickup/)

> 一个N x N的网格(grid) 代表了一块樱桃地，每个格子由以下三种数字的一种来表示：
>
> 0 表示这个格子是空的，所以你可以穿过它。
> 1 表示这个格子里装着一个樱桃，你可以摘到樱桃然后穿过它。
> -1 表示这个格子里有荆棘，挡着你的路。
> 你的任务是在遵守下列规则的情况下，尽可能的摘到最多樱桃：
>
> 从位置 (0, 0) 出发，最后到达 (N-1, N-1) ，只能向下或向右走，并且只能穿越有效的格子（即只可以穿过值为0或者1的格子）；
> 当到达 (N-1, N-1) 后，你要继续走，直到返回到 (0, 0) ，只能向上或向左走，并且只能穿越有效的格子；
> 当你经过一个格子且这个格子包含一个樱桃时，你将摘到樱桃并且这个格子会变成空的（值变为0）；
> 如果在 (0, 0) 和 (N-1, N-1) 之间不存在一条可经过的路径，则没有任何一个樱桃能被摘到。
> 示例 1:
>
> 输入: grid =
> [[0, 1, -1],
>  [1, 0, -1],
>  [1, 1,  1]]
> 输出: 5
> 解释： 
> 玩家从（0,0）点出发，经过了向下走，向下走，向右走，向右走，到达了点(2, 2)。
> 在这趟单程中，总共摘到了4颗樱桃，矩阵变成了[[0,1,-1],[0,0,-1],[0,0,0]]。
> 接着，这名玩家向左走，向上走，向上走，向左走，返回了起始点，又摘到了1颗樱桃。
> 在旅程中，总共摘到了5颗樱桃，这是可以摘到的最大值了。

思路的转换

- 这里要避免使用贪心法，比如保证单次路径最多的樱桃采摘，这样子对于两次结果反而不好的情况
- 既然一次不行，那么问题就变成按照两条路径一起找最优就可以

动态规划

- 定义
  - 这里要考虑降维，对于两条路径同时走，假设现在是 x1,y1 和 x2,y2, 当前步数是s,俺么满足 x1+y1=x2+y2=s,这里表示我们可以对于 x1,y1,x2,y2只要至少
  - 其中三个就可以退出另外一个
  - d[x1][y1][x2] 表示对于目前起点到最后 [N-1,N-1]的最多的樱桃数量

- 初始化
  - 因为我们计算是取最大值，默认可以把他们都设置为最小值 INT_MIN

- 计算
  这里要考虑多种情况
  - 如何避免重复计算，通过判断是否是初始值 INT_MIN 可以直接返回d的结果
  - 如何返回无法找到的路径，可以设置一个 负数表示无效，如 -1，而我们数值最小是0，那么结果就是0
  - 如果当前就是 N-1,N-1, 直接返回 grid[N-1][N-1] 的数值就是樱桃数量
  - 其他情况，我们会有四个选择（每条路径两种走法，所以是2*2=4），然后取最大结果
    - 选择1：路径1向右， 路径2向右
    - 选择2：路径1向右，路径2向下
    - 选择3：路径1向下， 路径2向右
    - 选择4：路径1向下，路径2向下
    - 计算下一格的数值需要判断去重，避免下一跳一样但是重复计算grid数值
  - 计算从顶到底的计算方式
- 结果
  - max(0, d[0][0][0]): 这里就是考虑可能找不到路径的情况 返回0

```python
class Solution:
    def cherryPickup(self, grid):
        n = len(grid)
        dp = [[[None for _ in range(n)] for _ in range(n)] for _ in range(n)]

        def dfs(r1, c1, c2) -> int:
            r2 = r1 + c1 - c2               #八皇后问题  左斜线 的特点 rc_sum为同一个值
            if r1 == n or r2 == n or c1 == n or c2 == n or grid[r1][c1] == -1 or grid[r2][c2] == -1:   #超范围了  或者是荆棘
                return float('-inf')
            elif r1 == c1 == n-1:           #到右下的终点了
                return grid[r1][c1]
            elif dp[r1][c1][c2]  is not None:   #访问过了  计算过了  就不用再算了 直接返回
                return dp[r1][c1][c2]
            else:
                res = grid[r1][c1]
                if c1 != c2:
                    res += grid[r2][c2]
                res += max(dfs(r1, c1+1, c2+1), dfs(r1, c1+1, c2), dfs(r1+1, c1, c2), dfs(r1+1, c1, c2+1))  #右右  右下  下下  下右
                dp[r1][c1][c2] = res
                return res
                
        return max(0, dfs(0, 0, 0))
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

### 694-不同岛屿的数量

> 给定一个非空01二维数组表示的网格，一个岛屿由四连通（上、下、左、右四个方向）的 1 组成，你可以认为网格的四周被海水包围。
>
> 请你计算这个网格中共有多少个**形状不同**的岛屿。
> 两个岛屿被认为是相同的，当且仅当一个岛屿可以通过平移变换（不可以旋转、翻转）和另一个岛屿重合。
>
> ```python
> 样例 1:
> 11000
> 11000
> 00011
> 00011
> 给定上图，返回结果 1。
> 
> 样例 2:
> 11011
> 10000
> 00001
> 11011
> 给定上图，返回结果 3。
> 
> 注意:
> 11
> 1
> 和
>  1
> 11
> 是不同的岛屿，因为我们不考虑旋转、翻转操作。
> 
> 注释 :  二维数组每维的大小都不会超过50。
> ```

考虑去重，一般都会考虑集合set()，那么只需要在原来的基础上，保留相对路径即可，最后再用set()去重，输出集合元素的个数即可。

```python
class Solution(object):
    def numDistinctIslands(self, grid):
        shape = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==1:
                    self.start = [i,j]	# 岛屿的起始点
                    self.path = []		# 保存岛屿的路径
                    self.dfs(grid, i, j)
                    shape.add(tuple(self.path))	# 将路径加入集合中
        return len(shape)
    
    def dfs(self, grid, i, j):
        if not (0<=i<len(grid)) or not (0<=j<len(grid[0])) or grid[i][j]==0: return 
        grid[i][j]=0
        self.path.append((i -self.start[0], j-self.start[1])) # 保存相对路径
        self.dfs(grid, i-1, j)
        self.dfs(grid, i+1, j)
        self.dfs(grid, i, j-1)
        self.dfs(grid, i, j+1)

grid = [
  [1, 1, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 1],
  [0, 1, 0, 1, 1, 1]
]
        
s = Solution()
print(s.numDistinctIslands(grid)) 
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

树的递归模板：

- 终止条件：什么时候递归到头了？此题自然是root为空的时候，空树当然是平衡的。
- 思考返回值，每一级递归应该向上返回什么信息？
- 单步操作应该怎么写？因为递归就是大量的调用自身的重复操作，因此从宏观上考虑，只用想想单步怎么写就行了，左树和右树应该看成一个整体，即此时树一共三个节点：root，root.left，root.right。

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

### [226-翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

> 翻转一棵二叉树。
>
> 示例：
>
> 输入：
>
>             4
>          /   \
>        2     7
>       / \   / \
>      1   3 6   9
>
> 输出：
>
>   	       4
>   	     /   \
>   	   7     2
>   	  / \   / \
>   	 9   6 3   1

```python
# 前序遍历
class Solution(object):
    def invertTree(self, root):
        if root is None: return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

class Solution(object):
    def invertTree(self, root):
        if root is None: return   	
        stack = [root]
        while stack:
            cur = stack.pop()
            cur.left, cur.right = cur.right, cur.left
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)            
        return root
    
# 中序遍历
class Solution(object):
    def invertTree(self, root):
        if root is None: return
        self.invertTree(root.left)
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        return root

class Solution(object):
    def invertTree(self, root):
        if root is None: return
        stack, cur = [], root
        while stack or cur:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                tmp = stack.pop()
                tmp.left, tmp.right = tmp.right, tmp.left
                cur = tmp.left
        return root
        
# 后序遍历
class Solution(object):
    def invertTree(self, root):
        if root is None: return
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root
    
class Solution(object):
    def invertTree(self, root):
        if root is None: return
        stack, mark_node, cur = [], None, root
        while stack or cur:
            if cur:
                stack.append(cur)
                cur = cur.left
            elif stack[-1].right != mark_node:
                cur = stack[-1].right
                mark_node = None
            else:
                mark_node = stack.pop()
                mark_node.left, mark_node.right = mark_node.right, mark_node.left
        return root

# 层序遍历
class Solution(object):
    def invertTree(self, root):
        def helper(node, level):
            if not node:
                return
            else:
                sol[level-1].append(node.val)
                node.left, node.right = node.right, node.left
                if len(sol) == level:  
                    sol.append([])
                helper(node.left, level+1)
                helper(node.right, level+1)
        sol = [[]]
        helper(root, 1)
        return root
    
class Solution(object):
    def invertTree(self, root):
        if root is None: return
        curr, queue = root, [root]
        while queue:
            curr = queue.pop(0) #弹出根节点
            curr.left, curr.right = curr.right, curr.left
            if curr.left: # 将左右节点加入队列中
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return root
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
>  	      3
>  	    / \
>  	   9  20
>  	     /  \
>  	    15   7

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
>  	      3
>  	    / \
>  	   9  20
>  	     /  \
>  	    15   7

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

### [104-二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。
>
> 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
>
> 说明: 叶子节点是指没有子节点的节点。
>
> 示例：
> 给定二叉树 [3,9,20,null,null,15,7]，
>
>  	      3
>  	    / \
>  	   9  20
>  	     /  \
>  	    15   7
> 返回它的最大深度 3 。

按照递归的模板写即可：

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0 # 到达根节点，返回0
        # 左右子树，取最大深度
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

由于递归中，存在重复计算，故采用标记法(旗鼓相当？)。

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.depth = 0
        self.dfs(root, 0) 
        return self.depth

    # 深度优先搜索，逐层标记
    def dfs(self, root, level):
        if not root: return
        if self.depth < level + 1:
            self.depth = level + 1
        
        self.dfs(root.left, level+1)
        self.dfs(root.right, level+1)
```

### [110-平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。
>
> 本题中，一棵高度平衡二叉树定义为：
>
> 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
>
> **示例 1：**
>
> <img src="https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg" alt="img" style="zoom:80%;" />
>
> ```
> 输入：root = [3,9,20,null,null,15,7]
> 输出：true
> ```

上一题的升级版···

**前序遍历**

root子树平衡 = root的左右深度差不超过1 且 left子树平衡 且 right子树平衡

```python
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True

        def depth(root):
            if not root: return 0
            return max(depth(root.left), depth(root.right))+1
        
        if abs(depth(root.left) - depth(root.right)) >1:
            return False
        if not self.isBalanced(root.left) or not self.isBalanced(root.right):
            return False
        return True
```

由于每一次都存在递归，存在重复，所以比较耗时。

**后序遍历：**

注意到在求根节点深度的过程中其实已经把每一个子节点的深度都求过了，因此我们可以采用后序遍历，先求左右子树的深度，一边计算深度，一边判断是否平衡。

定义helper(root)的含义为：当root为根节点的子树平衡时，helper(root)等于其深度，否则helper(root)=False

```python
# 后序递归，深度优先搜索
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True
        def helper(root):
            if not root: return 0
            left=helper(root.left)
            right=helper(root.right)
            # 左子树不平衡或者右子树不平衡
            if left is False or right is False: return False
            # 左右子树都平衡 此时left和right为左右子树的深度 判断root为根节点的子树是否平衡
            if abs(left-right)>1: return False 
            # 左右子树 和root为根节点的当前树 都平衡 返回root为根的树深度
            return max(left, right)+1
        return True if helper(root) else False
```

### [543-二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

> 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
>
> 示例 :
> 给定二叉树
>
>            1
>           / \
>          2   3
>         / \  
>        4   5   
>
>  返回 3, 它的长度是路径 [4->2->1->3] 或者 [5,2,1,3]。
>
> 注意：两结点之间的路径长度是以它们之间边的数目表示。

将二叉树的直径转换为：二叉树的每个节点的左右子树的**高度和**的最大值。主要还是递归、分治的思想。

```python
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.max = 0
        self.deep(root)
        return self.max

    def deep(self, root):
        if not root: return 0
        left, right = self.deep(root.left), self.deep(root.right) # 获取当前节点的左右分支的高度
        if left + right > self.max: # 如果和(左右分支可构成通路)大于当前值，替换
            self.max = left + right
        return max(left, right) + 1 # 返回当前节点的最大高度

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        left, right = self.deep(root.left), self.deep(root.right)
        return max(max(self.diameterOfBinaryTree(root.left), self.diameterOfBinaryTree(root.right)), left+right)

    def deep(self, root):
        if not root: return 0
        left, right = self.deep(root.left), self.deep(root.right)
        return max(left, right) + 1
```

### [124-二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

> 路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
>
> 路径和 是路径中各节点值的总和。
>
> 给你一个二叉树的根节点 root ，返回其 最大路径和 。
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg" alt="img" style="zoom:80%;" />
>
> 输入：root = [1,2,3]
> 输出：6
> 解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
> 示例 2：
>
> <img src="https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg" alt="img" style="zoom:67%;" />
>
> 输入：root = [-10,9,20,null,null,15,7]
> 输出：42
> 解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42

此题是上一题的变种，需要计算当前值。

```python
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.max = float("-inf")
        self.sum(root)
        return self.max
    
    def sum(self, root):
        if not root: return 0 # 叶子节点的左右支路为0
        left = max(0, self.sum(root.left)) 		# 如果左右分支，小于0，则丢弃
        right = max(0, self.sum(root.right))
        self.max = max(self.max, left+root.val+right) # 更新最大值
        return max(left, right) + root.val  	# 返回当前节点的所计算的值
```

### [687-最长同值路径](https://leetcode-cn.com/problems/longest-univalue-path/)

> 给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。
>
> 注意：两个节点之间的路径长度由它们之间的边数表示。
>
> 示例 1:
>
> 输入:
>
>               5
>              / \
>             4   5
>            / \   \
>           1   1   5
> 输出: 2
> 示例 2:
>
> 输入:
>
>               1
>              / \
>             4   5
>            / \   \
>           4   4   5
> 输出: 2

同样与上两题类似，但是增加约束条件是值相等，同时表示为边数，而不是长度。

```python
class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.max = 0
        self.value(root)
        return self.max

    def value(self, root):
        if not root.left and not root.right: return 0      		# 没有左右分支
        left = self.value(root.left)+1 if root.left else 0 		# 有左分支
        right = self.value(root.right)+1 if root.right else 0	# 有右分支
        
        if left != 0 and root.left.val != root.val: # 判断是否值相等，否则为0
            left = 0
        if right !=0 and root.right.val != root.val:
            right = 0
        self.max = max(self.max, left + right) 		# 左右相加，为边数
        return max(left, right)
```

```python
class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.max = 0
        self.value(root)
        return self.max

    def value(self, root):
        if not root: return 0
        left = self.value(root.left)
        right = self.value(root.right)
        L = R = 0
        if root.left and root.left.val == root.val:
            L = left + 1
        if root.right and root.right.val == root.val:
            R = right + 1
        self.max = max(self.max, L + R)
        return max(L, R) 
```

### [129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

> 给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
> 每条从根节点到叶节点的路径都代表一个数字：
>
> 例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
> 计算从根节点到叶节点生成的 所有数字之和 。
>
> 叶节点 是指没有子节点的节点。
>
> 示例 2：
>
> <img src="https://assets.leetcode.com/uploads/2021/02/19/num2tree.jpg" alt="img" style="zoom:80%;" />
>
> 输入：root = [4,9,0,5,1]
> 输出：1026
> 解释：
> 从根到叶子节点路径 4->9->5 代表数字 495
> 从根到叶子节点路径 4->9->1 代表数字 491
> 从根到叶子节点路径 4->0 代表数字 40
> 因此，数字总和 = 495 + 491 + 40 = 1026

主要是深度优先搜索，采用前向搜索：

```python
class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.res = 0
        self.dfs(root, 0)
        return self.res

    def dfs(self, root, tmp):
        if not root: return
        tmp = 10*tmp + root.val 	# tmp用于保存到当前的值
        if not root.left and not root.right:
            self.res += tmp
            return 
        self.dfs(root.left, tmp) 
        self.dfs(root.right, tmp)
```

### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

> 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。
>
> 叶子节点 是指没有子节点的节点。
>
> 示例 1：
>
> <img src="https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg" alt="img" style="zoom:67%;" />
>
> 输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
> 输出：true

与上一题类似：

```python
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        if not root: return False
        self.flag = False
        def dfs(root, tmp):
            if not root: return
            tmp += root.val
            if not root.left and not root.right:
                if tmp == targetSum:
                    self.flag = True
                    return
            dfs(root.left, tmp) 
            dfs(root.right, tmp)
        
        dfs(root, 0)
        return self.flag
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

## 排列-组合

### 46-全排列（回溯算法）

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

### [47-全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

> 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
>
> 示例 1：
>
> 输入：nums = [1,1,2]
> 输出：
> [[1,1,2],
>  [1,2,1],
>  [2,1,1]]

在上一题的基础上，去掉重复的。

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans, path = [], []
        def dfs(num):
            if not num and path not in ans:
                    ans.append(path[:])
            for i,it in enumerate(num):
                path.append(it)
                dfs(num[:i]+num[i+1:])
                path.pop()
        dfs(nums)
        return ans
```

采用集合结构去重，可提升效率.

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans, path = [], []
        def dfs(num):
            if not num:
                    ans.append(path[:])
            tmp = set()
            for i,it in enumerate(num):
                if it in tmp: # 集合元素去重
                    continue
                tmp.add(it)
                path.append(it)
                dfs(num[:i]+num[i+1:])
                path.pop()
        dfs(nums)
        return ans
```

### [31-下一个排列](https://leetcode-cn.com/problems/next-permutation/)

> 实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
>
> 如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
>
> 必须 原地 修改，只允许使用额外常数空间。
>
> 示例 1：
>
> 输入：nums = [1,2,3]
> 输出：[1,3,2]

从数组倒着查找，找到nums[i] 比nums[i+1]小的位置，就将nums[i]跟nums[i+1]到nums[nums.length - 1]当中找到一个最小的比nums[i]大的元素交换。交换后，再把nums[i+1]到nums[nums.length-1]排序。

- k，l=0，0
- loop：找到a<b的最大位置， 记录k,l， 并对换ab
- loop：位置k之后的序列反向
- 如果 k\==l\==0， 也就是说没有找到符合条件的ab：则整个数组反向

```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        N = len(nums)-1
        idx = 0
        L = 0
        for i in range(N, 0, -1):
            if nums[i-1]<nums[i]:
                idx=i-1
                break
        for j in range(N, idx, -1):
            if nums[j]>nums[idx]:
                L=j
                break
        if idx!=0 or L!=0:
            nums[idx], nums[L] = nums[L], nums[idx]
            p = nums[idx+1:]
            nums[idx+1:] = p[::-1]
        else:
            nums[:] = nums[::-1]

        return nums
```

```python
class Solution(object):
    def nextPermutation(self, nums):
        n, index, minindex =len(nums), n-1, -1    #-1是初始最小下标，因为不可能取到负的位置

        while index > 0 and nums[index] <= nums[index-1]:  #找出第一个升序的位置
                index -= 1
            
        if index == 0:    #整个数组为降序，直接翻转数组即可
            nums[::-1]    
        else:          #否则在nums[index:]中找出比nums[index-1]第一大的元素；
            for i in range(index,n):   
                if nums[i] > nums[index-1]:
                    if minindex == -1 or nums[minindex]>nums[i]:      
                        minindex = i

            nums[index-1], nums[minindex] = nums[minindex], nums[index-1]       #找出后，交换两人的位置再进行排序即可
            nums[index:] = sorted(nums[index:])
                    
        return nums

```

### [39-组合总和](https://leetcode-cn.com/problems/combination-sum/)

> 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
>
> candidates 中的数字可以无限制重复被选取。
>
> 说明：
>
> 所有数字（包括 target）都是正整数。
> 解集不能包含重复的组合。 
> 示例 1：
>
> 输入：candidates = [2,3,6,7], target = 7,
> 所求解集为：
> [
>   [7],
>   [2,2,3]
> ]

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        n, res = len(candidates), []

        def dfs(i, tmp_sum, tmp):
            for j in range(i, n):
                if tmp_sum + candidates[j] > target:
                    break
                if tmp_sum + candidates[j] == target:
                    res.append(tmp+[candidates[j]])
                    break
                dfs(j, tmp_sum + candidates[j], tmp+[candidates[j]])
        dfs(0, 0, [])
        return res
```

### [77-组合](https://leetcode-cn.com/problems/combinations/)

> 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
>
> 示例:
>
> 输入: n = 4, k = 2
> 输出:
> [
>   [2,4],
>   [3,4],
>   [2,3],
>   [1,2],
>   [1,3],
>   [1,4],
> ]

采用DFS进行

- 首先建立DFS函数，有两个参数，一个是从0开始一直加到n的整数，一个是暂存的列表。
- 先用一个if语句判断回溯结束条件，就是列表的长度是否等于k，如果是的话就添加进res中，然后结束递归函数。
- 如果列表长度小于k的话，就进入循环，将j增大1，且暂存列表加入数字j。
- 全部结束完输出结果ans即可。

```python
class Solution(object):
    def combine(self, n, k):
        ans = []
        def dfs(i, num):
            if len(num) == k:
                ans.append(num)
                return
            for j in range(i+1, n+1):
                dfs(j, num+[j])
        dfs(0, [])
        return ans
```

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        nums = [i for i in range(1,n+1)]

        # 明显用回溯法:
        res, path = [], []

        def dfs(num,i):
            if len(path)==k:
                res.append(path[:]) # 浅拷贝，这一步很重要
                return 

            for j in range(i,n):
                path.append(nums[j]) # 
                dfs(num[i:],j+1)
                path.pop()

        # 特殊情况处理
       if n==0 or k==0:
           return res

        dfs(nums,0)
        return res
```

### [78-子集](https://leetcode-cn.com/problems/subsets/)

> 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
>
> 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
>
> 示例 1：
>
> 输入：nums = [1,2,3]
> 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        path, ans, n = [], [], len(nums)
        def dfs(nums, index):
            ans.append(path[:])
            for i in range(index, n):
                path.append(nums[i])
                dfs(nums, i+1) # 递归
                path.pop() # 回溯
        dfs(nums, 0)
        return ans
```

每轮都传递一个数组起始指针的值，保证遍历顺序：

- 第一轮：先遍历以1 开头的所有子集，1→12→123 →13

- 第二轮：遍历以2开头的所有子集，2→23

- 第三轮:遍历以3开头的所有子集，3

- 这样三轮遍历保证能找到全部1开头，2开头，3开头的所有子集；同时，每轮遍历后又把上轮的头元素去掉，这样不会出现重复子集。（包括空集）

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(nums, index, path, res):
            res.append(path) # 添加路径
            for i in range(index, len(nums)):	# 逐个取单个元素
                dfs(nums, i+1, path+[nums[i]], res)  # path+[nums[i]]实现了元素的组合，并且从左往右依次取，不重复

        res = []
        dfs(nums, 0, [], res)
        return res
```

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []

        def dfs(start, path):
            if start == len(nums):
                ans.append(path[:])
            else:
                dfs(start + 1, path) # 不选
                path.append(nums[start])
                dfs(start + 1, path) # 选
                path.pop()
        dfs(0, [])
        return ans
```

### [90-子集 II](https://leetcode-cn.com/problems/subsets-ii/)

> 给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
>
> 解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。
>
> **示例 1：**
>
> ```
> 输入：nums = [1,2,2]
> 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
> ```

添加去重：

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums= sorted(nums)
        ans = []

        def dfs(start, path):
            if start == len(nums):
                if not path in ans:
                    ans.append(path[:])
            else:
                dfs(start + 1, path) # 不选
                path.append(nums[start])
                dfs(start + 1, path) # 选
                path.pop()
        dfs(0, [])
        return ans  
```

提高效率：

```python
class Solution:
    def subsetsWithDup(self, nums):
        # 去重需要排序, 用来存放符合条件结果, 存放符合条件结果的集合
        nums, path, ans, n = sorted(nums), [], [], len(nums)
        def dfs(nums, index):
            ans.append(path[:])
            for i in range(index, n):
                if nums[i]==nums[i-1] and i >index:
                    continue
                path.append(nums[i])
                dfs(nums, i+1) # 递归
                path.pop() # 回溯
        dfs(nums, 0)
        return ans
```

### [306-累加数](https://leetcode-cn.com/problems/additive-number/)

> 累加数是一个字符串，组成它的数字可以形成累加序列。
>
> 一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
>
> 给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。
>
> 说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。
>
> 示例 1:
>
> 输入: "112358"
> 输出: true 
> 解释: 累加序列为: 1, 1, 2, 3, 5, 8 。1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8

回溯加剪枝:

- 确认边界，当数组长度小于3，则说明无法组合成三个数，返回错误
- 回溯遍历，每次递归的参数有选中剩余的num数组，已符合要求的数字个数（主要为了判断是否有两个数字），前一个数和前第二个数
- 如果当前数字个数小于等于2且剩余数组为空，凑不够三个数则返回false
- 如果剩余数组为空说明所有数字已用完，返回True
- 循环剩余数组
  - 剪枝部分：单个的0可以作为一个数，而01，02之类的不能，直接返回
  - 剪枝部分：当res已经为True时，可以不用继续回溯了，直接返回
  - 循环递归：当数字个数大于等于2时，则判断是否等于前两个数之和，如果是则继续递归，如果不是则直接返回
  - 当数字个数不大于2时，则直接递归下一层

```python
class Solution:
    def isAdditiveNumber(self, num):
        # 回溯法DFS
        if len(num)<3:return False
        self.res = False
        def dfs(res_num, cur_count, first, second):
            # res_num:剩余的num数组
            # cur_count:当前已选过的num个数
            # first、second:前两个数
            if cur_count<=2 and not res_num:return # 因为至少要三个数才能做判断
            if not res_num:
                self.res =True
                return

            for i in range(len(res_num)):
                # 剪枝--单个的0可以作为一个数，而01，02之类的不能，直接返回
                if len(res_num[:i+1])!=len(str(int(res_num[:i+1]))):
                    return
                if cur_count>=2:
                    if int(res_num[:i+1])==first+second:
                        dfs(res_num[i+1:], cur_count+1, second, int(res_num[:i+1]))
                else:
                    dfs(res_num[i+1:], cur_count+1, second, int(res_num[:i+1]))
                if self.res:    # 剪枝--当res已经为True时，可以不用继续回溯了，直接返回
                    return
        dfs(num,0,0,0)
        return self.res
```

首先遍历前两个数，确定之后判断后面的是不是累加的就可以了。

```python
class Solution:
    def isAdditiveNumber(self, num):
        n = len(num)
        if n < 3:
            return False
        
        def check(p1, p2, j):
            while j < n:
                p = str(int(p1) + int(p2))
                if num[j: j+len(p)] != p:
                    return False
                j += len(p)
                p1, p2 = p2, p
            return True

        for i in range(1, n//2+1) if num[0] != "0" else [1]:	# 第一个数，逐长度判断
            for j in range(i+1, n) if num[i] != "0" else [i+1]:	# 第二个数，也逐长度判断
                p1 = num[:i]
                p2 = num[i:j]
                if check(p1, p2, j):
                    return True

        return False
```

### [842-将数组拆分成斐波那契序列](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/)

> 给定一个数字字符串 S，比如 S = "123456579"，我们可以将它分成斐波那契式的序列 [123, 456, 579]。
>
> 形式上，斐波那契式序列是一个非负整数列表 F，且满足：
>
> 0 <= F[i] <= 2^31 - 1，（也就是说，每个整数都符合 32 位有符号整数类型）；
> F.length >= 3；
> 对于所有的0 <= i < F.length - 2，都有 F[i] + F[i+1] = F[i+2] 成立。
> 另外，请注意，将字符串拆分成小块时，每个块的数字一定不要以零开头，除非这个块是数字 0 本身。
>
> 返回从 S 拆分出来的任意一组斐波那契式的序列块，如果不能拆分则返回 []。
>
> 示例 1：
>
> 输入："123456579"
> 输出：[123,456,579]
> 示例 2：
>
> 输入: "11235813"
> 输出: [1,1,2,3,5,8,13]

模板，再多带个f(n) + f(n+1) == f(n+2)的判断条件

```python
class Solution:
    def splitIntoFibonacci(self, S):
        def backtrack(cur, temp_state):
            if len(temp_state) >= 3 and cur == n:  # 退出条件
                self.res = temp_state
                return
            if cur == n:  # 剪枝
                return
            for i in range(cur, n):
                if S[cur] == "0" and i > cur:  # 当数字以0开头时,应该跳过
                    return
                if int(S[cur: i+1]) > 2 ** 31 - 1 or int(S[cur: i+1]) < 0:  # 剪枝
                    continue
                if len(temp_state) < 2:
                    backtrack(i+1, temp_state + [int(S[cur: i+1])])
                else:
                    if int(S[cur: i+1]) == temp_state[-1] + temp_state[-2]:
                        backtrack(i+1, temp_state + [int(S[cur: i+1])])

        n = len(S)
        self.res = []
        backtrack(0, [])
        return self.res
```

## 字符串

### 20-有效的括号

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

### [394-字符串解码](https://leetcode-cn.com/problems/decode-string/)

> 给定一个经过编码的字符串，返回它解码后的字符串。
>
> 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
>
> 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
>
> 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
>
> **示例 3：**
>
> ```
> 输入：s = "2[abc]3[cd]ef"
> 输出："abcabccdcdcdef"
> ```

**栈：先进后出，依次遍历完成整个字符串**

```python
class Solution(object):
    def decodeString(self, s):
        stack, num, ans = [], 0, ''
        for c in s:
            if c.isdigit():				# 计算值
                num = num*10 + int(c)
            elif c == '[':				# 标志位,保存新的字符串
                stack.append((ans, num))
                num, ans = 0, ''
            elif c == ']':				# 结束位，对[]之间的值计算
                pre, n = stack.pop()
                ans = pre + ans * n
            else:
                ans += c
        return ans
```

### [8-字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

> 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
>
> 函数 myAtoi(string s) 的算法如下：
>
> 读入字符串并丢弃无用的前导空格
> 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
> 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
> 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
> 如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
> 返回整数作为最终结果。
> 注意：
>
> 本题中的空白字符只包括空格字符 ' ' 。
> 除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
>
>
> 示例 1：
>
> 输入：s = "42"
> 输出：42
> 解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
> 第 1 步："42"（当前没有读入字符，因为没有前导空格）
>          ^
> 第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
>          ^
> 第 3 步："42"（读入 "42"）
>            ^
> 解析得到整数 42 。
> 由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
> 示例 2：
>
> 输入：s = "   -42"
> 输出：-42
> 解释：
> 第 1 步："   -42"（读入前导空格，但忽视掉）
>             ^
> 第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
>              ^
> 第 3 步："   -42"（读入 "42"）
>                ^
> 解析得到整数 -42 。
> 由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。
> 示例 3：
>
> 输入：s = "4193 with words"
> 输出：4193
> 解释：
> 第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
>          ^
> 第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
>          ^
> 第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
>              ^
> 解析得到整数 4193 。
> 由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。
> 示例 4：
>
> 输入：s = "words and 987"
> 输出：0
> 解释：
> 第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
>          ^
> 第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
>          ^
> 第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
>          ^
> 解析得到整数 0 ，因为没有读入任何数字。
> 由于 0 在范围 [-231, 231 - 1] 内，最终结果为 0 。
> 示例 5：
>
> 输入：s = "-91283472332"
> 输出：-2147483648
> 解释：
> 第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
>          ^
> 第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
>           ^
> 第 3 步："-91283472332"（读入 "91283472332"）
>                      ^
> 解析得到整数 -91283472332 。
> 由于 -91283472332 小于范围 [-231, 231 - 1] 的下界，最终结果被截断为 -231 = -2147483648 。

直接按照示例写程序，同时利用上一题的思路。

```python
class Solution(object):
    def myAtoi(self, s):
        if not s: return 0
        while s[0] == ' ': # 去掉空格
            s = s[1:]
            if not s: return 0
            
        num, flag = 0, 0
        if s[0] == '-' or s[0] == '+': # 判断正负号
            flag = 0 if s[0] == '+' else 1
            s = s[1:]
        elif not s[0].isdigit():
            return 0
        
        for c in s: # 获取数值
            if c.isdigit():				# 计算值
                num = num*10 + int(c)
            else:
                break
        # 范围限制
        ans = -num if flag else num
        if -2**31 <= ans <= 2**31 - 1:
            return ans
        else:
            return -2**31 if flag else 2**31 - 1
```

### [409-最长回文串](https://leetcode-cn.com/problems/longest-palindrome/)

> 给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
>
> 在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
>
> 注意:
> 假设字符串的长度不会超过 1010。
>
> 示例 1:
>
> 输入:
> "abccccdd"
>
> 输出:
> 7
>
> 解释:
> 我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

````python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic, ans = {}, 0
        for ch in s:
            if ch not in dic:
                dic[ch] = 1
            else:
                dic[ch] += 1
            if dic[ch] == 2:
                ans  += 2
                dic[ch] = 0
        return ans if ans == len(s) else ans + 1 # 长度不等，说明是奇数个，放中间，加1
````

### [670-最大交换](https://leetcode-cn.com/problems/maximum-swap/)

> 定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。
>
> 示例 1 :
>
> 输入: 2736
> 输出: 7236
> 解释: 交换数字2和数字7。
> 示例 2 :
>
> 输入: 9973
> 输出: 9973
> 解释: 不需要交换。

将数字转换为字符，再进行操作

```python
class Solution:
    def maximumSwap(self, num):
        nums = [int(x) for x in str(num)] # 转化为单个字符
        num_sorted = sorted(nums, reverse=True) # 进行排序，从大到小

        value = -1
        for i in range(len(nums)):
            if nums[i] != num_sorted[i]:  # 如果排序后的列表与原来不一致，则说明替换了，找到第一个不同的位置
                value = num_sorted[i]
                index = i
                break

        if value == -1: # 如果没有，不需要排序，返回即可
            return num
        else:
            for j in range(len(nums)-1, -1, -1):  # 反向找到替换的值，并进行交换
                if nums[j] == value:
                    nums[j], nums[index] = nums[index], nums[j]
                    break
        return int(''.join(str(it) for it in nums)) # 将字符串变回整数
```

```python
class Solution:
    def maximumSwap(self, num):
        nlist = [n for n in str(num)]
        num_len = len(nlist)
        m = max(nlist)
        for i, c in enumerate(nlist):
            m = i
            for j in range(i+1, num_len):
                if nlist[j] >= nlist[m]:
                    # 注意相等情况, 取最后一个位置, 保证 1993 => 9913, 而不是 9193 ;
                    m = j
            if m != i and nlist[i] != nlist[m]:
                # 注意此处找到的最大值和当前位置相等, 则不交换!!!, 比如 98368 => 98863
                nlist[i], nlist[m] = nlist[m], nlist[i]
                break
        return int("".join(nlist))
```

```python
class Solution(object):
    def maximumSwap(self, num):
        if num <= 10:
            return num
        
        # 首位数最大，则不需要移动首位数
        # 首位数不是最大，与最后一个最大数交换位置
        def helper(num_list):
            if not num_list:
                return []
            max_n = max(num_list)
            if max_n == num_list[0]:
                return [max_n] + helper(num_list[1:])
                
            i = 0
            for idx, n in enumerate(num_list):
                if max_n == n:
                    i = idx
            num_list[0], num_list[i] = num_list[i], num_list[0]
            return num_list
        
        return int(''.join(helper(list(str(num)))))
```

## 数组（二分法查找）

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

### [81-搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

> 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
>
> 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
>
> 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
>
> 示例 1：
>
> 输入：nums = [2,5,6,0,0,1,2], target = 0
> 输出：true
> 示例 2：
>
> 输入：nums = [2,5,6,0,0,1,2], target = 3
> 输出：false

与上一题的区别在于，这题的数组中可能会出现重复元素。二分查找的本质就是在循环的每一步中考虑排除掉哪些元素，本题在用二分查找时，只有在`nums[mid]`严格大于或小于左边界时才能判断它左边或右边是升序的，这时可以再根据`nums[mid], target`与左右边界的大小关系排除掉一半的元素；当`nums[mid]`等于左边界时，无法判断是`mid`的左边还是右边是升序数组，而只能肯定左边界不等于`target`（因为`nums[mid] != target`），所以只能排除掉这一个元素，让左边界加一。

```python
class Solution(object):
    def search(self, nums, target):
        if not nums: return False
        L, R = 0, len(nums)-1
        while L <= R:
            mid = (L+R)//2
            if nums[mid] == target:
                return True
            elif nums[L] < nums[mid]:
                if nums[L] <= target < nums[mid]:
                    R = mid -1
                else:
                    L = mid + 1
            elif nums[mid] < nums[L]:
                if nums[mid] < target <= nums[R]:
                    L = mid + 1
                else:
                    R = mid - 1
            elif nums[L]==nums[mid]:
                L += 1
        return False
```

### [153-寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

> 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
> 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
> 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
> 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
>
> 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
>
> 示例 1：
>
> 输入：nums = [3,4,5,1,2]
> 输出：1
> 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
>

```python
class Solution(object):
    def findMin(self, nums):
        if not nums: return None
        L, R = 0, len(nums) - 1
        while L <= R:
            mid = (L + R) // 2
            if nums[mid] < nums[R]: # mid可能为最小值
                R = mid
            else:   # mid肯定不是最小值
                L = mid + 1
        return nums[mid]
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

### [376-摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

> 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
>
> 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。
>
> 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
> 子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
>
> 给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。
>
> ```python
> 示例 1：
> 输入：nums = [1,7,4,9,2,5]
> 输出：6
> 解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
> 
> 示例 2：
> 输入：nums = [1,17,5,10,13,15,10,5,16,8]
> 输出：7
> 解释：这个序列包含几个长度为 7 摆动序列。
> 其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
> ```

主要采用贪心算法，

用示例二来举例，如图所示：

<img src="https://pic.leetcode-cn.com/1625284975-CVAnlk-file_1625284970854" alt="376.摆动序列" style="zoom:50%;" />

局部最优：删除单调坡度上的节点（不包括单调坡度两端的节点），那么这个坡度就可以有两个局部峰值。

整体最优：整个序列有最多的局部峰值，从而达到最长摆动序列。

实际操作上，其实连删除的操作都不用做，因为题目要求的是最长摆动子序列的长度，所以只需要**统计数组的峰值数量**就可以了（相当于是删除单一坡度上的节点，然后统计长度）

这就是贪心所贪的地方，让峰值尽可能的保持峰值，然后删除单一坡度上的节点。

针对序列[2,5]，可以假设为[2,2,5]，这样它就有坡度了即preDiff = 0，如图：

<img src="https://pic.leetcode-cn.com/1625284975-quOlUE-file_1625284970751" alt="376.摆动序列1" style="zoom:67%;" />

针对以上情形，result初始为1（默认最右面有一个峰值），此时curDiff > 0 && preDiff <= 0，那么result++（计算了左面的峰值），最后得到的result就是2（峰值个数为2即摆动序列长度为2）

```python
class Solution:
    def wiggleMaxLength(self, nums):
        preC, curC, res = 0, 0, 1  #题目里nums长度大于等于1，当长度为1时，其实到不了for循环里去，所以不用考虑nums长度
        for i in range(len(nums) - 1):
            curC = nums[i + 1] - nums[i]
            if curC * preC <= 0 and curC !=0:  #差值为0时，不算摆动
                res += 1
                preC = curC  #如果当前差值和上一个差值为一正一负时，才需要用当前差值替代上一个差值
        return res
```

利用摆动序列，波峰和波谷的差值最多为1的特点。一次遍历，常数空间。

```python
class Solution:
    def wiggleMaxLength(self, nums):
        up ,down = 1,1
        if len(nums)<2:return len(nums)
        for i in range(1,len(nums)):
            if nums[i]>nums[i-1]:
                up = down+1
            if nums[i]<nums[i-1]:
                down = up+1
        return max(up,down)
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

### [498-对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)

> 给定一个含有 M x N 个元素的矩阵（M 行，N 列），请以对角线遍历的顺序返回这个矩阵中的所有元素，对角线遍历如下图所示。
>
> 示例:
>
> 输入:
> [
>  [ 1, 2, 3 ],
>  [ 4, 5, 6 ],
>  [ 7, 8, 9 ]
> ]
>
> 输出:  [1,2,4,7,5,3,6,8,9]
>
> 解释:
>
> <img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/diagonal_traverse.png" alt="img" style="zoom:50%;" />

思路：

```python
'''
每层的索引和相等：
1. 假设矩阵无限大；
2. 索引和为{偶}数，向上遍历，{横}索引值递减，遍历值依次是(x,0),(x-1,1),(x-2,2),...,(0,x)
3. 索引和为{奇}数，向下遍历，{纵}索引值递减，遍历值依次是(0,y),(1,y-1),(2,y-2),...,(y,0)

   每层的索引和:
            0:              (00)
            1:            (01)(10)
            2:          (20)(11)(02)
            3:        (03)(12)(21)(30)
            4:      (40)(31)(22)(13)(04)
            5:    (05)(14)(23)(32)(41)(50)
            6:  (60)(51)................(06)

        按照“层次”遍历，依次append在索引边界内的值即可
'''
```

```python
class Solution(object):
    def findDiagonalOrder(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[int]
        """
        n, m = len(mat) - 1 , len(mat[0]) - 1 # 横轴 索引最大值n, 纵轴 索引最大值m
        level = n + m + 1 					  # 层数等于横纵最大索引之和 + 1
        res = []
        for x in range(level): # 层数值
            #索引和为{偶}数，向上遍历，{横}索引值递减，遍历值依次是(x,0),(x-1,1),(x-2,2),...,(0,x)，不要索引出界的，即可
            if x%2==0:  
                for i in range(x, -1, -1):
                    j = x - i
                    if i <=n and j <=m:
                        res.append(mat[i][j])
                    elif j > m: # j递增，超出纵轴索引
                        break
            #索引和为{奇}数，向下遍历，{纵}索引值递减，遍历值依次是(0,y),(1,y-1),(2,y-2),...,(y,0)，不要索引出界的，即可
            else:
                for j in range(x, -1, -1):
                    i = x - j
                    if i <=n and j <=m:
                        res.append(mat[i][j])
                    elif i > n: # i递增，超出横轴索引
                        break
        return res
```

简化代码：

```python
'''
以3*4为例
0:              
(00)

1:            
(01)
(10)

2:          
(20)
(11)
(02)

3:        
(03)
(12)
(21)

4:      
(22)
(13)

5:    
(23)
'''

class Solution(object):
    def findDiagonalOrder(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[int]
        """
        m, n, res = len(mat), len(mat[0]), []
        level = n + m + 1 # 层数
        for i in range(level):
            #左上三角：层数i小于纵轴时，范围是从0开始，到横轴最大值m
            #右下三角：层数i大于纵轴时，范围是从i+1-n开始，到横轴最大值m
            tmp = [mat[j][i-j] for j in range(max(0, i+1-n), min(i+1, m))]
            res += tmp if i%2 else tmp[::-1] # 奇数层则反向排序
        return res
```

### 311-稀疏矩阵的乘法

> <img src="https://img-blog.csdnimg.cn/20200927091414387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3ODIxNzAx,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:75%;" />

暴力计算：$C[i][k] += A[i][j]*B[j][k]$

```python
class Solution:
    def multiply(self, A, B) -> List[List[int]]:
        size1, size2 = len(A), len(B[0])  # A的行数 和 B的列数
        size3 = len(A[0])  # A的列数和B的行数相等
        ans = [[0] * size2 for _ in range(size1)]
        for i1 in range(size1):
            for i2 in range(size2):
                for i3 in range(size3):
                    ans[i1][i2] += A[i1][i3] * B[i3][i2]
        return ans
```

利用hashmap来储存非0元素。这应该是面试中比较标准的解法。一个encode来把sparse_matrx转换成dense_matrix。然后对dense_matrix做乘法，最后用decode把dense_matrix的结果转换为sparse_matrix的结果

```python
class Solution:
    def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        def encode(sparse_matrix):
            dense_matrix = {}
            for i in range(len(sparse_matrix)):
                for j in range(len(sparse_matrix[0])):
                    if sparse_matrix[i][j]:
                        dense_matrix[(i,j)] = sparse_matrix[i][j]
            return dense_matrix
        
        def decode(dense_matrix,row,col):
            sparse_matrix = [[0]*col for _ in range(row)]
            for (i,j),val in dense_matrix.items():
                sparse_matrix[i][j] = val
            
            return sparse_matrix
        
        A_dense = encode(A)
        B_dense = encode(B)
        ans_dense = collections.defaultdict(int)
        
        for (i,j) in A_dense.keys():
            for k in range(len(B[0])):
                if (j,k) in B_dense:
                    ans_dense[(i,k)] += A_dense[(i,j)]*B_dense[(j,k)]
                    
        return decode(ans_dense,len(A),len(B[0]))
```

## 丑数-质数

### [263-丑数](https://leetcode-cn.com/problems/ugly-number/)

> 给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。
>
> 丑数 就是只包含质因数 2、3 和/或 5 的正整数。
>
> 示例 1：
>
> 输入：n = 6
> 输出：true
> 解释：6 = 2 × 3

思路：对能被2,3,5整除的数不断除2,3,5，最后剩1就是，剩0就不是

```python
class Solution(object):
    def isUgly(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n < 1: return False
        while n % 2 ==0:
            n = n/2
        while n % 3 ==0:
            n = n/3
        while n % 5 ==0:
            n = n/5

        return n==1
```

### [264-丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

> 给你一个整数 n ，请你找出并返回第 n 个 丑数 。
>
> 丑数 就是只包含质因数 2、3 和/或 5 的正整数。
>
> 示例 1：
>
> 输入：n = 10
> 输出：12
> 解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
> 示例 2：
>
> 输入：n = 1
> 输出：1
> 解释：1 通常被视为丑数。

思路：由于是找出丑数序列中的第n个，因而需要生成该序列。采用穷举法，对每个数进行判断，必定会超出时间，因而另寻思路。

要生成第 n 个丑数，我们必须从第一个丑数 1 开始，向后逐渐的寻找。丑数只包含 2， 3，5 三个因子，所以生成方式就是在已经生成的丑数集合中乘以 [2, 3, 5] 而得到新的丑数。

现在的问题是在已经生成的丑数集合中，用哪个数字乘以 2？ 用哪个数字乘以 3？用哪个数字乘以 5？

很显然的一个结论：**用还没乘过 2 的最小丑数乘以 2；用还没乘过 3 的最小丑数乘以 3；用还没乘过 5 的最小丑数乘以 5。然后在得到的数字中取最小，就是新的丑数。**

实现的方法是用**动态规划**(三个指针)：

- 我们需要定义 3 个指针idx2, idx3,idx5 分别表示丑数集合中还没乘过 2，3，5 的丑数位置。
- 然后每次新的丑数 dp[i] = min(dp[idx2] * 2, dp[idx3] * 3, dp[idx5] * 5) 。
- 然后根据 dp[i] 是由 idx2, idx3,idx5 中的哪个相乘得到的，对应的把此 index + 1，表示还没乘过该 index 的最小丑数变大了。

```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        res=[1]
        id2 = id3 = id5 = 0
        for i in range(n-1):
            m = min(res[id2]*2, res[id3]*3, res[id5]*5) # 找到其中的最小数
            res.append(m) # 排序插入
            if m == res[id2]*2: # 判断是哪个质因数，其个数加1
                id2 +=1
            if m == res[id3]*3:
                id3 +=1
            if m == res[id5]*5:
                id5 +=1
        return res[n-1] # 返回值
```

### [313-超级丑数](https://leetcode-cn.com/problems/super-ugly-number/)

> 编写一段程序来查找第 n 个超级丑数。超级丑数是指其所有质因数都是长度为 k 的质数列表 primes 中的正整数。
>
> 示例:
>
> 输入: n = 12, primes = [2,7,13,19]
> 输出: 32 
> 解释: 给定长度为 4 的质数列表 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32] 。

此题是上一题的升级版，即质因数是变化的，那么用一个数组来存储其指针即可。

```python
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        res, k = [1], len(primes) # k是质因数的个数
        
        dp = [0 for _ in range(k)] # 质因数指针数组
        for i in range(n-1):
            m = min([res[dp[j]]*primes[j] for j in range(k)]) # 查询最小值
            res.append(m) # 排序插入
            for j in range(k):
                if m == res[dp[j]]*primes[j]: # 判断是哪个质因数，其个数加1
                    dp[j] += 1
            
        return res[n-1]
```

### [204-计数质数](https://leetcode-cn.com/problems/count-primes/)

> 统计所有小于非负整数 n 的质数的数量。
>
> 示例 1：
>
> 输入：n = 10
> 输出：4
> 解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。

**埃氏筛选法**，关键思想是：当找到一个质数，则**将其所有整数倍的数全部标记为合数**。

解释：对于质数x，其整数倍2x、3x、4x、...都是合数，可以被x整除；对于合数y，一定存在一个比y小的质数x可以整除y。

遍历时可以优化：

- 第一层循环不用遍历到n-1，只需遍历到sqrt(n)即可。
- 第二层循环不用从i*2开始遍历，从$i^2$开始遍历即可，因为$i^2$之前的i的整数倍已经被其他更小的质数标记过了。

```python
class Solution:
    def countPrimes(self, n):
        is_primes = [1 for _ in range(2, n)]
        for i in range(2, int(n**0.5)+1):
            if is_primes[i-2]:
                for j in range(i*i, n, i):
                    is_primes[j-2] = 0
        return sum(is_primes)
```

## 课程表

### [207-课程表](https://leetcode-cn.com/problems/course-schedule/)

> 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
>
> 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
>
> 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
> 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
>
> ```python
> 输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
> 输出：false
> 解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
> ```

正常思路，超时：

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        class_dict = {item[0]:item[1] for item in prerequisites}
        for item in prerequisites:
            tmp = []
            class_ = item[1]
            tmp.append(item[0])
            while class_ in class_dict.keys():
                tmp.append(class_)
                if class_dict[class_] in tmp:
                    return False
                else:
                    class_ = class_dict[class_]
        return True
```

考察**拓扑排序：入度表BFS法 / DFS法**

- **本题可约化为：** 课程安排图是否是 **有向无环图(DAG)**。拓扑排序原理： 对 DAG 的顶点进行排序，使得对每一条有向边 (u, v)，均有 u（在排序记录中）比 v 先出现。亦可理解为对某点 v 而言，只有当 v 的所有源点均出现了，v 才能出现。
- 通过课程前置条件列表 prerequisites 可以得到课程安排图的 **邻接表 adjacency**，以降低算法时间复杂度，以下两种方法都会用到邻接表。

**方法一：入度表（广度优先遍历）**

算法流程：

- 统计课程安排图中每个节点的入度，生成 入度表 indegrees。
- 借助一个队列 queue，将所有入度为 0 的节点入队。
- 当 queue 非空时，依次将队首节点出队，在课程安排图中删除此节点 pre：
  - 并不是真正从邻接表中删除此节点 pre，而是将此节点对应所有邻接节点 cur 的入度 -1，即 indegrees[cur] -= 1。
  - 当入度 -1后邻接节点 cur 的入度为 0，说明 cur 所有的前驱节点已经被 “删除”，此时将 cur 入队。
- 在每次 pre 出队时，执行 numCourses--；
  - 若整个课程安排图是有向无环图（即可以安排），则所有节点一定都入队并出队过，即完成拓扑排序。换个角度说，若课程安排图中存在环，一定有节点的入度始终不为 0。
  - 因此，拓扑排序出队次数等于课程个数，返回 numCourses == 0 判断课程是否可以成功安排。

复杂度分析：

- 时间复杂度 O(N + M)： 遍历一个图需要访问所有节点和所有临边，N 和 M 分别为节点数量和临边数量；
- 空间复杂度 O(N + M)： 为建立邻接表所需额外空间，adjacency 长度为 N ，并存储 M 条临边的数据。

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        indegrees = [0 for _ in range(numCourses)]
        adjacency = [[] for _ in range(numCourses)]
        queue = deque()

        for cur, pre in prerequisites:
            indegrees[cur] += 1
            adjacency[pre].append(cur) # 每门课需要先修的课程
            
        for i in range(len(indegrees)): # 将所有入度为 0的节点入队
            if not indegrees[i]: queue.append(i)
        # BFS TopSort.
        while queue: # 当 queue 非空时，依次将队首节点出队，在课程安排图中删除此节点 pre
            pre = queue.popleft()
            numCourses -= 1
            for cur in adjacency[pre]: # 需要学习的课程
                indegrees[cur] -= 1
                if not indegrees[cur]: queue.append(cur) # 如果入度为0，重新入栈
        return not numCourses
```

**方法二：通过 DFS 判断图中是否有环**。

算法流程：

- 借助一个标志列表 flags，用于判断每个节点 i （课程）的状态：
  - 未被 DFS 访问：i == 0；
  - 已被其他节点启动的 DFS 访问：i == -1；
  - 已被当前节点启动的 DFS 访问：i == 1。
- 对 numCourses 个节点依次执行 DFS，判断每个节点起步 DFS 是否存在环，若存在环直接返回 False。DFS 流程；
  - **终止条件**：
    - 当 flag[i] == -1，说明当前访问节点已被其他节点启动的 DFS 访问，无需再重复搜索，直接返回 True。
    - 当 flag[i] == 1，说明在本轮 DFS 搜索中节点 i 被第 2 次访问，即 课程安排图有环 ，直接返回 False。
  - 将当前访问节点 i 对应 flag[i] 置 1，即标记其被本轮 DFS 访问过；
  - 递归访问当前节点 i 的所有邻接节点 j，当发现环直接返回 False；
  - 当前节点所有邻接节点已被遍历，并没有发现环，则将当前节点 flag 置为 -1并返回 True。
- 若整个图 DFS 结束并未发现环，返回 True。

复杂度分析：

- 时间复杂度 O(N + M)： 遍历一个图需要访问所有节点和所有临边，N 和 M 分别为节点数量和临边数量；
- 空间复杂度 O(N + M)： 为建立邻接表所需额外空间，adjacency 长度为 N，并存储 M 条临边的数据。

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        def dfs(i, adjacency, flags):
            if flags[i] == -1: return True  # 已被其他节点启动的 DFS 访问
            if flags[i] == 1: return False  # 已被当前节点启动的 DFS 访问，存在循环
            flags[i] = 1                    # 当前访问标志
            for j in adjacency[i]:          # 对逐门课的依次判断
                if not dfs(j, adjacency, flags): return False
            flags[i] = -1                   # 设置访问完标志
            return True
            
        adjacency = [[] for _ in range(numCourses)] # 邻接矩阵，其index表示哪门课，值表示要先修哪些课程
        flags = [0 for _ in range(numCourses)] # 所有课程访问的标志
        for cur, pre in prerequisites:
            adjacency[pre].append(cur) # 每门课需要先修的课程
        for i in range(numCourses):
            if not dfs(i, adjacency, flags): return False # 逐门课判断是否有循环
        return True
```

### [210-课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

> 现在你总共有 n 门课需要选，记为 0 到 n-1。
>
> 在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
>
> 给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
>
> 可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。
>
> 示例 1:
>
> 输入: 2, [[1,0]] 
> 输出: [0,1]
> 解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

在上一题的BFS基础上，添加几行用于保存课程即可。

```python
from collections import deque

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        indegrees = [0 for _ in range(numCourses)] # 入度
        adjacency = [[] for _ in range(numCourses)] #邻接矩阵
        queue = deque() 

        for cur, pre in prerequisites:
            indegrees[cur] += 1
            adjacency[pre].append(cur) # 每门课需要先修的课程
            
        for i in range(len(indegrees)): # 将所有入度为 00 的节点入队
            if not indegrees[i]: queue.append(i)
        # BFS TopSort.
        res = []
        while queue: # 当 queue 非空时，依次将队首节点出队，在课程安排图中删除此节点 pre
            pre = queue.popleft()
            res.append(pre) # 保存课程
            numCourses -= 1
            for cur in adjacency[pre]: # 需要学习的课程
                indegrees[cur] -= 1
                if not indegrees[cur]: queue.append(cur) # 如果入度为0，重新入栈
        # 拓扑排序出队次数等于课程个数(即numCourses == 0)，返回排序，否则[]
        return [] if numCourses else res
```

### [630-课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/)

> 这里有 n 门不同的在线课程，他们按从 1 到 n 编号。每一门课程有一定的持续上课时间（课程时间）t 以及关闭时间第 d 天。一门课要持续学习 t 天直到第 d 天时要完成，你将会从第 1 天开始。
>
> 给出 n 个在线课程用 (t, d) 对表示。你的任务是找出最多可以修几门课。
>
> 示例：
>
> 输入: [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
> 输出: 3
> 解释: 
> 这里一共有 4 门课程, 但是你最多可以修 3 门:
> 首先, 修第一门课时, 它要耗费 100 天，你会在第 100 天完成, 在第 101 天准备下门课。
> 第二, 修第三门课时, 它会耗费 1000 天，所以你将在第 1100 天的时候完成它, 以及在第 1101 天开始准备下门课程。
> 第三, 修第二门课时, 它会耗时 200 天，所以你将会在第 1300 天时完成它。
> 第四门课现在不能修，因为你将会在第 3300 天完成它，这已经超出了关闭日期。

1.每门课程的开始时间是没有要求的

2.按照deadline将课程从小到大排序

3.当前时刻能学这门课程的话，就学。如果学不了了，超过了deadline，必须扔一门课。那就扔耗时最长的那门课。time（当前时刻）可以回退

```python
class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        courses.sort(key=lambda x:(x[1],x[0])) # #按照deadline排序
        q , d = [] , 0
        for c in courses:
            d += c[0]
            heapq.heappush(q,-c[0]) #将-c[0]压入堆q中
            if d > c[1] : d += heapq.heappop(q)  #选择的课程序列中，必须弹出一个。就弹那个耗时最长的(从堆中弹出最小的元素)
        return len(q)
```

