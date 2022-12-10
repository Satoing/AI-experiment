import math

class Solution(object):
    # 搜索过程中的一个状态，主要是要记录父状态才引入的
    class State():
        def __init__(self, cur: list, pre = None):
            self.cur = cur
            self.pre = pre  # 如果是C语言的话，这里应该使用State的指针形成链表
        def __repr__(self):
            return f"{self.cur}"
            
    # 初始化问题的参数
    def __init__(self, d: int, g: list, t: list, m = 1):
        # @dimension: 3代表八数码，4代表15数码，依次类推
        # @graph: 初始状态，用一维数组表示
        # @target: 目标状态，用一维数组表示
        # @method: 计算价值函数时选择的方法
        self.dimension = d
        self.graph = g
        self.target = t
        self.method = m
        self.path = []  # 记录搜索过的状态

    # 判断n数码问题的有解性
    def solvable(self):
        gi = Solution.inversion(self.graph)%2
        ti = Solution.inversion(self.target)%2
        # 维度是奇数，则只要初始和目标逆序数奇偶性相同即可
        if self.dimension%2 == 1:  
            if (gi ^ ti) == 1:
                return True
        else:  #维度是偶数，判断有解性要复杂一些
            # 奇偶性相同，那么0所在行的差值k必须是偶数时，才能互达
            if (gi ^ ti) == 1: 
                if abs(self.graph.index(0)%3-self.target.index(0)%3)%2 == 0:
                    return True
            else:  # 奇偶性不同，那么0所在行的差值k必须是奇数时，才能互达
                if abs(self.graph.index(0)%3-self.target.index(0)%3)%2 == 1:
                    return True
        return False
    
    # 计算启发值
    def heuristic(self, g): 
        # 使用曼哈顿距离
        if self.method == 1:
            return Solution.manhaton(g, self.target, self.dimension)
        # 使用欧氏距离
        if self.method == 2:
            return Solution.euclidean(g, self.target, self.dimension)
            
    # 启发式搜索
    def search(self):
        # 首先判断有解性
        if not self.solvable():
            print("该问题无解");return
        # 左上右下在一维数组上的表现 
        dirs = [-1, -self.dimension, 1, self.dimension]
        # 第一个列表记录状态，第二个列表记录权重，第三个列表记录g值
        openList = [[Solution.State(self.graph)], [self.heuristic(self.graph)], [0]]
        # 记录走过的状态（仅需要当前状态）
        closeList = []  
        count = 0
        
        while True:  
            # 从openList中取出优先级最高的状态
            index = Solution.priority(openList[1])
            depth = openList[2][index]
            print("DEBUG:当前状态的g值次数:", depth)
            current_state = openList[0].pop(index)
            openList[1].pop(index);openList[2].pop(index)
            current = current_state.cur
            closeList.append(current)
            self.path.append(current_state)
            # 如果搜索到目标状态，结束循环
            if current == self.target: break
            print("DEBUG:当前选择的状态:", current)
            gn = depth + 1
            
            # 考察该状态的四个方向
            zero = current.index(0)
            for dir in dirs:
                if zero+dir < 0 or zero+dir >= self.dimension**2: continue
                count += 1
                temp = Solution.swap(current.copy(), zero, zero+dir)
                # 如果该状态在closeList中，则忽略
                if temp in closeList: continue
                
                # 计算权值fn=gn+hn
                fn = gn + self.heuristic(temp)
                index = Solution.inObjList(openList[0], temp)
                # 如果该状态不在openList中，则加入
                if index == -1:
                    openList[0].append(Solution.State(temp, current_state))
                    openList[1].append(fn)
                    openList[2].append(gn)
                # 如果在openList中，则比较原值和新值
                elif openList[1][index] > fn:  # 更新权值和父状态
                    openList[1][index] = fn
                    openList[0][index].pre = current_state
                    openList[2][index] = gn
                    
        print(f"\n搜索成功！总共搜索了{count}个状态")
    
    # 使用栈输出结果
    def show(self):
        current = self.path[-1]
        stack = []
        while current is not None:
            stack.append(current.cur)
            current = current.pre
        length = len(stack)
        print(f"搜索路径的深度：{length-1}")
        for i in range(length):
            print(f"第{i}次搜索")
            temp = stack[length-i-1]
            for j in range(self.dimension):
                print(temp[j*self.dimension:j*self.dimension+self.dimension])
            print("---------")

    @staticmethod  # 计算逆序和
    def inversion(g):
        res = 0
        for i in range(len(g)):
            for j in range(i):
                if g[j] > g[i]: res += 1
        return res
    
    @staticmethod  # 曼哈顿距离
    def manhaton(g: list, t: list, d: int):
        res = 0
        for i in range(d**2):
            if g[i] == 0: continue
            j = t.index(g[i])
            row = abs(i//d - j//d)  # 行的差值
            col = abs(i%d - j%d)  # 列的差值
            res += (row+col)
        return res
    
    @staticmethod  # 欧式距离
    def euclidean(g: list, t: list, d: int):
        res = 0
        for i in range(d**2):
            if g[i] == 0: continue
            j = t.index(g[i])
            row = i//d - j//d  # 行的差值
            col = i%d - j%d  # 列的差值
            res += math.sqrt(row**2 + col**2)
        return res
    
    @staticmethod  # 判断元素t是否在对象列表的cur属性中
    def inObjList(openList: list, t):
        for i in range(len(openList)):
            if t == openList[i].cur: return i
        return -1
    
    @staticmethod  # 选择值最小的那个，相当于实现优先队列
    def priority(values):
        min_value = float('inf')
        index = -1
        for i in range(len(values)):
            if values[i] < min_value:
                min_value = values[i]
                index = i
        return index

    @staticmethod  # 以返回值的形式返回交换两个位置数字后的列表
    def swap(g, a, b):
        g[a], g[b] = g[b], g[a]
        return g
    
if __name__ == "__main__":
    # 8数码用例1
    # g = [2, 8, 3, 1, 6, 4, 7, 0, 5]
    # t = [1, 2, 3, 8, 0, 4, 7, 6, 5]
    # s = Solution(3, g, t, m = 1)
    
    # 8数码用例2
    # g = [1, 0, 2, 3, 4, 5, 6, 7, 8]
    # t = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    # s = Solution(3, g, t, m = 2)
    
    # 15数码用例1
    # g = [5, 1, 2, 4, 9, 6, 3, 8, 13, 15, 10, 11, 14, 0, 7, 12]
    # t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    # s = Solution(4, g, t, m = 1)
    
    g = [1, 2, 7, 3, 9, 5, 6, 4, 10, 11, 8, 0, 13, 14, 15, 12]
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    s = Solution(4, g, t, m = 1)
    
    # 15数码用例2，要跑很久
    # g = [11, 9, 4, 15, 1, 3, 0, 12, 7, 5, 8, 6, 13, 2, 10, 14]
    # t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    # s = Solution(4, g, t, 1)
    
    s.search()
    s.show()