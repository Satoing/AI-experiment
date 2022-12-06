def discount(start, test, end):
    # 使用曼哈顿距离
    gn = abs(start[0]-test[0]) + abs(start[1]-test[1])
    hn = abs(end[0]-test[0]) + abs(end[1]-test[1])
    return hn+gn

def priority(values):
    min_value = float('inf')
    index = -1
    for i in range(len(values)):
        if values[i] < min_value:
            min_value = values[i]
            index = i
    return index

# todo:应该打印从起点到终点，所以使用栈
def output(father, end, start):
    loc = end
    stack = [end]
    while(loc != start):
        loc = father[loc[0]][loc[1]]
        stack.append(loc)
    stack.reverse()
    print(stack)

# 这里的图是网格图，所以使用n×m的二维列表表示，1表示可以通行，0表示为障碍物
def AStar(graph, start: tuple, end: tuple):
    flag = False  # 表示最后是否找到了一条路径
    openSet = [[], []]  # 第一个列表记录坐标，第二个列表记录权值
    closeSet = []  # closeSet记录搜索过的网格坐标
    father = graph.copy()  # 和graph形状一样的列表，只是其中的元素是元组，记录父节点
    openSet[0].append(start);openSet[1].append(0)  # 初始化
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 相邻结点
    while len(openSet[0]) != 0:  # openSet如果为空则目标结点不可到达
        index = priority(openSet[1])
        current = openSet[0][index]
        openSet[0].pop(index);openSet[1].pop(index);
        print("DEBUG: 当前坐标", current, "\n-------------")
        if current != end:  # 还没有搜索到终点
            closeSet.append(current)
            for dir in dirs:
                test = tuple(map(sum, zip(current, dir)))  # 相邻结点
                print("DEBUG: test坐标", test)
                if(test[0]<0 or test[1]<0 or test[0]>=len(graph) or 
                   test[1]>=len(graph[0]) or graph[test[0]][test[1]]==0): continue  # 结点不可到达
                
                if test not in closeSet:
                    # 计算估价函数
                    fn = discount(start, test, end)
                    if test not in openSet[0]:
                        print("DEBUG: 不在openSet中")
                        # 以估价函数为权放入openSet，并记录父节点
                        openSet[0].append(test)
                        openSet[1].append(fn)
                        father[test[0]][test[1]] = current
                    else:  # 在openSet中
                        print("DEBUG: 在openSet中")
                        # 和原来的值作比较，选择小的那个，并替换父节点
                        index = openSet[0].index(test)
                        if openSet[1][index] > fn:  # 更新
                            openSet[1][index] = fn
                            father[test[0]][test[1]] = current
            print("DEBUG: openSet", openSet)
        else: flag = True;break  # 搜索到了终点
        input()  # 便于单步调试
    if flag: output(father, end, start)  # 输出这条路径
    else: print("找不到路径")

if __name__ == "__main__":
    graph = [[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1],[1, 1, 1, 1]]
    AStar(graph, (0, 0), (0, 3))