import numpy as np
import pandas as pd
from math import log2

# 将样本中离散的数据离散化，使用中位数二分
def discretization_train(X: pd.DataFrame, cut = 2):
    changes = []
    for col in X:
        counter = X[col].value_counts()
        # 如果值的个数小于4，那么就认为这个特征是离散的，不需要进行离散化
        if len(counter) <= 4: continue
        cuts = [-float('inf')]
        for i in range(1, cut): 
            cuts.append(round(X[col].quantile(i/cut)))
        cuts.append(float('inf'))
        for i in range(len(X[col])):
            for j in range(len(cuts)):
                if(X[col][i]<=cuts[j]):
                    X[col].iloc[i] = f"({cuts[j-1]}, {cuts[j]})"
                    break
        changes.append({"col": col, "cuts": cuts})
    return changes
        
def discretization_test(X: pd.DataFrame, changes: list):
    for change in changes:
        col = change["col"]
        for i in range(len(X[col])):
            for j in range(len(change["cuts"])):
                if X[col].iloc[i] <= change["cuts"][j]:
                    X[col].iloc[i] = f"({change['cuts'][j-1]}, {change['cuts'][j]})"
                    break
        
# 决策树中结点的数据结构，使用双亲表示法
class node(object):
    def __init__(self, X: pd.DataFrame, y: pd.Series, depth: int, ent=-1, father=None, f_value=None, name=None):
        '''
        @X: 划分到该节点的样本（特征）
        @y: 划分到该节点的标签
        @depth: 当前节点的深度，根节点为0
        @ent: 该节点的信息熵
        @father: 该节点的父节点，也是node
        @f_value: 记录父节点到该节点的取值
        @name: 选择特征的名字
        @clildren: 如果下一个节点可分的话依然是node，否则就是标签的值
        '''
        self.X = X
        self.y = y
        self.ent = ent
        self.depth = depth
        self.values = None
        self.father = father
        self.f_value = f_value
        self.name = name
        if name is not None:
            self.values = X[name].value_counts().index
        self.children = []   
    def __repr__(self):
        return f"{self.name}"

class Solution(object):
    def __init__(self, X, y, depth = None, method = 1):
        # @X: 特征
        # @y: 标签
        # @depth: 最大深度
        # @pre: 是否进行预剪枝
        # @dt: 存储生成的决策树，其中的元素应该是node
        # @method: 计算样本纯度的方法，1表示ID3，2表示C4.5，3表示CART
        self.X = X
        self.y = y
        self.depth = len(X.iloc[0])-1
        self.depth = min(depth-1, self.depth)
        print("最大深度", self.depth+1)
        self.dt = []
        self.pre_y = []
        self.method = method

    # 生成决策树
    def fit(self):
        # 首先初始化dt，向其中加入根节点
        root = node(self.X, self.y, 0, ent=Solution.entropy(self.y))
        print("DEBUG: 总的信息熵为", root.ent)
        Solution.get_root(root)
        self.dt.append(root)
        # 对于每一个节点，寻找它的下一层节点
        for n in self.dt:
            # print("!DEBUG: try split", n)
            # 终止条件一：如果当前节点的标签都属于一类
            if len(n.y.value_counts())==1: continue
            # 终止条件二：如果特征已经用完或已达到限制的深度
            if n.depth >= self.depth: continue
            # 向其中加入节点
            nodes = Solution.try_split(n, self.method)
            if len(nodes) == 0: continue
            if nodes is not None: self.dt.extend(nodes)
        print(self.dt)
        self.shape()
        self.draw()
    
    # 遍历dt，补充子节点，便于后面的遍历 
    def shape(self):
        for i in range(len(self.dt)):
            j = i
            for value in self.dt[i].values:
                group = self.dt[i].y[self.dt[i].X[self.dt[i].name]==value].value_counts()
                # 如果结果已经确定，直接加入结果的值
                if len(group) == 1: 
                    self.dt[i].children.append({'type':0, 'value':value, 'next':group.index[0]})
                elif self.dt[i].depth >= self.depth:
                    self.dt[i].children.append({'type':0, 'value':value, 'next':group.idxmax()})
                else:  # 如还需要继续决策，则放入下一个节点
                    while(j < len(self.dt)-1 and self.dt[j].father != self.dt[i]): j += 1
                    self.dt[i].children.append({'type':1, 'value':value, 'next':self.dt[j]})
                    j += 1
                
    # 绘制决策树
    def draw(self):
        # 直接打印就行了，就不花时间绘制了
        print('\n树的结构：\n-------------------------------')
        for n in self.dt:
            print(f"第{n.depth}层 {n.name} 的子节点: ",n.children)
    
    # 后剪枝
    def post(self, train):
        pass
    
    # 使用模型进行预测
    def predict(self, data: pd.DataFrame):
        count = 0
        while count < len(data):
            print(f"DEBUG: 测试第{count+1}行的用例")
            type = 1
            next_value = data.iloc[count][self.dt[0].name]
            print(f"DEBUG: 特征{self.dt[0].name}对应的值为{next_value}")
            next = None
            for n in self.dt[0].children:
                if next_value == n["value"]:
                    next = n["next"];type = n['type'];break
            
            while type != 0:
                next_value = data.iloc[count][next.name]
                print(f"DEBUG: 特征{next}对应的值为{next_value}")
                for n in next.children:
                    if next_value == n['value']:
                        next = n['next'];type = n['type'];break
            self.pre_y.append(next)
            print(f"DEBUG: 预测值为{next}")
            count += 1
        print("\n预测结果: ")
        print(self.pre_y)
 
    # 判断模型的准确率
    def accurcy(self, y: pd.Series):
        y_predict = pd.Series(self.pre_y)
        res = (y_predict == y)
        res = res.value_counts()
        print(f"准确率为:{round(res.iloc[0]/res.sum(), 4)*100}%")
    
    def cal_value(self, y: pd.Series):
        if self.method == 1:
            return Solution.entropy(y)
        elif self.method == 2:
            return Solution.entropy_ratio(y)
        return Solution.gini(y)
    
    @staticmethod  # 计算信息熵
    def entropy(y: pd.Series):
        counter = y.value_counts()
        res = 0.0
        for num in counter.values:
            p = num/len(y)
            res += -p * log2(p)
        return res
    
    # todo，之前写的代码耦合性太高，不好引入CART
    @staticmethod  # 计算基尼系数
    def gini(y: pd.Series):
        pass
    
    @staticmethod
    def get_root(root: node):
        name = None;inc = 0
        for col in root.X:
            ent = 0
            # 统计这个特征中每个值出现的次数
            group = root.X[col].value_counts()
            for i in group.index:  # 计算条件熵的公式
                ent += (group.loc[i]/len(root.y)) * \
                    Solution.entropy(root.y[root.X[col] == i])
            
            temp = root.ent - ent  # 此时的信息增益
            if temp > inc: name = col;inc=temp
            print("DEBUG: 信息熵为", ent)
        print("DEBUG: 选择特征后的信息增益为", inc)
        root.name = name
        root.values = root.X[name].value_counts().index
    
    # 选择出当前结点的下一层结点
    @staticmethod
    def try_split(n: node, m = 1):
        X_all = n.X.copy()
        print("\nDEBUG: 考虑节点", n.name)
        del X_all[n.name]
        nodes = []  # 记录下一层节点
        # 考虑当前特征的每一个取值
        for value in n.values:
            print("DEBUG: 此时取值为", value)
            X = X_all[n.X[n.name]==value]  # 划分下去的样本
            y = n.y[n.X[n.name]==value]  # 划分下去的标签
            if len(y.value_counts()) == 1: continue
            # print(X)
            # print(y)
            a_ent = Solution.entropy(y)
            print("DEBUG: 其信息熵为", a_ent)
            name = None  # 记录选择的特征的名字
            inc = 0  # 信息增益
            tmp = False
            # 如果最后信息熵不是0且只剩一个特征，直接选择这个特征
            if(len(X.columns)==1):
                nodes.append(node(X, y, n.depth+1, a_ent, n, value, X.columns[0]))
                continue
            # 依次计算各个特征的条件熵
            for col in X: 
                print("test")
                ent = 0
                # 统计这个特征中每个值出现的次数
                group = X[col].value_counts()
                for i in group.index:  
                    # 计算条件熵的公式
                    if m == 1 or m == 2:
                        ent += (group.loc[i]/len(y)) * Solution.entropy(y[X[col] == i])
                # ID3算法计算信息增益
                if m == 1: temp = a_ent - ent  
                # C4.5算法计算信息增益率
                elif m == 2: temp = (a_ent - ent)/Solution.entropy(X[col])
                if not tmp or temp > inc: inc = temp;name = col;tmp = True
                print(f"DEBUG: {col}条件熵为", ent)
            if m == 1: print("DEBUG: 选择特征后的信息增益为", inc)
            elif m == 2: print("DEBUG: 选择特征后的信息增益率为", inc)
            print("DEBUG: 选择的特征为", {name})
            nodes.append(node(X, y, n.depth+1, a_ent, n, value, name))
        return nodes
    
if __name__ == "__main__":
    # 离散型测试用例1
    # X = [[0, 1, 'T'], [0, 1, 'S'], [0, 1, 'S'], [0, 0, 'T'], [0, 1, 'T'],
    #     [0, 0, 'T'], [0, 0, 'D'], [1, 0, 'T'], [1, 0, 'T'], [1, 0, 'D'], 
    #     [1, 1, 'D'], [1, 1, 'T'], [1, 1, 'T'], [1, 0, 'S'], [1, 0, 'S']]
    # y = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1]
    # X = pd.DataFrame(X, columns=['有工作', '有房屋', '学历等级'])
    # y = pd.Series(y)
    # s = Solution(X, y, depth=3, method=1)
    # s.fit()
    # s.predict(X)
    # s.accurcy(y)
    
    # 离散型测试用例
    X = pd.read_csv('./data2/train.csv')
    y = X.iloc[:, -1]
    print(X)
    X = X.drop([X.columns[-1]], axis=1)
    X_test = pd.read_csv('./data2/test.txt')
    y_test = X_test.iloc[:, -1]
    X_test = X_test.drop([X_test.columns[-1]], axis=1)
    
    # 连续型测试用例
    # X = pd.read_csv('./traindata.txt')
    # y = X.iloc[:, -1]
    # X = X.drop([X.columns[-1]], axis=1)
    # X_test = pd.read_csv('./testdata.txt')
    # y_test = X_test.iloc[:, -1]
    # X_test = X_test.drop([X_test.columns[-1]], axis=1)
    # changed = discretization_train(X, cut=3)
    # print(X)
    # discretization_test(X_test, changed)
    # print(X_test)
    
    s = Solution(X, y, depth=3, method=1)
    s.fit()
    s.predict(X_test)
    s.accurcy(y_test)