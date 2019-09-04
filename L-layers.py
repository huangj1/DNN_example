import numpy as np

#import matplotlib.pyplot as plt

'''python实现L层神经网络（原理上，不借助tensorflow等框架）'''

# 激活函数
def sigmoid(Z): # 二分类输出层激活函数
	'''
	在numpy中实现sigmoid激活函数（适用于二分类）
	参数：Z——任意形状的numpy数组(线性层的输出，任意形状)
	返回: A——sigmoid(z)的输出，形状（维度）与z相同
		  cache——返回Z，在反向传播时有用
	'''
	A = 1 / (1 + np.exp(-Z))
	assert(A.shape == Z.shape)
	cache = Z
	return A, cache

def relu(Z): # 常用的激活函数（不确定用哪个激活函数，就可以用）
	'''
	在numpy中实现sigmoid激活函数
	参数：Z——任意形状的numpy数组(线性层的输出，任意形状)
	返回: A——激活函数的输出，形状（维度）与z相同
		  cache——返回Z，在反向传播时有用
	'''
	A = np.maximum(0, Z)
	assert(A.shape == Z.shape)
	cache = Z
	return A, cache

def tanh(Z): # 几乎适合所有场合
	'''
	在numpy中实现thanh激活函数
	参数：Z——任意形状的numpy数组(线性层的输出，任意形状)
	返回: A——激活函数的输出，形状（维度）与z相同
		  cache——返回Z，在反向传播时有用
	'''
	A = np.tanh(Z)
	assert(A.shape == Z.shape)
	cache = Z
	return A, cache
'''
def softmax(Z):
	'''
	多分类激活函数
	'''
	Z_exp = np.exp(Z)
	Z_sum = np.sum(Z_exp, axis=1, keepdims=True)
	A = x_exp / x_sum
	cache = Z
	return A, cache
'''

# 激活函数导数（反向传播用）
def sigmoid_backward(dA, cache):
	'''
	为单个SIGMOID单元实现反向传播。
	参数：dA——任意形状的梯度（损失函数）
		  cache——“Z”，
	返回: dZ——关于Z的成本（损失函数）梯度
	'''
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	assert (dZ.shape == Z.shape)
	return dZ

def relu_backward(dA, cache):
	'''
	为单个RELU单元实现反向传播。
	参数：dA——任意形状的梯度（损失函数）
		  cache——“Z”，
	返回: dZ——关于Z的成本（损失函数）梯度
	'''
	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.
	#  当 z <= 0, 设置dz=0  或者dZ用dZ = np.multiply(dA, np.int64(Z > 0))
	dZ[Z <= 0] = 0
	assert (dZ.shape == Z.shape)
	return dZ

def tanh_backward(dA, cache):
	'''
	为单个tanh单元实现反向传播。
	参数：dA——任意形状的梯度（损失函数）
		  cache——“Z”，
	返回: dZ——关于Z的成本（损失函数）梯度
	'''
	Z = cache
	s = np.tanh(Z)
	dZ = dA * (1 - np.power(s, 2))
	assert (dZ.shape == Z.shape)
	return dZ

# 初始化权重W, b
def initialize_parameters_zeros(layers_dims):  # 零初始化（不推荐）
	'''
	参数:layer_dims——python数组(列表)，包含每个层的大小。
	返回:python字典包含参数“W1”，“b1”，..., "WL", "bL";
	WL——权重矩阵(layers_dims[L]， layers_dims[L-1])
	bL——偏置向量(layers_dims[L]， 1)
	'''
	parameters = {}
	L = len(layers_dims) # 网络的层数
	for l in range (1, L):
		parameters['W' + str(l)] = np.zeros((layers_dims[l], layer_dims[l-1]))
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
	return parameters

def initialize_parameters_random(layers_dims): # 随机初始化（不推荐）
	np.random.seed(3) #随机种子
	parameters = {}
	L = len(layers_dims)
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
	return parameters

def initialize_parameters_he(layers_dims): # He初始化(He是论文作者命名)（改善梯度消失或爆炸问题）
	np.random.seed(3) #随机种子
	parameters = {}
	L = len(layers_dims)
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
	return parameters

# 前向传播
def linear_forward(A, W, b): # 线性部分
	'''
	正向传播的线性部分。
	参数:A——前一层(或输入数据)的激活函数值:(前一层的大小，实例数量)
		 W——权值矩阵:numpy数组的形状(当前层大小，前一层大小)
		 b——偏差向量，numpy数组的形状(当前层大小，1)
	返回:
		Z——激活函数的输入，也称为预激活参数
		cache——python字典，包含“A”、“W”和“b”;存储以便有效地计算向后传递
	'''
	Z = np.dot(W, A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation = 'relu'): # 非线性（激活函数）部分
	'''
	正向传播的激活部分。
	参数:A_prev——前一层(或输入数据)的激活函数值:(前一层的大小，实例数量)
		 W——权值矩阵:numpy数组的形状(当前层大小，前一层大小)
		 b——偏差向量，numpy数组的形状(当前层大小，1)
		 activation——该层使用的激活，存储为文本字符串:“sigmoid”或“relu”等
	返回:
		A——激活函数的输出，也称为激活后值
		cache——python字典，包含“linear_cache”和“activation_cache”;;存储以便有效地计算向后传递
	'''
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	if activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	if activation == "tanh":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = tanh(Z)
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)  # A,W,b  Z
	return A, cache

def L_model_forward(X, parameters): #前向传播
	caches = []
	A = X
	L = len(parameters) // 2   # parameters包含W、b,神经网络的层数
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
	caches.append(cache)
	assert(AL.shape == (1,X.shape[1]))
	return AL, caches

# 损失函数
def compute_cost(AL, Y):
    """
    实现成本函数(交叉熵损失函数)。
	参数:
	AL——与标签预测相对应的概率向量，形状(1，例子数量)
	Y——含真实标签的向量(例如:包含0 if non-cat, 1 if cat)， shape(1，例数)
	返回:
	交叉熵
	"""
    m = Y.shape[1]
    # Compute loss from AL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)) # 二分类的交叉熵损失函数
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost
	
# 反向传播
def linear_backward(dZ, cache): # 线性部分
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = 1.0 / m * np.dot(dZ, A_prev.T)
	db = 1.0 / m * np.sum(dZ, axis = 1, keepdims = True) # dZ按行求和得到的列向量
	dA_prev = np.dot(W.T,dZ)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation): # 非线性激活部分
	linear_cache, activation_cache = cache
	if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db
	
def L_model_backward(AL, Y, caches): 
	'''
	实现反向传播[LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID
	'''
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # Y和AL的形状相同
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # 初始化反向传播（二分类）代价函数对AL的偏导
	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	for l in reversed(range(L-1)):  # reversed反向顺序
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	return grads
	
def update_parameters(parameters, grads, learning_rate): # 参数更新（梯度）
	L = len(parameters) // 2 # number of layers in the neural network
	for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
	return parameters

'''	
# 预测函数
def predict(X, y, parameters):
	m = X.shape[1]
	n = len(parameters) // 2 # number of layers in the neural network
	p = np.zeros((1,m), dtype = np.int)
	probas, caches = L_model_forward(X, parameters)
	for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
	print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:])))) # print("Accuracy: "  + str(np.sum((p == y)/m)))   
	return p
'''

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False)  
	np.random.seed(1)
	costs = []
	parameters = initialize_parameters_he(layers_dims)
	
	for i in range(0, num_iterations):
		AL, caches = L_model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL, Y, caches)
		parameters = update_parameters(parameters, grads, learning_rate)
		
		if print_cost and i % 100 == 0:
			costs.append(cost)
			print ("Cost after iteration %i: %f" %(i, cost))
			
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per 100)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
		
	return parameters

#训练调用例
#train_x训练数据、train_y真实标签、layer_dims网络层数（如[3,5,2]表示三层）
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)