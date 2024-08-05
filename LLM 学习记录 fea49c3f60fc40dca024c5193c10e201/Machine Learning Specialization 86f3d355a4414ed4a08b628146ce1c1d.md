# Machine Learning Specialization

课程：[吴恩达《机器学习》](https://www.bilibili.com/video/BV1Pa411X76s/?spm_id_from=333.337.search-card.all.click&vd_source=196cfe00b7812ae8efb6d8458530dccf)

# Course 1 - Supervised Learning Algorithms

**Regression and Classification**

**Linear regression, logistic regression, gradient descent**

## Week 1 - **Machine learning**

**Machine learning**: Field of study that gives computers the ability to learn **without** being **explicitly programmed**. 

### Supervised Learining

- Def: Learns from being given “right answers” label $y$, learn to predict input, output or $X$ to $Y$ mapping.
- Types:
    - Regression(回归): predict a number infinitely many possible outputs.
        - Linear Regression
        Cost function: Squared error cost function $J(w, b) = \frac{1}{2m} \sum\limits_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$
            
            Where $f_{w,b}(x)=wx+b$
            
            The goal is $\underset{w, b}{\text{minimize}} \ J(w, b)$
            
        - Polynomial Regression (多项式回归)
    - Classification(分类): predict categories.
        - Can be non-numeric e.g. cat, dog
        - Can also be numbers which are a small finite limited set of possible output categories

### Unsupervised Learning

- Def: Data only comes with inputs $x$, but not output labels $y$. Algorithm has to find the structure in the data.
- Types:
    - Clustering(聚类): Group similar data points together. Takes data without labels and tries to automatically group them into clusters.
    - Anomaly detection(异常点检测): Find unusual data points.
    - Dimensionality reduction(数据降维): Compress data using fewer numbers.

### Terminology

- Training set: data used to train the model
- Notation:
$x$ = “input” variable or feature, $y$ = ”output” variable or “target” variable, $\hat{y}$ is the estimate or the prediction for y.
$m$ = number of training examples, 
$(x,y)$ = single training example, $(x^{(i)},y^{(i)}) = i^{th}$ training example

### Gradient Descent algorithm

- Goal: $\underset{w_1,...,w_n, b}{\text{minimize}} \ J(w_1,w_2,...,w_n, b)$
- Outline:
    - Start with some initial $w,b$
    - Keep changing $w,b$ to reduce $J$
    - $w = w - \alpha \frac{\partial J(w, b)}{\partial w},\ b = b - \alpha \frac{\partial J(w, b)}{\partial b}$, which $\alpha$ is called learning rate
        - If $\alpha$ is too small, gradient descent may be slow.
        If $\alpha$ is too large, gradient descent may overshoot and never reach minimum (fail to converge, diverge 发散)
    - Until we settle at or near a minimum
        - When we near a local minimum, Derivative (Derivative ) becomes smaller and update steps become smaller so we can reach minimum without decreasing learning rate $\alpha$
- “Batch” gradient descent. "Batch": Each step of gradient descent uses all the training examples. 一次迭代训练所有样本
- stochastic gradient descent:
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled.png)
    

## Week 2

### Multiple Features(variables)

多特征（多变量）问题

- Notition
    
    $x_j$ = $j^{th}$ feature, $n$ = number of features, 
    $\vec{x}^{(i)}$
     = features of $i^{th}$ training example,
    $\vec{x}_j^{(i)}$
     = value of $j^{th}$ feature in the $i^{th}$ training example
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%201.png)
    
- Multiple linear regression
    
    多元线性回归，多个变量，但仍是线性的
    
    features: $\vec{x}=[x_1,x_2,\ldots,x_n]$
    
    $f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b=w_1x_1+w_2x_2+\ldots+w_nx_n+b$
    
- Why vectorize implementations of learning algorithms run more efficiently?
    
    numpy arrays use parallel processing (并行处理) hardware to compute
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%202.png)
    
- Implement gradient descent for multiple linear regression with vectorization
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%203.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%204.png)
    
- An alternative way to gradient descent: Normal equation
    - Normal equation: only for linear regression to solve $w, b$ without iterations.
    - Disadvantages:
        - doesn't generalize to other learning algorithms.
        - Slow when number of features is large(> 10,000)
    - Some machine learning libraries may use normal equation method to implement linear regression.
    - Gradient descent is the recommended method for finding parameters $w, b$

### Operate Features

- Feature re-scaling
    - Why to rescaling the different features
        
        it will cause the gradient descent to run more slowly
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%205.png)
        
    - How to
        - Mean normalization(均值归一化)
        - Z-score normalization(Z-score 标准化)
- Feature Engineering
    
    Using intuition to design new features, by transforming or combining original features.
    

### Checking Gradient Descent for Convergence

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%206.png)

### Choosing the Learning Rate

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%207.png)

### Polynomial Regression

多项式回归

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%208.png)

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%209.png)

## Week 3 Classification

binary classification: label $y$ can only be one of two values

### Logistic Regression

- Logistic Regression is an algorithm of classification
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2010.png)
    
- Decision boundary
    
    The decision boundary of logistic regression is **$z=0$**
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2011.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2012.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2013.png)
    

### Cost Function for Logistic Regression

called log loss or cross−entropy loss

选这个 lost function 的好处：该函数是凸的，可以找到全局最小值

Lost function: measures how well you’re doing on one training example

$L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) =
\begin{cases}
-\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y = 1 \\
-\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y = 0
\end{cases}$
Cost function: summing up the losses on all of the training examples

$J(\vec{w}, b) = \frac{1}{m} \sum\limits_{i=1}^{m} L( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} )$

Simplified loss lunction:

$L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) - (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))$

$J\left( \vec{w}, b\right) = - \frac{1}{m} \sum\limits_{i=1}^{m} \left[y^{(i)} \log\left(f_{\vec{w},b}\left(\vec{x}^{(i)}\right)\right) + \left(1 - y^{(i)}\right) \log\left(1 - f_{\vec{w},b}\left(\vec{x}^{(i)}\right)\right)\right]$

### Gradient Descent Implementation

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2014.png)

- difference between linear regression and logistic regression
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2015.png)
    
    Sigmoid 函数把直线压缩到 0 和 1 之间
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2016.png)
    

### The Problem of Overfitting

underfit → high bias

overfit → high variance

- Example
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2017.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2018.png)
    
- Tool to deal with **overfitting**
    - Collect more **training examples** to train the model
    - Select features to include/exclude
        - all features + insufficient data → overfit
        - disadvantage: useful features could be lost
    - Regularization
        - Reduce the size of parameters $w_j$, without setting it to exactly 0
        - keep all of the features, but just prevents the features from having an overly large effect
- Regularization
    - Regularized Linear regression
        
        $J(\vec{w}, b) = \frac{1}{2m} \sum\limits_{i=1}^{m}  ( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} )^2+\frac{\lambda}{2m}\sum\limits_{j=1}^{n}w_j^2$
        
        - $\frac{1}{2m} \sum\limits_{i=1}^{m}  ( f_{w,b}(\vec{x}^{(i)}) - y^{(i)} )^2$: mean squared error → fit data
        - $\frac{\lambda}{2m}\sum\limits_{j=1}^{n}w_j^2$: regularization term → keep $w_j$ small
        - $\lambda$ > 0: regularization parameter → balance both goals
    - Gradient descent with regularization
        
        repeat {
        
        $w_j = w_j - \alpha \left\{\frac{1}{m} \sum\limits_{i=1}^{m} \left[ ( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} )x_j^{(i)}\right]+\frac{\lambda}{m}w_j\right\}$ 
        
        $b = b - \alpha \left[\frac{1}{m} \sum\limits_{i=1}^{m}  ( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} )\right]$
        
        } simultaneous update
        
    - Regularized Logistic Regression
        
        $J\left( \vec{w}, b\right) = - \frac{1}{m} \sum\limits_{i=1}^{m} \left[y^{(i)} \log\left(f_{\vec{w},b}\left(\vec{x}^{(i)}\right)\right) + \left(1 - y^{(i)}\right) \log\left(1 - f_{\vec{w},b}\left(\vec{x}^{(i)}\right)\right)\right]+\frac{\lambda}{2m}\sum\limits_{j=1}^{n}w_j^2$
        

# Course 2 - Advanced Learning Algorithms

**Neural Networks, Decision Tree Model, advice for ML**

## Week 1

**Neural Networks (Deep learning algorithms)**

### Intuitions of neural networks

- Usage
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2019.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2020.png)
    
- Neural network examples
    
    activation function (非线性激活函数)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2021.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2022.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2023.png)
    
    - a more complex neural network
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2024.png)
        
    - forward propagation(向前传播)

### Carry out interference in TensorFlow

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2025.png)

- build a neural network
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2026.png)
    

## Week 2

### Train a Neural Network in TensorFlow

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2027.png)

- Details:
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2028.png)
    
    - Step 1: Create the model
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2029.png)
        
    - Step 2: Loss and cost functions
    for binary classification: binary cross entropy$L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) - (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))$
    - Step 3: Gradient descent
    Compute derivatives for gradient descent using “back propagation”
- Alternatives to the sigmoid activation
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2030.png)
    
- Choosing activation functions
    - Output layer: when choose the activation function for output layer, usually depending on what is the label $y$ you are trying to predict
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2031.png)
        
    - Hidden layer: ReLU is most common
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2032.png)
    
- Why do we need activation functions?
    
    > 因为神经网络中每一层的输入输出都是一个线性求和的过程，下一层的输出只是承接了上一层输入函数的线性变换，所以如果没有激活函数，那么无论你构造的神经网络多么复杂，有多少层，最后的输出都是输入的线性组合，纯粹的线性组合并不能够解决更为复杂的问题。而引入激活函数之后，我们会发现常见的激活函数都是非线性的，因此也会给神经元引入非线性元素，使得神经网络可以逼近其他的任何非线性函数，这样可以使得神经网络应用到更多非线性模型中。
    > 
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2033.png)
    
    if all g(z) are linear → no different than linear regression
    
    **Fact: a linear function of a linear function is itself a linear function**
    
    Don't use linear activations in hidden layers
    

### Multiclass classification problem

- def: target $y$ can take on more than two possibe values. 一个多分类问题
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2034.png)
    
- Softmax
    
    softmax regression algorithm: a generalization of the logistic regression algorithm
    
    逻辑回归用于解决二分类问题，softmax 用于解决多分类问题
    
    - Def
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2035.png)
        
    - Cost
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2036.png)
        
        $L= -\sum\limits_{i=1}^{N} y_i \log a_i$
        
- Neural Network with Softmax output
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2037.png)
    
    `loss = SparseCategoricalCrossentropy()`  稀疏分类交叉熵
    

### Multi-label classification problem

associated with single input X, there may be many labels. 相当于多个二分类问题

用了多个 sigmoid 而不是 softmax 函数

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/cbfac23a-e838-44f3-ab7f-cc58853bfe4e.png)

### Advanced Optimization

Adam Algorithm: Adaptive Moment estimation

- Not just one global $\alpha$
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2038.png)
    
- Can adjust the learning rate automatically
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/5d5af004-4572-4cac-8bff-fd4b20b56e52.png)
    
- Code
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/b4f537f5-e63d-4222-accc-b1ae735a02b7.png)
    

### Additional Layer Types

- Dense layer(密集层)
    
    in which every neuron in the layer gets its inputs from all the activations  outputs of the previous layer
    
- Convolutional Layer(卷积层)
    
    each Neuron only looks at part of the previous layer's inputs
    
    Why:
    
    - Faster computation
    - Need less training data (less prone to overfitting)
- Convolutional Neural Network
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2039.png)
    

### Back **Propagation**

Conpute cost function $J$: left to right computation (forward prop)

Conpute all the derivative: right to left (back prop)

- Derivative
    
    $\text{Let } \frac{\partial J(w)}{\partial w} = k, \text{ When } w \text{ ↑ } \varepsilon \rightarrow 0, \text{ Thus } J(w) \text{ ↑ } k \times \varepsilon$
    
    微分代表特征（变量） $w$ 对目标 $J(w)$ 的影响程度的大小，微分越大，影响程度越大
    
- Computation Graph
    - Example: let linear activation $a=g(z)=z$，only 1 layer of 1 node
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2040.png)
        
    - Another example:
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2041.png)
        
- Advantage
    
    Backprop is an efficient way to compute derivatives
    
    Compute $\frac{\partial J}{\partial a}$ once and use it to compute both $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$ ,
    If N nodes and p parameters, compute derivatives in roughly N + p steps rather than N x p steps.
    
    (N x p 的情况是将 $w^{[1]}$、 $w^{[2]}$、 $b^{[1]}$、 $b^{[2]}$等参数都稍微提高一点点看 $J$ 有多少变化的时候，从左向右算会需要 N x p 步)
    

## Week 3

### Evaluating a model

- Notation
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2042.png)
    
- Train/test procedure
    
    Once parameters $w,b$ are fit to the training set, the training error $J_{train}(w, b)$ is likely lower than the actual generalization error.
    $J_{test}(w, b)$ is better estimate of how well the model will generalize to new data than $J_{train}(w, b)$.
    
    - Linear regression (with squared error cost)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/a204ee37-5011-4216-90e8-6ac0770eb1cd.png)
        
    - Classification problem
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2043.png)
        
        another way to define the $J$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2044.png)
        

### Model selection (choosing a model)

- Training/cross validation/test set
    
    cross validation set(交叉验证集), also called validation set or development set or dev set
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2045.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2046.png)
    
- Model selection
    - Procedure
        - fit parameters $\vec{w}$ and $b$ using the training set
        - choose the parameter $d$ (dimension) using the dev set
        - because we haven’t made any decisions using the test set, $J_{test}$ will be a fair and not overly optimistic estimate of the generalization error of the model to be reported
    - Example
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2047.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2048.png)
        

### Diagnosing bias and variance

- Machine learning diagnostic
    
    Is a test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance.
    Diagnostics can take time to implement but doing so can be a very good use of your time.
    
- Bias and Variance
    - High bias (underfit, 欠拟合): $J_{train}$ is high
    - High variance (overfit, 过拟合): $J_{cv}$ is much higher than $J_{train}$, which means the network does much better on data it has seen than on data it has not seen.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2049.png)
    
    - “$J$ - degree of polynomial”
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2050.png)
        
- Regularization and bias/variance
    - Linear regression with regularization
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2051.png)
        
    - Choosing the regularization parameter $\lambda$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2052.png)
        
    - $J$ - $\lambda$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2053.png)
        
        very small $\lambda$ → high variance (overfit)
        
        very large $\lambda$ → high bias (underfit)
        

### Establishing a baseline level of performance

What is the level of error you can reasonably hope the network to get to

- Human level performance
- Competing algorithms performance
- Guess based on experience

examples:

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2054.png)

### Learning curves

Learning curves are a way to indicate how learning algorithm is doing as a function of the amount of “experience” it has.

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2055.png)

- High bias
    
    If a learning algorithm suffers from high bias (underfit, 欠拟合), getting more training data will not (by itself) help much.
    
    The model is too simple to be fitting into too much data
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2056.png)
    
- High variance
    
    If a learning algorithm suffers from high variance (overfit,  过拟合), getting more training data is likely to help.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2057.png)
    

### Debugging a learning algorithm

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2058.png)

Debug a learning algorithm to meet the bias variance tradeoff

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2059.png)

### Bias/variance in Neural Networks

Neural networks offer us a way all of this dilemma of having to tradeoff bias and variance

- Large neural networks that trained on small size dataset are low bias machines
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2060.png)
    
- A large neural network will usually do as well or better than a smaller one so long as regularization is chosen appropriately.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2061.png)
    

### Iterative loop of ML development

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2062.png)

### Error analysis

- What: In terms of the most important ways to run **diagnostics** to choose what to do next to improve learning algorithm performance, bias and variance is the most important idea and error analysis is the second
- How: When algorithm misclassified of mislabled some of the examples in cross-validation set, **manually** examine (part of) them to categorize them based on common traits, to find out the most common types of errors to most fruitful to focus our attention
- Example: detecting fishing emails
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2063.png)
    
- Limitation: Error analysis can be a bit harder for tasks that humans are not good at.

### Adding more data

- Based on the result of error analysis to get more data of just the types chould be a more efficient way to add just a little bit of data but boost the performance of the algorithms by quite a lot.
- Data Augmentation (数据增强): modifying an existing training example to create a new training example.
The new training examples together with the original one have the same lable.
    - Examples:
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2064.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2065.png)
        
    - Tips: Distortion (失真) introduced should be representation of the type of noise/distertions in the test set.
    Usually, adding purely random/meaningless noise to your data does not help.
- Data synthesis
Using artificial data inputs to create a new training example.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2066.png)
    

### Transfer Learning

- Key idea
    
    Take data from a barely related tasks to get your algorithm to do better on your application.
    
- How to
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2067.png)
    
    - Step1: Supervised Pretraining
        - Pretraing the neural network on a very large dataset
    - Setp2: Fine Tuning
        - Fine tune the weights to suit the specific application of the task.
        - Make a copy of a pretrained network and keep parameters $\mathbf{W}^{[1]},\vec{b}^{[1]},\dots,\mathbf{W}^{[4]},\vec{b}^{[4]}$, and replace the output layer and the parameters of output layer.
            - Option 1: When we have a very small training set, only train output layers parameters
            - Option 2: When we have a slightly larger training set, train all parameters, and the parameters would be initialized using the values that had been pretrained
- Intuition
    
    Why transfer Learning can work well?
    
    By pretraining, the neural network can detect pretty **generic features of images** such as edges … 
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2068.png)
    
- Summary
    - Download open source neural network parameters pretrained on a large dataset with the same input type (e.g.,images, audio, text) as your application (or train your own).
    - Further train (fine tune) the network on your own data.

### Full cycle of a machine learning project

![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2069.png)

- Deployment
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2070.png)
    
    When deploying a ML model, software engineering may be needed for:
    
    - Ensure reliable and efficient predictions
    - Scaling
    - Logging
    - System monitoring
    - Model updates
- MLOps: machine learning operations
the practice of how to **systematically build and deploy and maintain** machine learing systems.
To do all of these things to make sure that your model is reliable, scales well, has good laws, is monitored and can be updated.

### Fairness, bias, and ethics

Guidelines:

- Get a diverse team to brainstorm things that might go wrong, with emphasis on possible harm to vulnerable groups.
- Carry out literature search on standards/guidelines for your industry.
- Audit systems against possible harm prior to (在…之前) deployment.
- Develop mitigation plan (if applicable), and after deployment, monitor for possible harm.

### Error metrics for skewed datasets

When the ratio of positive to negative examples is very skewed (偏斜, very far from 50-50), the usual error metrics (指标) like accuracy don’t work well.

- Rare disease classification example
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2071.png)
    
- Precision/Recall
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2072.png)
    
- Trading off precision and recall
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2073.png)
    
    - F1 score: the harmonic mean ([**调和平均数**](https://baike.baidu.com/item/%E8%B0%83%E5%92%8C%E5%B9%B3%E5%9D%87%E6%95%B0/9661021)) of $P$ and $R$, which pays more attention to whichever is much lower
    $\text{F1 score} = \frac{1}{\frac{1}{2}\left(\frac{1}{P}+\frac{1}{R}\right)} = 2\frac{PR}{P+R}$

## Week 4

### Decision Tree Model

- Example
    
    Notation: 
    
    top: root node, middle: decision nodes, bottom: leaf nodes
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2074.png)
    
- Learning Process
    - Steps
        - Start with all examples at the root node
        - Calculate **information gain** for all possible features, and pick the one feature with the highest information gain
        - Split dataset according to the selected feature, and create left and right branches of the tree
        - Keep repeating the splitting process above until one of the stopping criteria is met
        
        Recursive algorithm: build the overall decision tree by building the left and the right sub-branches and putting them together
        
    - Decision to make
        - How to choose what feature to split on at each node to maximize purity (or minimize impurity)? 
        Use **information gain**
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2075.png)
            
        - When to stop splitting?
        The **stopping criterium**
            - When a node is 100% one class
            - When number of examples in a node is below a threshold
            - When splitting a node will result in the tree exceeding a maximum depth.
            (One reason to limit the depth of the decision tree is to keep the tree small and reduce the risk of overfitting.)
            - When improvements in purity score (Information gain from additional splits) are below a threshold (阈值)
- Measuring impurity
    
    **Entropy** (熵) is a measure of the impurity of a set of data.
    
    denote:  $p_1$ = fraction of examples with labels $1$
    
    def: $H(p_{1}) = -p_{1} \log_{2}(p_{1}) - (1-p_{1}) \log_{2}(1-p_{1})$
    
    example:
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2076.png)
    
- Choosing a split: by **Information Gain**
    
    Aim: reduces entropy or reduces impurity, or maximizes purity.
    
    The reduction of entropy is called **information gain** (信息增益).
    
    def: $IG = H(p_{1}^{root}) - \left(w^{left}H(p_{1}^{left}) + w^{right}H(p_{1}^{right})\right)$
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2077.png)
    
- Using one-hot encoding of cateorical feature
    
    If a categorical feature can take on $k$ values, use $k$ binary features (0 or 1 valued) to replace it.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2078.png)
    
- Continuous valued feature
    
    Try different thresholds, do the usual **information gain calculation** and split on the continuous value feature with the selected threshold if it gives you the best information gain out of all possible features to split on.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2079.png)
    
- Regression Tree (回归树)
    - Choose a split
    use **reduction of variance** rather than reduction of entropy
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2080.png)
        
    - Regression with Decision Trees
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2081.png)
        

### Tree ensembles

Tree ensemble is a collection of multiple trees

- Using multiple decision trees
    
    A single tree is highly sensitive to smal changes of the data
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2082.png)
    
    use multiple trees to predict the result
    
    the final prediction is the average of all the predictions from the trees
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2083.png)
    
- Sampling with replacement (带放回抽样)
    
    Sampling with replacement let us construct a new training set that’s different from our original training set by picking from all examples with equal $\frac{1}{m}$ probability, if we need a training set of size $m$ (duplicates in a training set is workable)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2084.png)
    
- Random Forest algorithm
    - Bagged decision tree
        - Given training set of size $m$
        - For $b=1$ to $B$
            - Use sampling with replacement to create a new training set of size $m$
            - Train a decision tree on the new dataset
    - Random Forest algorithm
    To further randomizing the feature choice: 
    At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k<n$ features and allow the algorithm to only choose from that subset of features.
    A typical choice for the value of $k$ is $k=\sqrt{N}$
- XGBoost
    
    XGBoost is the most commonly used implementation of decision tree ensembles
    
    - Boosted trees intuition
        - Given training set of size $m$
        - For $b=1$ to $B$
            - Use sampling with replacement to create a new training set of size $m$. **But instead of picking from all examples with equal** $\frac{1}{m}$ **probability, make it more likely to pick examples that the previously trained trees misclassify.**
            - Train a decision tree on the new dataset
    - XGBoost (eXtreme Gradient Boosting)
        - Open source implementation of boosted trees
        - Fast efficient implementation
        - Good choice of default splitting criteria and criteria for when to stop splitting
        - Built in regularization to prevent overfitting
        - Highly competitive algorithm for machine learning competitions (eg:Kaggle competitions)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2085.png)
        

### When to use decision trees

Comparing decision trees and neural networks:

Decision Trees and Tree ensembles

- Works well on tabular (structured) data
- Not recommended for unstructured data (e.g. images, audio, text)
- Very fast to train
- Small decision trees may be human interpretable

Neural Networks

- Works well on all types of data, including tabular (structured) and unstructured data
- May be slower to train than a decision tree
- Work well with transfer learning
- When building a system of multiple models working together, it might be easier to string together multiple neural networks.
(The basic reason is neural networks compute the output $y$ as a smooth or continuous function of the input $x$, and so even if you string together a lot of different models, the outputs of all of these different models is itself differentiable (可微的) and so you can train them all at the same time using gradient descent.)

# Course 3 - Other Learning Algorithms

## Week 1 Unsupervised Learning

### Clustering

A **clustering** **algorithm** (聚类) looks at a number of data points (without label) and automatically finds data points that are related or similar to each other.

- Applications
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2086.png)
    
- K-means
    - K-means algorithm
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2087.png)
        
    - K-means for clusters that are not well separated
        
        can also do well
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2088.png)
        
    - K-means optimization objective
        - Notation
            - $c^{(i)}$ = index of cluster (1,2,..., $K$) to which example $x^{(i)}$ is currently assigned
            - $\mu_k$ = cluster centroid k
            - $\mu_{c^{(i)}}$  = cluster centroid of cluster to which example $x^{(i)}$ has been assigned
        - Cost function (Distortion)
        $J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K) = \frac{1}{m} \sum\limits_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$
        - Why K-means algorithm is optimizing cost function $J$
            - Assign points to cluster centroids:
            update $c^{(i)}$ to minimize $J$ while hoding $\mu_k$
            - Move cluster centroids
            update $\mu_k$ to minimize $J$ while hoding $c^{(i)}$
    - Initializing K-means
        - Algorithm
            
            $A\stackrel{\mathrm{def}}{=}$
             {
            
            Choose $K$< $m$ (the num of examples).
            
            Randomly pick $K$ training examples.
            
            Set $\mu_{1},\mu_{2},...,\mu_{k}$ equal to these $K$ examples.
            
            }
            
            - Step 1: For $i$ = 1 to 100 {
                Randomly initialize K-means using $A$ .
                Run K-means. Get $c^{(1)},...,c^{(m)},\mu_1,...,\mu_K$
                Computer cost function (distortion) $J$
            }
            - Step 2: Pick set of clusters that gave lowest cost $J$
        - Reason to repeat initializing K-means
            
            Some choice of random initialization may lead to local minimum (like the two examples below).
            
            Run it multiple times to try to find the best local optima.
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2089.png)
            
    - Choosing the Number of Clusters $K$
        - Elbow method
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2090.png)
            
        - By purpose
        Sometimes, you're running K-means to get clusters to use for some later/downstream purpose.
        Evaluate K-means based on a metric (指标) for how well it performs for that later purpose.
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2091.png)
            

### Anomaly detection

- Intuition: Finding unusual events
    - Examle
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2092.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2093.png)
        
    - method: Density estimation
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2094.png)
        
- Gaussian (Normal) distribution
    
    Say $x$ is a number. lf $x$ is a distributed Gaussian with mean $μ$, $σ$ variance 
    
    $$
    p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2095.png)
    
    - Parameter estimation
        
        Dataset: $\{{x^{(1)},x^{(2)},\ldots,x^{(m)}}\}$
        
        The maximum likelihood estimates(最大似然估计) of $\mu$ and $\sigma$
        
        $$
        \begin{align*}
        \mu_{\text{ML}} &= \frac{1}{m}\sum\limits_{i=1}^{m} x^{(i)} \\
        \sigma_{\text{ML}}^2 &= \frac{1}{m}\sum\limits_{i=1}^{m} (x^{(i)} - \mu_{\text{ML}})^2
        \end{align*}
        
        $$
        
- Anomaly detection algorithm (异常检测)
    - Choose $n$ features $x$,that you think might be indicative of
    anomalous examples.
    - Fit parameters $\mu_1,\dots,\mu_n,\sigma_{1}^n,\dots,\sigma_{n}^2$ using
     $\vec{\mu} = \frac{1}{m}\sum\limits_{i=1}^{m} \vec{x}^{(i)}$   $\vec\sigma^2= \frac{1}{m}\sum\limits_{i=1}^{m} (\vec{x}^{(i)} - \vec\mu)^2$
    - Given new examplex, compute $p(x)$:
    $p(\vec{x}) = \prod\limits_{j=1}^{n} p\left(x_j;\mu_j,\sigma_j\right)=\prod\limits_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} e^{-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}}$
    - Anomaly if $p(x)<\varepsilon$
- Developing and evaluating an detection system
    
    When developing a learning algorithm(choosing features, etc.).
    making decisions is much easier if we have a way of evaluating
    our learning algorithm.
    
    Assume we have some labeled data, of anomalous and non-
    anomalous examples.($y=0$ if normal, $y= 1$ if anomalous).
    
    Training set: $\{{x^{(1)},x^{(2)},\ldots,x^{(m)}}\}$(assume all normal $y=0$)
    
    Cross-validation set and test set (include a few anomalous examples $y=1$)
    
    - Aircraft engines monitoring example
        - Dataset: 10000 good (normal) engines and 20 flawed engines(anomalous) ($y=1$)
            - Training set: 6000 good engines → train algorithm
            - CV: 2000 good engines,10 anomalous → tune parameter $\varepsilon$, add or subtract features $x_j$
            - Test: 2000 good engines,10 anomalous → evaluation and report
        - Alternative:
            - Training set: 6000 good engines
            - CV: 4000 good engines,20 anomalous → tune, evaluation and report
            - No test set
    
    Since training set really has no label, this is still an unsupervised learning algorithm
    
    - Algorithm evaluation
        - Fit model $p(x)$ on training set $\{{x^{(1)},x^{(2)},\ldots,x^{(m)}}\}$
        - On a cross validation/test example $x$, predict
        $y =
        \begin{cases}
        0 \quad& \text{if}\  p(x)<\varepsilon \\
        1 \quad& \text{if}\ p(x)\geq\varepsilon
        \end{cases}$
        - Since the data distribution is very skewed, possible evaluation metrics:
        - True positive, false positive, false negative, true negative → Precision/Recal → F1-score
- Anomaly detection vs. supervised learning
    
    
    |  | Anomaly detection | Supervised learning |
    | --- | --- | --- |
    | Dataset | Very small number of positive examples (y = 1). (0-20 is common) Large number of negative (y=0) examples. | Large number of positive and negative examples. |
    | Ways to look at examples | Many different “types” of anomalies. Hard for any algorithm to learn from positive examples
    what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far. | Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set. |
    | Examples | Fraud detection
    Manufacturing- Finding new previously unseen defects in manufacturing.(e.g. aircraft engines)
    Monitoring machines in a data center | Email spam classification
    Manufacturing - Finding known, previously seen defects
    Weather prediction (sunny/rainy/etc.)
    Diseases classification |
- Choosing what features to use
    - Non-gaussian features
    whatever transformation applied to the training set,  remember to apply the same transformation to cross validation and test set data as well.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2096.png)
    
- Error analysis for anomaly detection
    - Want
    $p(x)\geq\varepsilon$  large for normal examples x.
    $p(x)<\varepsilon$  small for anomalous examples x.
    - Most common problem:
    $p(x)$ is comparable (say, both large) for normal and anomalous examples
    - The development process will often go through is to train the model and then to see what anomalies in the CV set the algorithm is failing to detect. And then to look at those examples to see if that can inspire the creation of new features that would allow the algorithm to spot. By the new features, the algorithm can successfully flag those examples as anomalies.
    - Deciding feature choice based on $p(x)$
    large for normal examples, and becomes small for anomaly in the CV set

## Week 2 Recommender Systems

Recommender Systems (推荐系统)

### Making recommendations

- Collaborative Filtering Algorithm(协同过滤算法)
    - Intuition
        
        multiple users have rated the same movie collaboratively, given we a sense of what this movie maybe like, given we a senseof what this movie maybe like,that allows you to guess what are appropriate features for that movie and this in turn allows you to predict how other users that haven't yet rated that same movie may decide to rate it in the future.
        
    - Using per-item features
        
        Assume that we know in advance the values of the features of the movies $x^{(i)}$.
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2097.png)
        
        - Notition:
            
            $n_{u}$ = no. of users, $n_m$ = no. of movies, 
            
            $r(i,j)=1$ if user $j$ has rated movie $i$, 0 otherwise
            
            $y(i,j)$ = rating given by user $j$ to movie $i$ (defined only if $r(i,j)=1$)
            
            $w^{(j)},b^{(j)}$ = parameters for user $j$
            $x^{(i)}$ = feature vector for movie $i$
            
            $m^{(j)}$ = no. of movies rated by user $j$
            
        - Cost function
        For user $j$ and movie $i$, predict rating: $w^{(j)}x^{(i)} + b^{(j)}$
        To learn $w^{(j)}$ and $b^{(j)}$:(like linear regression)
            
            $J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}} \sum\limits_{i:r(i,j)=1} (w^{(j)}x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2m^{(j)}} \sum\limits_{k=1}^{n} (w_k^{(j)})^2$
            
    - Conversely
        
        We don’t know in advance the values of the features.
        
        Assume that we have the parameters $w$ and $b$ of different users.
        
        Given $w^{(1)},b^{(1)},w^{(2)},b^{(2)},...,w^{(n_u)},b^{(n_u)}$
        
        To learn the features of the $i^{th}$ movie$i$$x^{(i)}$:
        
        $J(x^{(i)}) = \frac{1}{2} \sum\limits_{j:r(i,j)=1} (w^{(j)}x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum\limits_{k=1}^{n} (x_k^{(i)})^2$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2098.png)
        
    - Collaborative Filtering Algorithm
        - Cost function to learn $w^{(1)},b^{(1)},...,w^{(n_u)},b^{(n_u)}$:
        $J(W, b) = \frac{1}{2} \sum\limits_{j=1}^{n_u} \sum\limits_{i:r(i,j)=1} (w^{(j)} x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{1}{2} \sum\limits_{j=1}^{n_u} \sum\limits_{k=1}^{n} (w_k^{(j)})^2$
        - Cost function to learn $x^{(1)},x^{(2)},...,x^{(n_m)}$:
        $J(X) = \frac{1}{2} \sum\limits_{i=1}^{n_m} \sum\limits_{j:r(i,j)=1} (w^{(j)} x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{1}{2} \sum\limits_{i=1}^{n_m} \sum\limits_{k=1}^{n} (x_k^{(i)})^2$
        - Put them together:
        $J(W, b, X) = \frac{1}{2} \sum\limits_{(i,j):r(i,j)=1} (w^{(j)} x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{1}{2} \sum\limits_{j=1}^{n_u} \sum\limits_{k=1}^{n} (w_k^{(j)})^2 + \frac{1}{2} \sum\limits_{i=1}^{n_m} \sum\limits_{k=1}^{n} (x_k^{(i)})^2$
        $\min\limits_{W,b,X} J(W, b, X)$
    - Gradient Descent
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%2099.png)
        
- Binary labels: favs, likes and clicks
    - Example applications
        - Did user $j$ purchase an item after being shown?
        - Did user $j$ fav/like an item?
        - Did user $j$ spend at least 30sec with an item?
        - Did user $j$ click on an item?
        - Meaning of ratings:
            - 1 - engaged after being shown item
            - 0 - did not engage after being shown item
            - ? - item not yet shown
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20100.png)
        
    - Cost function for binary application
        
        $f_{(w,b,x)}(x) = g(w^{(j)}\cdot x^{(i)} + b^{(j)})$
        
        Loss for single example: 
        
        $L(f_{(w, b, x)},y^{(i,j)}) = -y \log(f_{(w, b, x)}(x)) - (1 - y) \log(1 - f_{(w, b, x)}(x))$
        
        Cost for all examples:
        
        $J(W, b, X) = \sum\limits_{(i,j):r(i,j)=1} L(f_{(w, b, x)},y^{(i,j)})$
        
- Mean normalization
    - Users who have not rated any movies
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20101.png)
        
    - Mean Normalization
        
        the effect ofthis algorithm is it will cause the initial guesses for the new user Eve to be just equal to the mean of whatever other users have rated these five movies. And that seems more reasonablecto take the average rating of the movies. rather than to guess that all the ratings by Eve will be zero.
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20102.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20103.png)
        
- TensorFlow implementation
    - Auto Diff
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20104.png)
        
    - Collaborative Filtering Algorithm
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20105.png)
    
- Finding related items
    
    By Collaborative filtering algorithm, we got the features $x^{(i)}$ of item $i$, but the features are hard to interpret.
    
    To find other items related to it,find item $k$ with $x^{(k)}$ similar to $x^{(i)}$ by distance
    
    $d(x^{(i)}, x^{(k)}) =\|x^{(k)} - x^{(i)}\|^2= \sum\limits_{j=1}^{n}(x_{j}^{(i)} - x_{j}^{(k)})^2$
    
- Limitations of Collaborative Filtering
    - Cold start problem (冷令启动问题(初始数据不足)). How to:
        - rank new items that few users have rated?
        - show something reasonable to new users who have rated few
        items?
        - Use side information about items or users (难以利用其他信息):
        Item: Genre, movie stars, studio,
        User: Demographics (age, gender, location), expressed
        preferences, …

### Content-based filtering

- Collaborative filtering VS Content-based filtering
    - Collaborative filtering:
    Recommend items to you based on **rating of users** who
    gave similar ratings as you
    - Content-based filtering:
    Recommend items to you based on **features of user and item** to **find** **good match** between users and the items
    ****Vector size could be different between $x^{(j)}_u$ and $x^{(i)}_m$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20106.png)
        
- Content-based filtering: Learning to match
    
    Compute the vectors, $v_u$ from $x_u$ for the users and $v_m$ from $x_m$ for the items and take dot products between them to try to find good matches.
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/9fdb821e-f31e-4178-beba-193cfe2c0d86.png)
    
- Deep learning for content-based filtering
    - Neural network architecture
        
        the prediction by user $j$ on movie $i$,
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20107.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20108.png)
        
    - Find movies similar
        
        $d(v_m^{(i)}, v_m^{(j)}) =\|v_m^{(i)} - v_m^{(j)}\|^2= \sum\limits_{j=1}^{n}(v_m^{(i)} - v_m^{(j)})^2$
        
    - Recommending from a large catalogue
        
        Two steps: Retrieval (检索) & Ranking (排名)
        
        - Retrieval:
            - Generate large list of plausible item candidates
             e.g.
                - For each of the last 10 movies watched by the user,
                find 10 most similar movies
                - For most viewed 3 genres, find the top 10 movies
                - Top 20 movies in the country
            - Combine retrieved items into list, removing duplicates and items already watched/ourchased
        - Ranking:
            - Take list retrieved during the last step and rank using learned model
            Additional potimization: compute $V_m$ for all the movies in advance
                
                ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20109.png)
                
            - Display ranked items to user
        
        In retrieval step: 
        
        Retrieving more items results in better performance, but slower recommendations.
        To analyse/optimize the trade-off, carry out offline experiment to see if retrieving additional items results in more relevant recommendations(i.e., $p(y^{(i,j)})=1$ of items displayed to user are higher).
        
- TensorFlow implementation
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20110.png)
    

### Principal Comoonents Analysis

PCA algorithm (主成分分析)

- Intuition
    
    The idea of PCA is to find new axis and coordinates Use fewer numbers
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20111.png)
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20112.png)
    
- Algorithm
    - Preprocess features: Normalized to have zero mean and apply feature scaling
    - Choose an axis
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20113.png)
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20114.png)
        
- Applications
    - Visualization: redvce to 2or 3 features
    
    Less frequently used for:
    
    - Data compression: to reduce storage or transmission costs
    - Speeding up training of a supervised learning model

## Week 3 Reinforcement Learning

(强化学习)

- Intro
    
    For a lot of task of controlling a robot, supervised learning approach doesn’t work well and use reinforcement learning instead.
    
    In reinforcement learning, we **tell the model what to do rather than how to do it**, and specifying the reward function rather than theoptimal action.
    
- The Return (收益) in reinforcement learning
    
    Reward (奖励): scalar feedback signal provided by the environment
    
    Return = $R_1+\gamma R_2+\gamma^2 R_3+...$ (until terminal state reached)
    
    $\gamma$ is discount factor (折扣率) is a number usually close to $1$
    
    The return depends on the actions you take
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20115.png)
    
    Negative reward (负奖励): cause the algorithm to try to push out it as far into the future as possible
    
- Making decisions: Policies in reinforcement learning
    
    A policy (策略) is a function $\pi(s)= a$ mapping from states to actions, that tells you what action $a$ to take in a given state $s$.
    
    The goal of reinforcement learning:
    
    Find a policy $\pi$ that tells you what action $a = π(s)$ to take in every state $s$ so as to maximize the return.
    
- Markov Decision Process(MDP)
    
    在状态 $s_t$ 时，采取动作 $a_t$ 后的状态 $s_{t+1}$ 和收益 $R_{t+1}$ 只与当前状态和动作有关，与历史状态无关
    
    ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20116.png)
    
- Reinforcement Learning
    - **State-action value function** definition
        
        (价值函数)
        
        价值函数使用期望对未来的收益进行预测，一方面不必等待未来的收益实际发生就可以获知当前状态的好坏，另一方面通过期望汇总了未来各种可能的收益情况。使用价值函数可以很方便地评价不同策略的好坏。
        
        Reward Signal 定义的是评判一次交互中的立即的（immediate sense）回报好坏。而Value function 则定义的是从长期看 action 平均回报的好坏。
        
        $Q(s,a) = \text{return}$ if:
        
        - start in state $s$
        - take action $a$
        - then behave optimally after that
        
        The best possible return from state $s$ is $\underset{a}{\max}\ Q(s, a)$
        The best possible action in state $s$ is the action $a$ that gives $\underset{a}{\max}\ Q(s, a)$
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20117.png)
        
    - Bellman Equation
        - notation
        $s$: current state, $R(s)$ = reward of current state, $a$: current action,
        $s'$: state you get to after taking action $a$
        $a'$: action that you take in state $s'$
        - Bellman Equation (动态规划方程)
        $Q(s, a) = R(s) + \gamma\ \underset{a'}{\max}\ Q(s', a')$
        $R(s)$: immediate reward, reward you get right away
        $\underset{a'}{\max}\ Q(s', a')$: Return from behaving optimally starting from state $s'$
            
            $$
            \begin{align*}
            \text{Return} &= R_1+\gamma R_2+\gamma^2 R_3+\gamma^3 R_4+\ldots \\
                          &= R_1+\gamma (R_2+\gamma R_3 + \gamma^2 R_4 +\ldots)
            \end{align*}
            
            $$
            
    - Random(stochastic) environment
        
        stochastic: 随机的
        
        with probability of going in the wrong direction
        
        $$
        \begin{align*}
        \text{Expected Return}&=\text{Average}(R_1+\gamma R_2+\gamma^2 R_3+\gamma^3 R_4+\ldots)\\
                      &=E[R_1+\gamma R_2+\gamma^2 R_3+\gamma^3 R_4+\ldots]
        \end{align*}
        
        $$
        
        Bellman Equation
        $Q(s, a) = R(s) + \gamma\ E[\underset{a'}{\max}\ Q(s', a')]$
        
    - Continuous state applications
        
        Car: $s = \begin{bmatrix}
        x & y   & \theta & \dot{x} & \dot{y}& \dot{\theta} 
        \end{bmatrix}$
        
        Helicopter: $s = \begin{bmatrix}
        x & y & z & \phi & \theta & \omega & \dot{x} & \dot{y} & \dot{z} & \dot{\phi} & \dot{\theta} & \dot{\omega}
        \end{bmatrix}$
        
    - Deep Reinforcement Learning
        
        Instead of inputing a state and output an action, we input a state action pair and try to output $Q(s,a)$.
        
        Use a neural network inside the reinforcement learning algorithm to learn the $Q(s,a)$.
        
        ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20118.png)
        
    - Learning Algorithm
        
        Initialize neural network randomly as guess of $Q(s, a)$
        
        Repeat {
        
        Take actions in the lunar lander and get tuples $(s, a,R(s),s')$ as examples.
        
        Store 10,000 most recent $(s,a,R(s),s')$ tuples
        
        Train neural network:
        
        Create training set of 10,000 examples using $x=(s,a)$ and $y = R(s) +\gamma\ \underset{a'}{\max}\ Q(s', a')$
        
        Train $Q_{new}$ such that $Q_{new}(s,a) ≈ y$
        
        Set $Q = Q_{new}$ 
        
        (repeatly try to improve the estimation of the $Q$ function)
        
        }
        
    - Algorithm refinement
        - Improved neural network architecture
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20119.png)
            
        - $\varepsilon$-greedy policy
            
            How to choose actions while still learning?
            
            In some state $s$
            
            - Option 1:
            Pick the action a that maximizes $Q (s, a)$.
            - Option 2 (better):
            With probability 0.95, pick the action $a$ that maximizes $Q(s,a)$. **Greedy** or **“Exploitation”**
            With probability 0.05, pick an action $a$ randomly. “**Exploration”**
            
            Option 2 is called $\varepsilon$-greedy policy with $\varepsilon = 0.05$
            
            Useful usage: state $\varepsilon$ high and gradually decrease
            
        - Mini-batch
            
            An idea that can both speed up reinforcement learning algorithm and supervised learning algorithm as well
            
            If we have a very large training set, for every iteration, the algorithm just look at just a subset of the data to make each iteration runs much more quickly.
            
            ![Untitled](Machine%20Learning%20Specialization%2086f3d355a4414ed4a08b628146ce1c1d/Untitled%20120.png)
            
        - soft update
            
            soft update method helps to prevent $Q_{new}$ to get worse through just one unlucky step.
            
            Rather than set $Q = Q_{new}$, we can set $Q = 0.01Q_{new}+0.99Q$
            
- Limitations of Reinforcement Learning
    - Much easier to get to work in a simulation than a real robot.
    - Far fewer applications than supervised and unsupervised learning.