Click Here !! [DEMO](http://soundlabc.herokuapp.com)

### To get started ... 
Choose a dataset you want to visualize. You can either choose the MNIST images 
(3000 image samples) or facial images for 8 female K-pop artists(100 samples per each).The Scatter plot 
above is the result of running the t-SNE algorithm, on image data resulting in 
a 3D visualization of the embedded features. For demo purposes, all the data 
were pre-generated using limited number of input parameters, and displayed 
instantly. The sections below will go over how the t-SNE Explorer works.


### Abstract
The goal of this project is to practice and implement dimensional reduction on image data with high dimensional features.

The features are extracted through VGG, which is a CNN architecture. Among various dimensional reduction techniques, 
Principle Component Analysis(PCA) is known for its statistical approaches to identify principle components within data points through orthogonal transformation. 
In order to visualize the features of each datapoints, t-SNE is implemented for its ability to preserve local structure and high interpretability in visual product.
Plotly, a python graphing library with interactive output, and Dash, a python framework for web application are implemented to host this mini project.
 

### How does t-SNE work?
Images can be seen as long vectors of hundred or thousands of dimensions, 
each dimension representing one shade of color, or of gray (if the image 
is black & white). For example, a 28x28 image (such as MNIST) can be unrolled 
into one 784-dimensional vector. Kept this way, it would be extremely hard to 
visualize our dataset, especially if it contains tens of thousands of samples; 
in fact, the only way would be to go through each and every image, and keep 
track of how they are all written. 

The t-SNE algorithm solves this problem by reducing the number of dimensions 
of your datasets, so that you can visualize it in low-dimensional space, 
i.e. in 2D or 3D. For each data point, you will be able to observe its position on 
your 3D plot, which can be compared with other data points to understand how 
close or far apart they are from each other.  

### Choosing the right parameters
The quality of a t-SNE visualization depends heavily on the input parameters when you train the algorithm. Each parameter has a great impact on how well each group of data will be clustered. Here is what you should know for each of them:
- **Number of Iterations:** This is how many steps you want to run the algorithm. A higher number of iterations often gives better visualizations, but more time to train.
- **Perplexity:** This is a value that influences the number of neighbors that are taken into account during the training. According to the [original paper](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf), the value should be between 5 and 50.
- **Learning Rate:** This value determines how much weight we should give to the updates given by the algorithm at each step. It is typically between 10 and 1000.
- **Initial PCA Dimensions:** Because the number of dimensions of the original data might be very big, we use another [dimensionality reduction technique called PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to first reduce the dataset to a smaller space, and then apply t-SNE on that space. Initial PCA dimensions of 50 has shown good experimental results in the original paper.





### Participants

- **Wonyoung Seo**
- **Jiyoon Cha**
- **Pyeongwon Seo**
- **Jinwoo Oh**

For more information visit our [Notion](https://www.notion.so/Documentation-6fd5abe0e947489a9be98ede3678fb68) page!