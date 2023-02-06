**Table of Content**
- [2023-01](#2023-01)
  - [31](#31)
- [2023-02](#2023-02)
  - [02](#02)
  - [05](#05)


The book [[*link*](https://d2l.ai/chapter_preface/index.html)]

# 2023-01
## 31
I'll start to revisit this book from today. I'll post notes everyday about the chapter I read and share some thoughts while reading this book. Hope I could finish this book by the end of March (2023) :-) 

**Content reviewed**
- Preface
- Installation
- Notation 

**Things I did**

- Removed the Anaconda package from my computer and started to use *Miniconda* as suggested [here](https://d2l.ai/chapter_installation/index.html#installing-miniconda). 
- Realized that I could automatically add conda env activation command to `.zshrc` by running the command `~/miniconda3/bin/conda init zsh` [[*source*](https://stackoverflow.com/questions/40370467/anaconda-not-found-in-zsh)]
- Refreshed my memory on `poetry` [[*link*](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment)], which is a great tool to manage project dependencies. I first tested `poetry` in my [python class](https://github.com/xiangshiyin/python-for-kids) for young kids, it's super handy.

I'll spend some time to review the preliminary chapter tomorrow.


# 2023-02
## 02

**Side topics**
- `fish` shell [[*link*](https://fishshell.com/)]
  - Enable conda in fish shell - run command `conda init fish` [[*stackoverflow*](https://stackoverflow.com/questions/34280113/add-conda-to-path-in-fish)]
  - Set `fish` to be the default shell [[*reference*](https://fishshell.com/docs/current/index.html#default-shell)]

**Content reviewed**
- 2. Preliminaries
  - 2.1 Data Manipulation
  - 2.2 Data Preprocessing

**Notes**
* The `tensor class` (ndarray in MxNet, Tensor in PyTorch and TensorFlow) resembles NumPy's ndarray, with a few killer features added (`vector --> matrix --> tensor`)
* Tensors in TensorFlow are immutable, and cannot be assigned to. Variables in TensorFlow are mutable containers of state that support assignments.
* TensorFlow provides the `tf.function` decorator to wrap computation inside of a TensorFlow graph that gets compiled and optimized before running. This allows TensorFlow to prune unused values, and to reuse prior allocations that are no longer needed. This minimizes the memory overhead of TensorFlow computations.
  ```python
  @tf.function
  def computation(X, Y):
      Z = tf.zeros_like(Y)  # This unused value will be pruned out
      A = X + Y  # Allocations will be reused when no longer needed
      B = A + Y
      C = B + Y
      return C + Y

  computation(X, Y)
  ```

## 05

**Content reviewed**
- 2. Preliminaries (Tensorflow)
  - 2.3 Linear Algebra

**Notes**
* `Scalars` are implemented as tensors that contain only one element, or 0-order tensors.
* `Vectors` are implemented as 1st-order tensors.
* Adding or multiplying a scalar and a tensor produces a result with the same shape as the original tensor. Here, each element of the tensor is added to (or multiplied by) the scalar.
* Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix `tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)`
* A smart way to understand the dot product between a $mxn$ matrix A and a $n$-dimensional vector $x$ and the dot product between 2 matrices [[*link*](https://d2l.ai/chapter_preliminaries/linear-algebra.html#matrix-vector-products)]

