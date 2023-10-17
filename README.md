# PrivAGM: Secure Construction of Differentially Private Directed Attributed Graph Models on Decentralized Social Graphs


This repository contains code for PrivAGM, which is a novel solution that combines differential privacy, secure multiparty computation, and generative graph models to construct differentially private directed attributed graph models on decentralized social graphs while safeguarding individual privacy.

## Features

- Privacy-preserving neighbor list collection: balancing privacy, utility, and efficiency, we employ selective MPC with RNL encryption, ensuring pure edge LDP.
- Differentially private characteristics extraction: oblivious edge clipping and MPC-based secure components mitigate sensitivity issues, enabling the extraction of essential characteristics.
- Effective capture of edge directionality: we extend the Chung-Lu model for directed graphs, adapting the AGM framework to account for directionality, leading to the dAGM.
  
## Getting Started

To get started with this code, you can simply clone the repository.

The MOOC user action dataset represents the actions taken by users on a popular MOOC platform. The actions are represented as a directed, attributed network.
Download and preprocess MOOC:

Download the [MOOC user action dataset](https://snap.stanford.edu/data/act-mooc.html) and place the dataset in Datasets/mooc/.

The Twitter network represents Twitter social network

Download the [Twitter dataset](https://snap.stanford.edu/data/ego-Twitter.html) and place the dataset in Datasets/twitter/.

The gplus dataset represents the gplus social network. 

Download the [Gplus dataset](https://snap.stanford.edu/data/ego-Gplus.html) and place the dataset in Datasets/gplus/.



## Execution Environment
We used Ubuntu 20.04.3 LTS with python 3.6.5.

## External Libraries used in our source code:

-sys: This is a built-in module in Python that provides access to system-specific parameters and functions. It is commonly used for system-level operations and interactions.


-time: Another built-in module, time provides various time-related functions, such as measuring time intervals or adding delays in your code.

-numpy (imported as np): NumPy is a popular library for numerical computing in Python. It provides efficient implementations of mathematical operations on arrays or matrices, making it useful for scientific and numerical applications.

-math: This is a built-in module in Python that provides mathematical functions and constants. It includes functions like sin, cos, sqrt, etc.

-numba (imported as cuda): Numba is a just-in-time (JIT) compiler for Python that specializes in optimizing code execution, particularly for numerical computations. The cuda module within Numba provides GPU-accelerated computing capabilities.

- PySyft is a Python library for secure and private machine learning using techniques such as federated learning, homomorphic encryption, and differential privacy. It is built on top of PyTorch and provides tools and functionalities for secure and privacy-preserving computations on distributed data.


## Installing dependencies (Ubuntu)
### Installing GMP
- Get gmp-6.2.0.tar.lz at https://gmplib.org/
- Unpack the tar archive and enter the resulting folder
- Install m4 if necessary: `sudo apt-get install m4`
- Run `./configure`
- Run `make`
- Check if it was successful using `make check`
- Install using `sudo make install`

### Installing NTL
- Get ntl-11.4.3.tar.gz at https://www.shoup.net/ntl/download.html
- Unpack the tar archive and enter the resulting folder
- Enter src
- Run `./configure`
- Run `make`
- Check if it was successful using `make check`
- Install using `sudo make install`

# There are three ways of running computation:

1. Separate compilation and execution. This is the default in the
   further documentation. It allows to run the same program several
   times while only compiling once, for example:

   ```
   ./compile.py <program> <argument>
   Scripts/mascot.sh <program>-<argument> [<runtime-arg>...]
   Scripts/mascot.sh <program>-<argument> [<runtime-arg>...]
   ```

2. One-command local execution. This compiles the program and the
   virtual machine if necessary before executing it locally with the
   given protocol. The name of the protocols correspond to the script
   names below (without the `.sh`). Furthermore, some
   protocol-specific optimization options are automatically used as
   well as required options.

   ```
   Scripts/compile-run.py -E mascot <program> <argument> -- [<runtime-arg>...]
   ```

3. One-command remote execution. This compiles the program and the
   virtual machine if necessary before uploading them together with
   all necessary input and certificate files via SSH.

   ```
   Scripts/compile-run.py -HOSTS -E mascot <program> <argument> -- [<runtime-arg>...]
   ```

   `HOSTS` has to be a text file in the following format:

   ```
   [<user>@]<host0>[/<path>]
   [<user>@]<host1>[/<path>]
   ...
   ```

   If <path> does not start with `/` (only one `/` after the
   hostname), the path with be relative to the home directory of the
   user. Otherwise (`//` after the hostname it will be relative to the
   root directory.

Even with the integrated execution it is important to keep in mind
that there are two different phases, the compilation and the run-time
phase. Any secret data is only available in the second phase, when the
Python compilation has concluded. Therefore, the types like `sint` and
`sfix` are mere placeholders for data to be used later, and they don't
contain any shares. See also [the
documentation](https://mp-spdz.readthedocs.io/en/latest/compilation.html#compilation-vs-run-time)
for what this means when using Python data structures and Python
language features.

## Performance Benchmark
To measure the performance, run `main.py`

```bash
python3 main.py
```

## Testing locally

To run the entire system locally, start as `fileIO.py`.
Modify the parameters in the corresponding config files to run with different settings (e.g. privacy budget $\varepsilon$, privacy parameter $\delta$ and the datasets).

