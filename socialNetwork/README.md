# Social Network Benchmark
The idea of this benchmark is to take a social network of users with a list of images they have posted, and find all instances of some label in the N nearest neighbors of some user. The graph is stored as an adjacency matrix and the users have a list of images (loaded into memory). The algorithm is an iterative N-nearest neighbor search using matmuls on the adjacency matrix. The outputs are then fed to independent resnet50 tasks that look for the label. Finally, there is an aggregation step that searches all the labels returned from the resnet instances.
 
## Resnet Wrapper
Resnet code is provided via resnetWrapper.py. This provides a class called ResnetHandle that manages most of the KaaS and TVM stuff and interacts with a pool created by the user. See the example at the bottom of that file for details.

Installation: You must run generateModel.py in the resnetModel/ directory before running anything.
