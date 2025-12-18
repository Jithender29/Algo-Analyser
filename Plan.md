Phase-1 / First-Half Algorithms to Include
1. Core Sorting Algorithms (Foundational Visualizers)
These are explicitly identified as the entry point of the implementation roadmap and serve as the pedagogical and architectural foundation.
Bubble Sort
Selection Sort
Insertion Sort
These algorithms support:
Step-by-step execution
Operation counting
Best / worst / average case demonstrations
Early Big-O empirical estimation

2. Database Indexing and Storage Algorithms
These algorithms introduce non-trivial data structures while remaining deterministic and highly visual.
Tree-Based Index Structures
B-Tree
B+ Tree
Key concepts supported:
Branching factor
Node splitting and promotion
Tree height vs disk I/O
Range query behavior (B+ Tree advantage)
Write-Optimized Indexing
Log-Structured Merge (LSM) Tree
Core behaviors:
MemTable buffering
SSTable flushing
Compaction and merge operations
Write amplification analysis
Multidimensional Indexing
R-Tree
Supported visual parameters:
Spatial data distribution (uniform, skewed, Gaussian)
Minimum Bounding Rectangles (MBRs)
Overlap and query degradation

3. Network Routing Algorithms
These algorithms introduce graph-based computation and iterative relaxation, suitable once tree and array visualizations are stable.
Shortest Path Algorithms
Dijkstra’s Algorithm
Bellman–Ford Algorithm
Key educational contrasts:
Priority queue vs full-edge relaxation
Sparse vs dense graph performance
Negative edge weight handling
Time complexity comparison (O(E + V log V) vs O(VE))
Inter-Domain Routing
Path Vector Algorithm (BGP-style abstraction)
Focus areas:
Path propagation instead of scalar distance
Routing table growth
Policy-aware routing visualization

4. (Optional but Early-Compatible) Graph Infrastructure
While not standalone algorithms, these are essential for the above routing algorithms and should be considered part of the first-half scope:
Graph construction and editing
Edge weight manipulation
Dynamic edge failure simulation
Iterative algorithm replay