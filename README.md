# py-dbclasd
Python implementation of DBCLASD: a non-parametric clustering algorithm

=== INTRODUCTION ===

Many recent (and not so recent) surveys on clustering algorithms, highlight the strength of this non-parametric
clustering method. As this is one of my research interests, I wanted to give it a try and compare it myself to see if
it suits my needs. Unfortunately, I haven't been able to find any implementation of this algorithm whatsoever :-(

This is why, I decided to do it myself. First to be able to test its novelties against other methods and second, because
I wanted to contribute to the scientific and CS community somehow and this seemed to me as a nice way to do it.

I've tried my best to stick to the description given in the paper, which I found to be rather confusing once you really
get into the implementation details. However, there is at least one thing that I felt needed to be done in order for
the algorithm to deliver meaningful results, namely a condition to prevent the evaluation of points whose 29 initial
nearest neighbors have been mostly labeled already (i.e., if more than half of the 29-NN of a candidate point have
labels assigned already).

I'd be happy to get feedback, specially if there are still bugs around. I tried to make the code as efficient as
possible without sacrificing readability. That means that I tried to stick to the pseudocode given in the paper as much
as possible. The code is of course, far from being efficient but I guess I can think of creating branch with an
efficient version afterwards.


=== EXECUTING DBCLASD ===

The code should run out of the box, given that you have all packages required (they're all standard):

 - NumPy
 - SciPy
 - scikit-learn
 - matplotlib

To give it a test-run, make sure you have a copy of the text file named Aggregation2d.txt which contains some sample
data (which was publicly available at http://cs.joensuu.fi/sipu/datasets/ ). Now, run the command

$ python dbclasd.py -i Aggregation2d.txt

This should load the data and perform the clustering on it. At the end, a plot of all classes is saved to the current
directory. Make sure matplotlib is properly configured and you have enough permissions to save in '.'


References:
Xiaowei Xu; Ester, M.; Kriegel, H.-P.; Sander, J., "A distribution-based clustering algorithm for mining in large
spatial databases," Data Engineering, 1998. Proceedings., 14th International Conference on , vol., no., pp.324,331,
23-27 Feb 1998 - http://www.ualr.edu/xwxu/publications/icde-98.pdf
