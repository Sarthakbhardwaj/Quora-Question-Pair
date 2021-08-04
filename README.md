# Quora-Question-Pair
A machine Learning Model to determine the similarity of two samples fo text

In this project,using Semantic Textual Similarity, we determine whether a pair of Quora Questions mean the same or not.

# STEP 1: Basic Data Exploratory Analysis
# STEP 2: Data Preprocessing
The following steps are performed on each of the sentences-
-Tokenisation
-Puncuation removed
-Replacing numbers
-Replacing stopwords

# STEP 4: Building model and reducing dimension using CCA
In the Word2Vec model, the model matrix has a dimension of V×N, where V is the size of unique vocabulary and the size of N is the number of neurons in a network. 300 neurons is what Google used in their published model trained on the Google news dataset. Thus we have 300 dimensions, which is why we need dimension reduction. CCA is great for scenarios where there are two high dimensional datasets from the same samples and it enables learning looking at the datasets simultaneously. By doing CCA, we can identify the common variation, the canonical variates that are highly correlated to the unknown latent variable.
# STEP 5: Cosine Similarity
Cosine similarity between the two vectors is generated, and a custom function is made which classifies the pair of sentences as duplicate if **cosine similarity** is greater than 70% (0.7). Else, they are classified as non-duplicate.

***




