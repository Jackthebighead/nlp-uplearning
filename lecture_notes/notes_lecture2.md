# Lecture 2: Word Vectors and Word Senses

- ### **A look-back of Word2vec**
    - the main idea of word2vec
        - initially start with random word vectors
        - then iterate through each word in the whole corpus
        - at last, it try to predict (in the skip-gram mode) the context using the center word.
        - the algorithm learns word vectors that capture word similarity and meaningful directions in the word space.

          <img src="pics_lecture2/Untitled.png" width = "300" height = "100" alt="d" align=center />

        - it maximizes objective function by putting similar words nearby in space.
        - why called 2vec: there are 2 model variants, Skip-grams and CBOW, word2vec just average both at the end.
    - optimization with word2vec
        - SGD: repeatedly sample windows, and update after each one.
            - input one sample and adjust parameters based on gradient descent.
            - consider there is a very big matrix of word vectors in the whole corpus, if we take gradients of a window of a certain size (let's say 5), the gradient J(\theta) on the whole corpus will be very sparse. In this way, we should update only the words appear in the window, so we only update certain rows in the matrix.
            - negative sampling
                - additional efficiency in training
                - randomly choose some words outside the window as the negative samples, then train the model and update only negative and positive words following the strategy of maximizing the probability of words within the window and minimizing the words randomly chosen.
                - speed up the process and enhance the quality.
                - paper: Distributed Representations of Words and Phrases and their Compositionality
    - can use mini-batch on GPU, e.g. sample 20 from the whole corpus.
    - shuffle before every epoch
- ### **Why not capture co-occurrence counts directly?**
    - co-occurrence matrix: two ways to construct
        - window: use window around each word to capture both positional and semantic (context) information.
        - document: use the whole corpus gives the general topic information of the word. It leads to the topic of Latent Semantic Analysis (LSA/LSI).
    - window-based co-occurrence matrix
        - window size: 5-10 commonly.
        - symmetric: irrelevant whether left or right context. otherwise it's asymmetric.
        - problems: less robust
            - large in size with large vocabulary
            - high dimension, larger storage required
            - sparsity issue
        - solution: low dimensional vectors
            - store most of the important information in a fixed number of dimensions, which is a dense vector.
            - 25-1000 dimensions commonly, similar to word2vec.
            - methods of dimension reduction
                - matrix decomposition

                    <img src="pics_lecture2/Untitled 1.png" width = "400" height = "200" alt="d" align=center />

                    - SVD: Single Value Decomposition.
                        - 奇异值分解
                        - retain only k singular best ranked values after decomposition.
                    - other methods: Hacks to X
                        - scaling the counts in the cells.
                            - problem: scale to some words like 'the', 'he', etc. that are too frequent.
                        - ramped windows that count closer words more.
                        - use Pearson correlations instead of counts.
- ### **GloVe**
    - both word2vec and GloVe captures the co-occurrence information to embed the word into a vector.
    - **count based vs. direct prediction: two ways to do word embedding**
        - **direct prediction**
            - word2vec
            - use window-based training methods are such as SG and CBOW, learning the probability, ability of capturing complex linguistic schema.
        - **count based**
            - LSA based on counts and decomposes the matrix based on SVD.
                - make use of global information.
                - more complex than GloVe.

                <img src="pics_lecture2/Untitled 2.png" width = "500" height = "200" alt="d" align=center />

    - GloVe
        - Global Vectors for Word Representation.
            - GloVe: combines the advantages of two main model families: global matrix decomposition and local context window. The model only trains the non-zero elements in the word word co-occurrence matrix, instead of the whole sparse matrix or large corpus with a single context window, so as to effectively utilize the statistical information. The model generates a vector space with meaningful substructures, and its performance in the most recent word analogy task is 75%. In terms of similarity task and named entity recognition, it is also superior to related models.
            - GloVe uses windows to do counting for each word.
            - Count based models, such as glove, essentially reduce the dimension of co-occurrence matrix and learn low-dimensional dense-vector representation.
            - GloVe pretrained model is based on Common Crawl dataset and well-preprocessed. It's more common used.
        - characteristics of GloVe
            - fast training(low cost), statistical, huge corpus, good performance even corpus and the dimension of vectors are small.
            - used to capture word similarity but disproportionate to large counts.
        - in practice, there are packages like
            - glove for GloVe
            - gensim for word2vec and glove2word2vec
            - spacy
        - the process of training GloVe model
            - **no neural network training in GloVe.**
            - encode meaning in vector differences.
            - first construct word vector and the co-occurrence matrix. use the following formula to approximately represent the relationship with the vector and the matrix.
                - $w_i^Tw_j+b_i+b_j=log(X_{ij})$
                    - X_{ij} stands for the number of appearance of j in the context of i.
                    - 比值代表a和b哪个与x更相关，说明通过概率比例而不是概率本身去学习词向量可能是一个更恰当的方法。
                        - ratios of co-occurrence probabilities can encode meaning components

                            <img src="pics_lecture2/Untitled 3.png" width = "500" height = "100" alt="d" align=center />

                            - so the form of w_i*w_j=logP(i|j) is constructed.
                    - w_i and w_j is the same intuitively in matrix decomposition, we average them as the final vector.
                    - Every word in n_dim is corresponding to a vector in k_dim, so we can use vector_k*vector_k to fit the original vector. that's the idea.
            - then construct the loss function and learn and update.
                - $w_i*w_j=logP(i|j)\\ J=\sum_{i,j=1}^{V}f(X_{ij})(w_i^Tw_j+b_i+b_j-logX_{ij})^2$
                - f(X) is a weight function used as a restriction, the parameter is set to 0.75 after many times of experiments.
                - AdaGrad for optimization
        - conclusion on GloVe: Glove uses global statistics to predict the probability of word J appearing in the context of word I with least squares as the target.
- ### **How to evaluate word vectors?**
    - general evaluation in NLP
        - Intrinsic
            - evaluation on an intermediate task
            - fast to compute
            - not sure of the utility in the real tasks
        - Extrinsic
            - evaluation on a real downstream task
            - time consuming to become accurate enough

    - the Intrinsic way
        - word vector analogies
        - evaluate word vectors by how well their cosine distance after addition captures intuitive semantic and syntactic analogy (类比) questions.
            - semantic
            - syntactic
            <img src="pics_lecture2/Untitled 4.png" width = "400" height = "60" alt="d" align=center />
            <img src="pics_lecture2/Untitled 5.png" width = "400" height = "250" alt="d" align=center />

        - problem: what if the relationship is not linear?
        - analogy evaluation and hyperparameters
            - dimension, 300 is good.
        - another way: measure word vector distances and their correlation with human judgements.
            - example: WordSim353, MC, RG, RW, SCWS. dataset.
    - the Extrinsic way
        - apply directly into a downstream task like named entity recognition.
- ### **Word senses and word sense ambiguity**
    - 歧义，词义消歧
    - cluster word windows around words, retrain with each word assigned to multiple different clusters. So as to refine different meaning of the same word.
        - so $v_{pike}=a_1v_{pike1}+a_2v_{pike_2}+a_3{v_{pike3}}$
        - thesis: 'Improving Word Representations Via Global Context And Multiple Word Prototypes'



- ### **Matetrial 1: GloVe**
    - paper name: *GloVe: Global Vectors for Word Representation*
    - links: [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)
    - **GloVe: Global Vectors for Word Representation**
        - Two main model families for learning word vectors:
        - global matrix factorization methods
            - LSA: Latent Semantic Analysis, rows correspond to words and columns correspond to the documents in the corpus and the values correspond to the tfidf value or sth based on the co-occurrence characteristics.
            - HAL: Hyperspace Analogue to Language, rows and columns correspond to words and the values correspond to the number of occurrence.
            - they utilize low-rank approximations to decompose large matrices.
            - they leverage the statistical information of the corpus but performs bad at word analogy tasks (an intrinsic evaluation of word vector quality).
            - a main problem may be that some meaningless but frequent words may contribute a bigger amount to the similarity or sth.
        - local context window methods
            - skip-gram model: learn word representations by making predictions with local context windows, train them with an objective.
            - CBOW
            - they behave well on word analogy tasks.
            - it poorly uses the statistics of the corpus (it trains on global co-occurrence counts, no global information)
    - GloVe is a weighted least squares model that trains on global word-word co-occurrence  counts. And performs better on similarity and NER tasks.
    - **the GloVe Model**
        - set up a word-word co-occurrence matrix X whose entries X_ij is the number of times word j occurs in the context (actually within a window).
        - set up the objective function, which is a weighted least squares regression model
        - $J=\sum_{i,j=1}^{V}f(X_{ij})(w_i^Tw_j+b_i+b_j-logX_{ij})^2$
            - adding f(X_ij), which is a cut-off function to avoid the overweighting problem of frequent meaningless words.
            - intuitively the learned vectors in the model: w_i and w_j, are the same because the matrix is supposed to be symmetric, but different actually because of the initialization.
        - model training
            - train by stochastically sampling non-zero elements from X and intialize a learning rate to 0.05 using AdaGrad. Run 50 iterations for vectors smaller than 300 dims and 100 iterations otherwise till convergence.
        - complexity of the model
            - no worse than O(n^2)
            - O(n^0.8), somewhat better than the on-line window-based methods which wcale like O(n).
    - Model Analysis
        - experiments on word analogy, word similarity and name entity recognition tasks, significantly better.
        - vector length and context size
            - symmetric is a context window that extends to the left and right of a target word, while the one which extends only to the left will be called asymmetric.
            - in the results below
                - diminishing returns for vectors larger than 200 dimensions.
                - performances on syntactic context is better with small window size and asymmetric
                - semantic information is more frequently non-local and easy captured with long window size.
                <img src="pics_lecture2/Untitled 6.png" width = "500" height = "200" alt="d" align=center />

            - semantic and syntactic
                - The word analogy task consists of questions like, “a is to b as c is to?” The dataset contains 19,544 such questions, divided into a semantic subset and a syntactic subset. The semantic questions are typically analogies about people or places, like “Athens is to Greece as Berlin is to?”. The syntactic questions are typically analogies about verb tenses or forms of adjectives, for example “dance is to dancing as ﬂy is to ?”.
        - corpus size

            <img src="pics_lecture2/Untitled 7.png" width = "500" height = "200" alt="d" align=center />

            - 300 dimensional vectors trained on different corpora. best overall and subtask-individually on Common Crawl dataset.
            - syntactic subtask increases as the size of corpus increases.
            - semantic subtask don't, maybe because it needs corpus full of knowledge like wikipedia.
        - Runtime
        - outperforms than Word2Vec

            <img src="pics_lecture2/Untitled 8.png" width = "500" height = "200" alt="d" align=center />

            - comparison experiments: the training time is hard to keep, for GloVe is iterations while w2v is the numbers of training epochs. the paper find adding negative samples actually increases the number of training sords and thus analogous to extra epochs.
            - test on the analogy tasks. w2v performs a drawback as the number of negative samples increases, mainly because negative sampling does not approximate the target probability distribution well.