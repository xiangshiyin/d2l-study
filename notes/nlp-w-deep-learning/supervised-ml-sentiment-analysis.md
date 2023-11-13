
# Supervised ML & Sentiment Analysis

## Overview
```
Text --> X --> Train LR --> Classify --> Positive: 1
```

## Represent text as a vector
### Build upon vocabulary
>"I am Happy because I am learning NLP"
--> [I, am, happy, ...]
--> [1,1,1,0,...] (1 represent present, 0 represent absent)

Length of the vector equals to $|V|$ (size of the vocabulary). As $V$ gets larger, the vector becomes more sparse. Furthermore, we end up having many more features (since we are considering the whole vocabulary), this could result in large training time and large prediction time.

### Use word frequency as features
- Separate positive and negative tweets
- Count the word frequency in positive and negative tweets --> positive/negative frequencies ![](../pics/positive_negative_freq.png)

$$
X_m = [1, \sum_{w}freqs(w,1), \sum_{w}freqs(w,0)]
$$

This way, the tweet would be represented as a vector of length 3: `[bias, sum of positive frequencies, sum of negative frequencies]`

$$
\begin{pmatrix}
    1 & X_1^{(1)} & X_2^{(1)} \\
    1 & X_1^{(2)} & X_2^{(2)} \\
    ... & ... & ... \\
    1 & X_1^{(m)} & X_2^{(m)}
\end{pmatrix}
$$


## Preprocess the text
- Remove stop words and punctuations from the tweets
- Stemming (convert to base stem) and lowercasing

    ```python
    freqs = build_freqs(tweet, labels)
    X = np.zeros((m, 3))
    for i in range(m):
        p_tweet = process_tweet(tweets[i]) #Process tweet
        X[i,:] = extract_features(p_tweet, freqs) #Extract features
    ```

## Overview of logistic regression

$$
h(x^{(i)}, \theta) = \frac{1}{1+e^{-\theta^T x^{(i)}}}
$$

## Training LR
![](../../pics/train-lr-demo.png)

```mermaid
graph TD;
    A(Initialize parameters)-->B(Classify/predict);
    B-->C(Get gradient);
    C-->D(Update)
    D-->E(Get Loss)
    E--"Until good enough"-->B
```


## Test LR
$$pred = h(X_{val}, \theta) \ge 0.5$$
- Accuracy: % of times predicted correct labels

## Cost function of LR
$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^m (y^{(i)} logh(x^{(i)} \theta) + (1-y^{(i)})log(1-h(x^{(i)}, \theta)))
$$

![](../../pics/train-cost-function.png)

A vectorized implementation is

$$
h = g(X \theta)
J(\theta) = \frac{1}{m} \cdot (-y^T log(h) - (1-y)^{T} log(1-h))
$$

## Reference
- [code sample 1](https://github.com/FahdSeddik/DeepLearning.AI-Natural-Language-Processing-Specialization/tree/main)
- [code sample 2](https://github.com/amanjeetsahu/Natural-Language-Processing-Specialization)




