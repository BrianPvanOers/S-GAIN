# Datasets

This folder is excluded from GitHub by default. So users can import datasets with sensitive information and keep it
private. All the listed datasets are directly imported from their repositories (UCI and Keras). Optionally, these
datasets can be stored for offline access.

### Characteristics of datasets

| Dataset                                                                 | Type    | Domain           | Features    | Classes | Instances | Missing values |
|-------------------------------------------------------------------------|---------|------------------|-------------|---------|-----------|----------------|
| [Maternal Health Risk](https://doi.org/10.24432/C5DP5D)                 | Tabular | Medical          | 6           | 1       | 1014 \*   | no             |
| [Breast Cancer Wisconsin (Diagnostic)](https://doi.org/10.24432/C5DW2B) | Tabular | Medical          | 30 (+1)     | 1       | 569       | no             |
| [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H)       | Tabular | Business         | 23 (+1)     | 1       | 30000     | no             |
| [Online News Popularity](https://doi.org/10.24432/C5NS3V)               | Tabular | Business         | 58 (+2)     | 1       | 39664 \** | no             |
| [Letter Recognition](https://doi.org/10.24432/C5ZP40)                   | Tabular | Computer Science | 16          | 1       | 20000     | no             |
| [Spambase](https://doi.org/10.24432/C53G6X)                             | Tabular | Computer Science | 57          | 1       | 4601      | no             |
| [MNIST](https://doi.org/10.1109/MSP.2012.2211477)                       | Images  | Computer Science | 28 x 28 x 1 | 1       | 60000     | no             |
| [Fashion_MNIST](https://doi.org/10.48550/arXiv.1708.07747)              | Images  | Computer Science | 28 x 28 x 1 | 1       | 60000     | no             |
| [CIFAR10](https://api.semanticscholar.org/CorpusID:18268744)            | Images  | Computer Science | 32 x 32 x 3 | 1       | 50000     | no             |

\*  On the UCI repository it is listed as 1013 instances, this is incorrect.
\** On the UCI repository it is listed as 39797 instances, this is incorrect.

---

## Description of datasets

All datasets listed are single class (highlighted in bold). No multi-class datasets are included. Non-predictive
features are highlighted in italics.

### Maternal Health Risk

This is a small medical dataset without missing values. It can therefore be used to assess the quality of the imputation
method, by introducing missing values ourselves. It has 6 features, 1 label, 1013 instances, and is anonymized. More
information can be found in the reference.

#### Explanation of features

| **Variable**  | **Type**        | **Description**                           | **Units** | **Missing values** |
|---------------|-----------------|-------------------------------------------|-----------|--------------------|
| Age           | Integer         |                                           | years     | no                 |
| SystolicBP    | Integer         | Upper value of Blood Pressure             | mmHg      | no                 |
| DiastolicBP   | Integer         | Lower value of Blood Pressure             | mmHg      | no                 |
| BS            | Integer         | Blood Glucose Levels                      | mmol/L    | no                 |
| BodyTemp      | Integer         |                                           | F         | no                 |
| HeartRate     | Integer         | Resting Heart Rate                        | bpm       | no                 |
| **RiskLevel** | **Categorical** | **Predicted Risk Level during pregnancy** |           | **no**             |

Ahmed, M. (2020). Maternal Health Risk [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D.

### Breast Cancer Wisconsin (Diagnostic)

This is also a small medical dataset without missing values. It has more features than the Maternal Health Risk dataset.
It is used to predict breast cancer. It has 30 (+1 non-predictive) features, 1 class, 569 instances, and is anonymized.
It was used in \[1]. More information can be found in the reference.

#### Explanation of features

| **Variable**       | **Type**        | **Description**                                      | **Missing values** |
|--------------------|-----------------|------------------------------------------------------|--------------------|
| *ID*               | *Categorical*   |                                                      | *no*               |
| radius1            | Continuous      | mean distance from center to points on the perimeter | no                 |
| texture1           | Continuous      | standard deviation of gray-scale values              | no                 |
| perimeter1         | Continuous      |                                                      | no                 |
| area1              | Continuous      |                                                      | no                 |
| smoothness1        | Continuous      | local variation in radius lengths                    | no                 |
| compactness1       | Continuous      | perimeter^2 / area - 1.0                             | no                 |
| concavity1         | Continuous      | severity of concave portions of the contour          | no                 |
| concave_points1    | Continuous      | number of concave portions of the contour            | no                 |
| symmetry1          | Continuous      |                                                      | no                 |
| fractal_dimension1 | Continuous      | "coastline approximation" - 1                        | no                 |
| radius2            | Continuous      | mean distance from center to points on the perimeter | no                 |
| texture2           | Continuous      | standard deviation of gray-scale values              | no                 |
| perimeter2         | Continuous      |                                                      | no                 |
| area2              | Continuous      |                                                      | no                 |
| smoothness2        | Continuous      | local variation in radius lengths                    | no                 |
| compactness2       | Continuous      | perimeter^2 / area - 1.0                             | no                 |
| concavity2         | Continuous      | severity of concave portions of the contour          | no                 |
| concave_points2    | Continuous      | number of concave portions of the contour            | no                 |
| symmetry2          | Continuous      |                                                      | no                 |
| fractal_dimension2 | Continuous      | "coastline approximation" - 1                        | no                 |
| radius3            | Continuous      | mean distance from center to points on the perimeter | no                 |
| texture3           | Continuous      | standard deviation of gray-scale values              | no                 |
| perimeter3         | Continuous      |                                                      | no                 |
| area3              | Continuous      |                                                      | no                 |
| smoothness3        | Continuous      | local variation in radius lengths                    | no                 |
| compactness3       | Continuous      | perimeter^2 / area - 1.0                             | no                 |
| concavity3         | Continuous      | severity of concave portions of the contour          | no                 |
| concave_points3    | Continuous      | number of concave portions of the contour            | no                 |
| symmetry3          | Continuous      |                                                      | no                 |
| fractal_dimension3 | Continuous      | "coastline approximation" - 1                        | no                 |
| **Diagnosis**      | **Categorical** |                                                      | **no**             |

#### Labels

| B      | M         |
|--------|-----------|
| Benign | Malignant |

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

### Default of Credit Card Clients

This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of
probability of default among six data mining methods. It has no missing values, 23 (+1 non-predictive) features, 1
class, 30000 instances, and is anonymized. It was used in \[1]. More information can be found in the reference.

#### Explanation of features

| **Variable** | **Type**   | **Description**                                                                                                                   | **Units** | **Missing values** |
|--------------|------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------|--------------------|
| *ID*         | *Integer*  |                                                                                                                                   |           | *no*               |
| X1           | Integer    | LIMIT_BAL: Amount of the given credit: it includes both the individual consumer credit and his/her family (supplementary) credit. | NT dollar | no                 |
| X2           | Integer    | Sex (1 = male; 2 = female)                                                                                                        |           | no                 |
| X3           | Integer    | Education Level (1 = graduate school; 2 = university; 3 = high school; 4 = others)                                                |           | no                 |
| X4           | Integer    | Marital Status (1 = married; 2 = single; 3 = others)                                                                              |           | no                 |
| X5           | Integer    | Age                                                                                                                               | years     | no                 |
| X6           | Integer    | PAY_0: The repayment status in September, 2005                                                                                    | *         | no                 |
| X7           | Integer    | PAY_2: The repayment status in August, 2005                                                                                       | *         | no                 |
| X8           | Integer    | PAY_3: The repayment status in July, 2005                                                                                         | *         | no                 |
| X9           | Integer    | PAY_4: The repayment status in June, 2005                                                                                         | *         | no                 |
| X10          | Integer    | PAY_5: The repayment status in May, 2005                                                                                          | *         | no                 |
| X11          | Integer    | PAY_6: The repayment status in April, 2005                                                                                        | *         | no                 |
| X12          | Integer    | BILL_AMT1: Amount of bill statement in September, 2005                                                                            | NT dollar | no                 |
| X13          | Integer    | BILL_AMT2: Amount of bill statement in August, 2005                                                                               | NT dollar | no                 |
| X14          | Integer    | BILL_AMT3: Amount of bill statement in July, 2005                                                                                 | NT dollar | no                 |
| X15          | Integer    | BILL_AMT4: Amount of bill statement in June, 2005                                                                                 | NT dollar | no                 |
| X16          | Integer    | BILL_AMT5: Amount of bill statement in May, 2005                                                                                  | NT dollar | no                 |
| X17          | Integer    | BILL_AMT6: Amount of bill statement in April, 2005                                                                                | NT dollar | no                 |
| X18          | Integer    | PAY_AMT1: Amount of previous payment in September, 2005                                                                           | NT dollar | no                 |
| X19          | Integer    | PAY_AMT2: Amount of previous payment in August, 2005                                                                              | NT dollar | no                 |
| X20          | Integer    | PAY_AMT3: Amount of previous payment in July, 2005                                                                                | NT dollar | no                 |
| X21          | Integer    | PAY_AMT4: Amount of previous payment in June, 2005                                                                                | NT dollar | no                 |
| X22          | Integer    | PAY_AMT5: Amount of previous payment in May, 2005                                                                                 | NT dollar | no                 |
| X23          | Integer    | PAY_AMT6: Amount of previous payment in April, 2005                                                                               | NT dollar | no                 |
| **Y**        | **Binary** | **default payment next month**                                                                                                    |           | **no**             |

**\* X6-X11:** -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment
delay for eight months; 9 = payment delay for nine months and above.

#### Labels

| 0  | 1   |
|----|-----|
| No | Yes |

Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H.

### Online News Popularity

This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years.
The goal is to predict the number of shares in social networks (popularity). It has no missing values, 58 (+2
non-predictive) features, 1 class, and 39797 instances. It was used in \[1]. More information can
be found in the reference.

#### Explanation of features

| **Variable**                  | **Type**      | **Description**                                                                     | **Missing values** |
|-------------------------------|---------------|-------------------------------------------------------------------------------------|--------------------|
| *url (ID)*                    | *Categorical* | *URL of the article (non-predictive)*                                               | *no*               |
| *timedelta*                   | *Continuous*  | *Days between the article publication and the dataset acquisition (non-predictive)* | *no*               |
| n_tokens_title                | Continuous    | Number of words in the title                                                        | no                 |
| n_tokens_content              | Continuous    | Number of words in the content                                                      | no                 |
| n_unique_tokens               | Continuous    | Rate of unique words in the content                                                 | no                 |
| n_non_stop_words              | Continuous    | Rate of non-stop words in the content                                               | no                 |
| n_non_stop_unique_tokens      | Continuous    | Rate of unique non-stop words in the content                                        | no                 |
| num_hrefs                     | Continuous    | Number of links                                                                     | no                 |
| num_self_hrefs                | Continuous    | Number of links to other articles published by Mashable                             | no                 |
| num_imgs                      | Continuous    | Number of images                                                                    | no                 |
| num_videos                    | Continuous    | Number of videos                                                                    | no                 |
| average_token_length          | Continuous    | Average length of the words in the content                                          | no                 |
| num_keywords                  | Continuous    | Number of keywords in the metadata                                                  | no                 |
| data_channel_is_lifestyle     | Continuous    | Is data channel 'Lifestyle'?                                                        | no                 |
| data_channel_is_entertainment | Continuous    | Is data channel 'Entertainment'?                                                    | no                 |
| data_channel_is_bus           | Continuous    | Is data channel 'Business'?                                                         | no                 |
| data_channel_is_socmed        | Continuous    | Is data channel 'Social Media'?                                                     | no                 |
| data_channel_is_tech          | Continuous    | Is data channel 'Tech'?                                                             | no                 |
| data_channel_is_world         | Continuous    | Is data channel 'World'?                                                            | no                 |
| kw_min_min                    | Continuous    | Worst keyword (min. shares)                                                         | no                 |
| kw_max_min                    | Continuous    | Worst keyword (max. shares)                                                         | no                 |
| kw_avg_min                    | Continuous    | Worst keyword (avg. shares)                                                         | no                 |
| kw_min_max                    | Continuous    | Best keyword (min. shares)                                                          | no                 |
| kw_max_max                    | Continuous    | Best keyword (max. shares)                                                          | no                 |
| kw_avg_max                    | Continuous    | Best keyword (avg. shares)                                                          | no                 |
| kw_min_avg                    | Continuous    | Avg. keyword (min. shares)                                                          | no                 |
| kw_max_avg                    | Continuous    | Avg. keyword (max. shares)                                                          | no                 |
| kw_avg_avg                    | Continuous    | Avg. keyword (avg. shares)                                                          | no                 |
| self_reference_min_shares     | Continuous    | Min. shares of referenced articles in Mashable                                      | no                 |
| self_reference_max_shares     | Continuous    | Max. shares of referenced articles in Mashable                                      | no                 |
| self_reference_avg_sharess    | Continuous    | Avg. shares of referenced articles in Mashable                                      | no                 |
| weekday_is_monday             | Continuous    | Was the article published on a Monday?                                              | no                 |
| weekday_is_tuesday            | Continuous    | Was the article published on a Tuesday?                                             | no                 |
| weekday_is_wednesday          | Continuous    | Was the article published on a Wednesday?                                           | no                 |
| weekday_is_thursday           | Continuous    | Was the article published on a Thursday?                                            | no                 |
| weekday_is_friday             | Continuous    | Was the article published on a Friday?                                              | no                 |
| weekday_is_saturday           | Continuous    | Was the article published on a Saturday?                                            | no                 |
| weekday_is_sunday             | Continuous    | Was the article published on a Sunday?                                              | no                 |
| is_weekend                    | Continuous    | Was the article published on the weekend?                                           | no                 |
| LDA_00                        | Continuous    | Closeness to LDA topic 0                                                            | no                 |
| LDA_01                        | Continuous    | Closeness to LDA topic 1                                                            | no                 |
| LDA_02                        | Continuous    | Closeness to LDA topic 2                                                            | no                 |
| LDA_03                        | Continuous    | Closeness to LDA topic 3                                                            | no                 |
| LDA_04                        | Continuous    | Closeness to LDA topic 4                                                            | no                 |
| global_subjectivity           | Continuous    | Text subjectivity                                                                   | no                 |
| global_sentiment_polarity     | Continuous    | Text sentiment polarity                                                             | no                 |
| global_rate_positive_words    | Continuous    | Rate of positive words in the content                                               | no                 |
| global_rate_negative_words    | Continuous    | Rate of negative words in the content                                               | no                 |
| rate_positive_words           | Continuous    | Rate of positive words among non-neutral tokens                                     | no                 |
| rate_negative_words           | Continuous    | Rate of negative words among non-neutral tokens                                     | no                 |
| avg_positive_polarity         | Continuous    | Avg. polarity of positive words                                                     | no                 |
| min_positive_polarity         | Continuous    | Min. polarity of positive words                                                     | no                 |
| max_positive_polarity         | Continuous    | Max. polarity of positive words                                                     | no                 |
| avg_negative_polarity         | Continuous    | words                                                                               | no                 |
| min_negative_polarity         | Continuous    | words                                                                               | no                 |
| max_negative_polarity         | Continuous    | words                                                                               | no                 |
| title_subjectivity            | Continuous    | Title subjectivity                                                                  | no                 |
| title_sentiment_polarity      | Continuous    | Title polarity                                                                      | no                 |
| abs_title_subjectivity        | Continuous    | Absolute subjectivity level                                                         | no                 |
| abs_title_sentiment_polarity  | Continuous    | Absolute polarity level                                                             | no                 |
| **shares**                    | **Integer**   | **Number of shares (target)**                                                       | **no**             |

**Labels:** The number of shares

Fernandes, K., Vinagre, P., Cortez, P., & Sernadela, P. (2015). Online News Popularity [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5NS3V.

### Letter Recognition

This is a dataset used for classifying capital letters in the english alphabet. It has no missing values, 16 features,
1 class, and 20000 instances. It was used in \[1]. More information can be found in the references.

#### Explanation of features

| **Variable** | **Type**        | **Description**               | **Missing values** |
|--------------|-----------------|-------------------------------|--------------------|
| **lettr**    | **Categorical** | **capital letter**            | **no**             |
| x-box        | Integer         | horizontal position of box    | no                 |
| y-box        | Integer         | vertical position of box      | no                 |
| width        | Integer         | width of box                  | no                 |
| high         | Integer         | height of box                 | no                 |
| onpix        | Integer         | total # on pixels             | no                 |
| x-bar        | Integer         | mean x of on pixels in box    | no                 |
| y-bar        | Integer         | mean y of on pixels in box    | no                 |
| x2bar        | Integer         | mean x variance               | no                 |
| y2bar        | Integer         | mean y variance               | no                 |
| xybar        | Integer         | mean x y correlation          | no                 |
| x2ybr        | Integer         | mean of x * x * y             | no                 |
| xy2br        | Integer         | mean of x * y * y             | no                 |
| x-ege        | Integer         | mean edge count left to right | no                 |
| xegvy        | Integer         | correlation of x-ege with y   | no                 |
| y-ege        | Integer         | mean edge count bottom to top | no                 |
| yegvx        | Integer         | correlation of y-ege with x   | no                 |

**Labels:** Capital letters A-Z

Slate, D. (1991). Letter Recognition [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP40.

### Spambase

This is a dataset used for classifying emails as either Spam or Non-Spam. It has 57 features, 1 class, and 4601
instances. It was used in \[1]. More information can be found in the reference.

#### Explanation of features

| Variable                   | Type       | Missing values |
|----------------------------|------------|----------------|
| word_freq_make             | Continuous | no             |
| word_freq_address          | Continuous | no             |
| word_freq_all              | Continuous | no             |
| word_freq_3d               | Continuous | no             |
| word_freq_our              | Continuous | no             |
| word_freq_over             | Continuous | no             |
| word_freq_remove           | Continuous | no             |
| word_freq_internet         | Continuous | no             |
| word_freq_order            | Continuous | no             |
| word_freq_mail             | Continuous | no             |
| word_freq_receive          | Continuous | no             |
| word_freq_will             | Continuous | no             |
| word_freq_people           | Continuous | no             |
| word_freq_report           | Continuous | no             |
| word_freq_addresses        | Continuous | no             |
| word_freq_free             | Continuous | no             |
| word_freq_business         | Continuous | no             |
| word_freq_email            | Continuous | no             |
| word_freq_you              | Continuous | no             |
| word_freq_credit           | Continuous | no             |
| word_freq_your             | Continuous | no             |
| word_freq_font             | Continuous | no             |
| word_freq_000              | Continuous | no             |
| word_freq_money            | Continuous | no             |
| word_freq_hp               | Continuous | no             |
| word_freq_hpl              | Continuous | no             |
| word_freq_george           | Continuous | no             |
| word_freq_650              | Continuous | no             |
| word_freq_lab              | Continuous | no             |
| word_freq_labs             | Continuous | no             |
| word_freq_telnet           | Continuous | no             |
| word_freq_857              | Continuous | no             |
| word_freq_data             | Continuous | no             |
| word_freq_415              | Continuous | no             |
| word_freq_85               | Continuous | no             |
| word_freq_technology       | Continuous | no             |
| word_freq_1999             | Continuous | no             |
| word_freq_parts            | Continuous | no             |
| word_freq_pm               | Continuous | no             |
| word_freq_direct           | Continuous | no             |
| word_freq_cs               | Continuous | no             |
| word_freq_meeting          | Continuous | no             |
| word_freq_original         | Continuous | no             |
| word_freq_project          | Continuous | no             |
| word_freq_re               | Continuous | no             |
| word_freq_edu              | Continuous | no             |
| word_freq_table            | Continuous | no             |
| word_freq_conference       | Continuous | no             |
| char_freq_;                | Continuous | no             |
| char_freq_(                | Continuous | no             |
| char_freq_[                | Continuous | no             |
| char_freq_!                | Continuous | no             |
| char_freq_$                | Continuous | no             |
| char_freq_#                | Continuous | no             |
| capital_run_length_average | Continuous | no             |
| capital_run_length_longest | Continuous | no             |
| capital_run_length_total   | Continuous | no             |
| **Class**                  | **Binary** | **no**         |

#### Labels

| 0    | 1        |
|------|----------|
| spam | not spam |

Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X.

### MNIST

This is a popular image dataset consisting of handwritten digits. It has 28 x 28 x 1 features, 1 class, and 60000
instances. More information can be found in the references.

**Labels:** Digits from 0-9.

Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing
Magazine, 29(6), 141–142. https://doi.org/10.1109/MSP.2012.2211477

TensorFlow website: https://www.tensorflow.org/datasets/catalog/mnist

### Fashion_MNIST TODO

This is another popular image dataset consisting of Zalando's article images. It is favored over MNIST, because of its
higher complexity, and is intended to be a drop in replacement. It has 28 x 28 x 1 features, 1 class, and 60000
instances. More information can be found in the references.

#### Labels

| 0           | 1       | 2        | 3     | 4    | 5      | 6     | 7       | 8   | 9          |
|-------------|---------|----------|-------|------|--------|-------|---------|-----|------------|
| T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |

Han Xiao, Kashif Rasul, & Roland Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
Algorithms. CoRR. https://doi.org/10.48550/arXiv.1708.07747

TensorFlow website: https://www.tensorflow.org/datasets/catalog/fashion_mnist

Homepage: https://github.com/zalandoresearch/fashion-mnist

### CIFAR10

This is also a popular image dataset. It uses color channels in contrast to (Fashion_)MNIST datasets, which are in
greyscale. It has 32 x 32 x 3 features, 1 class, and 50000 instances. More information can be found in the
references.

#### Labels

| 0        | 1          | 2    | 3   | 4    | 5   | 6    | 7     | 8    | 9     |
|----------|------------|------|-----|------|-----|------|-------|------|-------|
| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
https://api.semanticscholar.org/CorpusID:18268744

TensorFlow website: https://www.tensorflow.org/datasets/catalog/cifar10

Homepage: https://www.cs.toronto.edu/%7Ekriz/cifar.html

---

## References

\[1] J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", ICML,
\2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf.
