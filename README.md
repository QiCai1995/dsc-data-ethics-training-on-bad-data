# Garbage In, Garbage Out

## Introduction

Machine learning models can seem inherently trustworthy and impartial compared to human decision-makers, but the quality of the data they are trained on has a major impact on their objectivity.  This raises ethical concerns, including concerns of algorithmic bias.

## Objectives

You will be able to:

* Define algorithmic bias
* Describe how the quality of training data impacts the quality of machine learning models
* Describe the ethical considerations surrounding data quality

## GIGO: Garbage In, Garbage Out

The phrase "garbage in, garbage out" (GIGO) [has been attributed](https://www.atlasobscura.com/articles/is-this-the-first-time-anyone-printed-garbage-in-garbage-out) to various computer scientists of the 20th century, and the underlying concept dates back to Charles Babbage (1791-1871), the ["father of computing"](https://cse.umn.edu/cbi/who-was-charles-babbage).

The idea of GIGO seems fairly obvious and intuitive: if "bad" data is entered into a computational system, then the output is going to be correspondingly "bad". So why does this even need to be stated?

Consider this 1864 quote from Charles Babbage's [_Passages from the Life of a Philosopher_](https://www.gutenberg.org/ebooks/57532):

> On two occasions I have been asked [by members of Parliament],—“Pray, Mr. Babbage, if you put into the machine wrong figures, will the right answers come out?”

The "machine" Babbage described in that quote was "Difference Engine No. 1" -- essentially a mechanical calculator that was a distant ancestor of modern computers:

<img src="https://www.gutenberg.org/files/57532/57532-h/images/i-ii.jpg" width="300px">

<p><small>Image credit&nbsp;<a href="https://www.gutenberg.org/ebooks/57532">Project Gutenberg</a></small></p>

To a modern eye, this machine might not look particularly sophisticated, especially compared to current technology. But to political leaders of the day, this machine seemed borderline magical, to the point that they wondered if they could enter "bad" data into the machine and still get "good" data as a result.

This might seem like a silly 19th Century idea with no relevance to today. But recently more and more arguments have begun to appear stating that ["smart machines will be less biased than humans"](https://www.ge.com/news/reports/will-smart-machines-be-less-biased-than-humans) and ["machines are less biased than people"](https://www.verdict.co.uk/ai-and-bias/). The underlying assumption seems to be that even though machine learning algorithms are presented with the same information that humans are presented with, something about the scale and the quality of the algorithms will allow them to overcome data quality issues. The machines have changed, but the suggestion that we can get "right answers" despite garbage input has persisted.

As a data professional it is important to be aware of some fundamental issues that can be present in data and _cannot_ be overcome simply by using big data and advanced algorithms.

## Algorithmic Bias

Most of the time in the context of machine learning, [_bias_](https://en.wikipedia.org/wiki/Bias_of_an_estimator) refers to the difference between the predicted target values and their true values. The bias-variance tradeoff involves getting the lowest possible bias while also ensuring that the model's performance is generalizable to unseen data. We might also call this type of bias "estimator bias".

By contrast [***algorithmic bias***](https://en.wikipedia.org/wiki/Algorithmic_bias) refers to unfair, discriminatory outcomes resulting from algorithmic decisions. Originally these biased algorithms were primarily programmed by hand, such as the [system](https://spectrum.ieee.org/untold-history-of-ai-the-birth-of-machine-bias) used at St. George's Hospital Medical School in the early 1980s that docked applicants 3 points for being female and 15 points for being non-European. Nowadays these decisions are more likely to be made by a machine learning model, where the programmer is simply applying an algorithm to a dataset and not making individual decisions about how the features should be weighted.

Estimator bias and algorithmic bias may or may not overlap. On the one hand, discriminatory models can be a major cause of estimator bias. A [2019 study](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf) found that even though images of darker-skinned females made up about 1/5 of the dataset, they constituted about 2/3 of the classification error in commercial computer vision models. On the other hand, the program at St. George's matched outcomes from human assessors 90-95% of the time, meaning that the program "correctly" rejected applicants on the basis of gender and nationality. Thus building a model with strong performance is not enough to ensure that you aren't perpetuating algorithmic bias.

One of the best ways to avoid algorithmic bias is by ensuring that models are trained with high-quality data, following the logic of "garbage in, garbage out".

## Issues with Data Quality

Machine learning models tend to be trained on a set of features (columns) with multiple labeled samples (rows). Quality issues can emerge with either or both of these aspects of the data; inaccurate feature values can cause the model to learn relationships that don't reflect reality, and poor sampling can lead the model to learn relationships that don't generalize.

Even for datasets with accurate features and high-quality sampling, ethical issues may arise because of the past human decisions that went into creating the data in the first place.

Below we discuss all three of these types of issues.

### Inaccurate Features

Data professionals might assume that healthcare datasets are unlikely to have inaccurate features, since many of the factors being recorded are "objective" and medical mistakes can cause very high-stakes problems. However we actually find that these datasets frequently contain bad data that can lead to algorithmic bias.

First, while [efforts are being made](https://bmjopen.bmj.com/content/3/5/e002406) to improve data entry processes, ***data entry errors*** are common in healthcare datasets, "with some cancer databases recording error rates as high as almost 27%." Any time that a person is responsible for entering data into a computer system, there is a possibility for this type of data inaccuracy simply by mistake.

A more insidious source of quality issues is the ***bias of the people who entered the data***. For example, [studies have demonstrated](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4843483/) that many non-Black medical professionals believe that Black patients feel less pain than others, and they therefore under-rate Black patient pain in records. This has led to widespread under-treatment for this pain in a clinical setting, and any data analysis of pain ratings is likely to be contaminated by this bias.

Finally, even if a model is based on raw patient data directly from some kind of sensor device, there may be ***bias built into the measurement device itself***. For example, medical devices designed for an "average" person might not work correctly for [obese patients](https://www.nytimes.com/2016/09/26/health/obese-patients-health-care.html) or [patients with darker skin tones](https://annalsofintensivecare.springeropen.com/articles/10.1186/s13613-021-00974-7). This could cause data to be skewed or even cut off at a certain value.

#### Addressing Inaccurate Features

Unfortunately if you have detected any substantial issues with feature accuracy in your dataset then there is not much that can be done from a machine learning perspective. You might be able to work with a subject-matter expert to adjust the values of features in certain narrow circumstances, but typically you simply won't be able to build a useful model from this kind of data.

### Unrepresentative Samples

<img src="https://images.prismic.io/sketchplanations/f2fdb7cb-f126-4897-ad78-4fd11c743172_SP+723+-+Sampling+bias.png" width="500px">

<p><small>Image credit&nbsp;<a href="https://sketchplanations.com/sampling-bias">Sketchplanations</a></small></p>

***Sampling bias*** is another common factor that leads to algorithmic bias. Often, data is collected in ways that leave out various groups intentionally or unintentionally. As supervised learning relies heavily on "ground truth" training data, models become skewed when the data they’re trained on are insufficiently representative.

***Representative*** datasets are those that represent each group well; it goes beyond balance to ensure each group receives somewhat fair predictions by being large enough in sample size. This can mean across gender, race, age, and other sensitive features. Representation has a major impact on how well a model performs, but is easily overlooked.

A challenge for assessing this aspect of data quality is the lack of consensus on what a "representative" dataset really is. Some say racial breakdowns should follow the demographics of a country where a specific tool will be used. Other advocate reducing the sample size for dominant groups to make better predictions for marginalized groups. There is no cookie cutter ratio for deciding how much data is representative, but best practices include consulting social and behavioral scientists to aid in this decision-making process.

#### Addressing Unrepresentative Samples

Increasing representation can leverage many tactics. The most straightforward -- but also most expensive -- is to ***collect additional data*** in a way that intentionally gathers a representative sample. The field of market research has [good resources](https://www.qualtrics.com/experience-management/research/representative-samples/) on sampling strategies that help to produce a more representative sample.

Some machine learning teams rely on existing data for training their models and thus have to rely on computational techniques instead. For example, ***resampling*** techniques developed for imbalanced datasets can also be used to try to improve representation. ***Regularization*** techniques have also [been used to address prejudices](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3).

### Reaffirming Past Prejudices

While linear models are popular for various reasons, they also have limitations. Often, we build models as little experiments to study how our predictions work on a new dataset. There are various models that work poorly on certain types of distributions in datasets. For example, a linear model would be a poor choice if a dataset had had two naturally occurring, diverging groups.  

ML models exhibit unfairness quite frequently, but identifying this unfairness isn’t always straightforward. Let's take a dataset with naturally high imbalance like arrest records in policing In various parts of the world some racial groups are survilled and policed more often than others creating datasets that broadly look like some racial groups have higher criminal activity. On the basis of various kinds of crimes, studies have shown that the propensity for crime is pretty even across races, even though that’s not what it looks like in a dataset. 

With a dataset heavily influenced by the patterns repeated by law enforcement, it’s hard to create a fair model as the underlying data itself is not fair. On the surface we should strongly consider if we can make a fair classifier using policing data, 

Many would consider using policing data for machine learning applications is unethical from the start considering how skewed the underlying data is, and the concept misalignment which assumes that simple arrest records mean someone has commit a crime, and the overall use case that crime is predictable when in fact, predictive policing tools predict policing. If an ML system is tasked with predicting who the police should arrest next, it would suggest they arrest similar people to ones they’ve arrested in the past, regardless of participation in a crime. 

When we think about the goals of machine learning, at the core we want to discriminate or classify groups based on previously identified patterns. This intentional grouping or separating of groups can be enacted in ways that 

For example, if there’s a dataset of employees and a company wants to find similar employees, we make various assumptions about what a model trained on this data can do. 

First, we assume that the employees hired are successful in their roles. While one can assume that since they have not been terminated from their positions they must be successful. This may be true, but inevitably doesn’t capture employees that are not meeting expectations. One work-around to this assumption that some orgs do is have managers highlight star employees and those meeting expectations. However, this again skews training data towards those well liked by their managers. This process doesn’t scale and is likely to encode subjective human biases, as managers alone can determine who is a good employee.

Second, we assume we want to hire employees who are similar to our newly categorized 
Successful employees. While looking for characteristics such as honesty or adaptability may not encapsulate human bias, seeking to find candidates with similar aptitude scores, personality types, or interests is likely to pose ethical issues. For a company to meet its goals, it's unclear that hiring candidates similar to their existing employees will achieve or advantage their effort. .

## External Reading

* A [short video](https://youtu.be/TWWsW1w-BVo) on the Gender Shades project that identified differing error rates based on gender and race
* [An Introduction to Sampling Bias](https://www.scribbr.com/methodology/sampling-bias/)
* [The Impact of Data Preparation on the Fairness of Software Systems](https://arxiv.org/pdf/1910.02321.pdf)
* [Datasets Have Worldviews](https://pair.withgoogle.com/explorables/dataset-worldviews/)
* [The Social Cost of Strategic Classification](https://arxiv.org/abs/1808.08460)
* [The Quest for Ethical Artificial Intelligence](https://www.youtube.com/watch?v=b_--xrN3eso)
