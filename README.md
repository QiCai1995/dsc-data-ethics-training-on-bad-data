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

This might seem like a silly 19th Century idea with no relevance to today. But recently more and more arguments have begun to appear stating that ["smart machines will be less biased than humans"](https://www.ge.com/news/reports/will-smart-machines-be-less-biased-than-humans) and ["machines are less biased than people"](https://www.verdict.co.uk/ai-and-bias/). The underlying assumption seems to be that even though machine learning algorithms are presented with the same information that humans are presented with, something about the scale and the quality of the algorithms will allow them to overcome data quality issues.

As a data professional it is important to be aware of some fundamental issues that can be present in data and _cannot_ be overcome simply by using big data and advanced algorithms.

## Issues with Data Quality

One of the hardest parts about identifying cognitive bias in datasets is that they may not be so apparent. For example, a healthcare dataset may initially appear to be free of bias, as all data is assumed to be ground truth, but even a raw dataset of patient vital signs may be biased for various reasons. 

Let’s take into account how medical sensor devices like blood oximeters, which measure the amount of oxygen in the blood, work poorly for patients with dark skin tones. If we use the Fitzpatrick skin type we’d be talking about patients in groups V and VI. 

In addition, human biases always come into play when data is collected about people. In medicine, studies have shown that systemically, medical providers don’t take the complaints about pain as seriously from Black patients as they do those of other races. One reason is that mostly non-black medical communities have long held beliefs about Black patients feeling less pain than others. While this belief his held amongst many in the medical community, it impacts the data collected about patients. Medical providers often ask patients to rate their pain and the actual data patients give may not match what is recorded by biased medical professionals.
 
When considering datasets to be good sources of information to build models, we must consider that no matter how much data we have, we rarely have multiple data points for the same person, in other words, we never get the full picture. We have just a snapshot of information about a single patient, without the context of if their vitals were in or outside of normal for that patient. We make major assumptions that the dataset as a whole represents the norm for each patient.

We must also consider the details around how this data is collected. Were patients notified (or did they have the chance to opt out) of having their data collected by their medical institution? Which institution collected the data and how? Was it the hospital system itself, a university- affiliated hospital, or another group. Is the group private or public? These factors are rarely considered by the time data gets to Data Analysts and Scientists, but may cause systemic inequalities downstream. 

## The Problems with Sampling

One of the main issues slowing the development of ML with responsible practices is how difficult it is to change biased data. Often, data is collected in ways that leave out various groups intentionally or unintentionally. 

Sampling bias is one of the most common factors that leads to algorithmic bias. As supervised learning relies heavily on “ground truth” training data, models are skewed when the data they’re trained on are insufficiently representative. 

Representative datasets are those that represent each group well, it goes beyond balance to ensure each group receives somewhat fair predictions by being large enough in sample size. This can mean across gender, race, age, and other sensitive features. Technically, we can use strategies like stratified random sampling to build in goals for representation. For example, if a housing dataset only includes 2% of home buyers who are Native American, whatever model built using that data will predict poorly for Native American subjects. Representation has a major impact on how well a model performs, but is often overlooked by Data Science teams due to the work necessary to make datasets more representative.

Increasing representation can leverage many tactics. The first is additional data collection. Sometimes teams rely on existing data to create ML models rather than curating datasets meant for training ML. This can lead to hesitancy around additional data collection which includes updating and notifying users about privacy/data collection changes, restricting who to collect additional data from, and ______. 

When faced with an imbalanced dataset, practitioners can also resample existing data with fairness constraints. Going back to our housing example, we can set parameters 

Data can be manipulated to prioritize fairness at all steps of the development process. Preprocessing methods often look like data transformations, relabeling, reweighting and resampling. In-process methods require teams to constrain ML models during the training process, developing regularizers to remove prejudice, and using adversarial learning to identify biases. Lastly, post-processing methods include making black-box model predictions fair after a model has been built. 

The difficult part about making data representative is knowing how representative they should be. Some say racial breakdowns should follow the demographics of a country where a specific tool will be used. Other advocate underrepresenting dominant groups to make better predictions for marginalized groups. There is no cookie cutter ratio for deciding how much data is representative, but best practices include consulting social and behavioral scientists to aid in this decision making process. 

## When Some Models Can't Be Fair

While linear models are popular for various reasons, they also have limitations. Often, we build models as little experiments to study how our predictions work on a new dataset. There are various models that work poorly on certain types of distributions in datasets. For example, a linear model would be a poor choice if a dataset had had two naturally occurring, diverging groups.  

ML models exhibit unfairness quite frequently, but identifying this unfairness isn’t always straightforward. Let's take a dataset with naturally high imbalance like arrest records in policing In various parts of the world some racial groups are survilled and policed more often than others creating datasets that broadly look like some racial groups have higher criminal activity. On the basis of various kinds of crimes, studies have shown that the propensity for crime is pretty even across races, even though that’s not what it looks like in a dataset. 

With a dataset heavily influenced by the patterns repeated by law enforcement, it’s hard to create a fair model as the underlying data itself is not fair. On the surface we should strongly consider if we can make a fair classifier using policing data, 

Many would consider using policing data for machine learning applications is unethical from the start considering how skewed the underlying data is, and the concept misalignment which assumes that simple arrest records mean someone has commit a crime, and the overall use case that crime is predictable when in fact, predictive policing tools predict policing. If an ML system is tasked with predicting who the police should arrest next, it would suggest they arrest similar people to ones they’ve arrested in the past, regardless of participation in a crime. 

When we think about the goals of machine learning, at the core we want to discriminate or classify groups based on previously identified patterns. This intentional grouping or separating of groups can be enacted in ways that 

## The Problems with Supervised Learning

Supervised learning has risen to popularity due to its seemingly high accuracy when compared to expert or rules-based systems. However, supervised learning has had shortcomings when exposed to data that isn’t very similar to its training data. 

Supervised learning makes more assumptions than many data science teams don’t initially realize. For example, since at its core supervised learning is pattern matching, we must recognize the hard truths behind supervised learning. 

It’s relatively easy to create a supervised learning model that works well, when our definition of working well involves accurate predictions. However, as an industry, it’s easier to over-index on accuracy asw a success metric. 

Instead, we shouldn’t just aim to create accurate models, models should be developed for 
Accomplishing specific tasks and maximizing both fairness and accuracy. This is a very active research space as the tradeoff between accuracy and model fairness is highly contested. One of the reasons this is a problem in data science is the data models are trained on. 

For example, if there’s a dataset of employees and a company wants to find similar employees, we make various assumptions about what a model trained on this data can do. 

First, we assume that the employees hired are successful in their roles. While one can assume that since they have not been terminated from their positions they must be successful. This may be true, but inevitably doesn’t capture employees that are not meeting expectations. One work-around to this assumption that some orgs do is have managers highlight star employees and those meeting expectations. However, this again skews training data towards those well liked by their managers. This process doesn’t scale and is likely to encode subjective human biases, as managers alone can determine who is a good employee.

Second, we assume we want to hire employees who are similar to our newly categorized 
Successful employees. While looking for characteristics such as honesty or adaptability may not encapsulate human bias, seeking to find candidates with similar aptitude scores, personality types, or interests is likely to pose ethical issues. For a company to meet its goals, it's unclear that hiring candidates similar to their existing employees will achieve or advantage their effort. .

Many scenarios in which engineering teams use supervised learning are not appropriate for

## External Reading

* [An Introduction to Sampling Bias](https://www.scribbr.com/methodology/sampling-bias/)
* [The Impact of Data Preparation on the Fairness of Software Systems](https://arxiv.org/pdf/1910.02321.pdf)
* [Datasets Have Worldviews](https://pair.withgoogle.com/explorables/dataset-worldviews/)
* [The Social Cost of Strategic Classification](https://arxiv.org/abs/1808.08460)
* [The Quest for Ethical Artificial Intelligence](https://www.youtube.com/watch?v=b_--xrN3eso)
