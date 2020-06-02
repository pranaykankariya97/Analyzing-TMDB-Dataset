
# Analyze TMDB Dataset

## Project Overview
The motion picture industry is raking in more revenue than ever with its expansive growth the world over.
Can we build models to accurately predict movie revenue? Could the results from these models be used to further increase revenue?
In this project we will use Exploratory Data Analysis (EDA) and Feature Engineering to analyze the dataset,find correlation between
different fetures and use Seaborn and Plotly to generate interactive graphs.

## Project Objectives
Below are the certain tasks that are to be accomplished in this project
* Visualizing the Target Distribution
* Comparing Film Revenue to Budget
* Do Official Homepages Impact Revenue?
* Distribution of Languages across Films
* Common Words in Film Titles and Descriptions
* How do Film Descriptions Impact Revenue?
* Analyze Movie Release Dates
* Create Features Based on Release Date
* Visualize the Number of Films Per Year
* Number of Films and Revenue Per Year
* Do Release Days Impact Revenue?
* Relationship between Runtime and Revenue

## Results
* Revenue is log-transformed so that distribution in normal.
* Budget is log-transformed and plotted against revenue.We find that they are correleated.As budget increases revenue increases.
* Movie which have a homepage generates more revenue than one which doesnt has.
* Inter-quartile range of revenue is highest for Chinese films  whereas extreme points like maximum and minimum is highest for English films.
* Frequqent words in movie titles are Man,last and Love.
* Frequent words in movie overviews are find,life and one.
* Words like bombing and complications in movie overviews postively affect revenue whereas words like politicians,violence and 18     negatively affect revenue.
* Number of films has been increaing every year.Maximum was 141 in the year 2013 (training set).
* Total revenue also has bee increasing every year and is proportional to the no of film made in a year.Maximum was 13.29Billion for the year 2015.
* Mean revenue is also somewhat proportional to the no of films made in a year.But there is a peak for the year 1975 where mean revenue was 90.48 million where no of films was only 8.This shows that the movies in this year generated a lot of revenue.
* Release day and revenue are not much related to each other.
* Average leangth of films is 2hrs.Also the the revenue is maximum for films with a runtime of 2hrs and decreases with increase or decrease in runtime.

## Software And Libraries
This project uses the following software and libraries:
* [python 3.8.0](https://www.python.org/downloads/release/python-380/)
* [Jupyter Notebook](https://jupyter.org/)
* [pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [SciPy](https://www.scipy.org/)
* [seaborn](https://seaborn.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [plotly](https://plotly.com/)
* [ELI5](https://eli5.readthedocs.io/en/latest/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [wordcloud](https://pypi.org/project/wordcloud/)
* [Natural Language Toolkit](https://www.nltk.org/)

## Contact
Email: pranaykankariya97@gmail.com \
Project Link: [https://github.com/pranaykankariya97/Analyzing-TMDB-Dataset](https://github.com/pranaykankariya97/Analyzing-TMDB-Dataset)

