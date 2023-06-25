#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import cross_val_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn import under_sampling, over_sampling
import scikitplot as skplt

#---function---
def load_data():
    df = pd.read_csv('./transaction_dataset.csv')
    # Fix column names that still have spaces at the beginning of the name of the each columms and change with the correct writing
    df.columns = [i.strip() for i in list(df.columns)]
    df.rename(columns={'total transactions (including tnx to create contract': 'total transactions (including tnx to create contract)'}, inplace=True)
    df.drop(columns=['ERC20 most sent token type', 'ERC20_most_rec_token_type', 'Unnamed: 0', 'Index', 'Address'], inplace=True)
    # Drop features
    df.drop(columns=['ERC20 avg time between sent tnx', 'ERC20 avg time between rec tnx',
                 'ERC20 avg time between rec 2 tnx', 'ERC20 avg time between contract tnx',
                 'ERC20 min val sent contract','ERC20 max val sent contract', 'ERC20 avg val sent contract'], inplace=True)
    # Replace missings of numerical variables with median
    df.fillna(df.median(), inplace=True)
    df = df.drop_duplicates()
    return df

# ==== Main Processs ======
st.set_option('deprecation.showPyplotGlobalUse', False)

def bar_stack(df,col_x,col_y='FLAG'):

    # create crosstab and normalize the values
    ct = pd.crosstab(df[col_x], df[col_y])
    ct_pct = ct.apply(lambda x: x / x.sum(), axis=1)

    # plot stacked horizontal bar chart
    ax = ct_pct.plot(kind='barh', stacked=True, color=['#1f77b4', 'red'])

    # add annotations
    for i in ax.containers:
        # get the sum of values in each container
        total = sum(j.get_width() for j in i)
        
        for j in i:
            # get the width and height of the bar
            width = j.get_width()
            height = j.get_height()
            
            # calculate the position of the text
            x = j.get_x() + width / 2
            y = j.get_y() + height / 2
            
            # format the text as percentage
            text = f'{width:.0%}'
            
            # set the position and format of the annotation
            ax.annotate(text, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center', color='white', fontsize=12,
                        fontweight='bold')


df = load_data()

#List Choice
list_choice = ['Background', 'Exploratory Data Analysis', 'Machine Learning Experiment']

#Side bar
sidebar = st.sidebar.selectbox(
    "Menu", list_choice
)

#Background
if sidebar == 'Background':
    st.title('Prologue 📖')
    st.markdown('---')
    st.subheader(':green[Ethereum] and :green[ERC20]')
    st.write(':green[Ethereum] is a decentralized blockchain platform that enables the creation of smart contracts and decentralized applications (DApps). It has its own native cryptocurrency called Ether (ETH) that serves as the primary means of payment and settlement on the Ethereum network.')
    st.write(':green[ERC20] is a technical standard for creating and issuing tokens on the Ethereum blockchain. ERC20 tokens are fungible, meaning that they are interchangeable and have the same value as one another. Many initial coin offerings (ICOs) and token sales have been launched using the ERC20 standard.')
    st.subheader('Business :green[Problems]')
    st.write('The business problem faced is customer fraud found in the digital financial industry. It can affect the company\'s profits and brand reputation, so companies could take action to retain existing customers.')
    st.write('Both :green[Ethereum] and :green[ERC20] tokens can be involved in various types of fraud, including:')
    st.markdown('''
    - Fake ICOs: Scammers may create fraudulent ICOs, offering investors the opportunity to purchase new ERC20 tokens at a discounted price or in exchange for Ether, but then take the investors' money and disappear.
    - Pump and dump schemes: Fraudsters may artificially inflate the price of an ERC20 token by promoting it heavily on social media or other channels, then sell their holdings at a profit before the price collapses.
    - Token theft: Hackers may steal ERC20 tokens from users' wallets or from exchanges that hold the tokens.
    - Smart contract vulnerabilities: Smart contracts on the Ethereum network can be vulnerable to bugs or flaws that allow hackers to exploit them and steal Ether or ERC20 tokens.
    ''')
    st.subheader('Objectives')
    st.write(':green[This study aims to] perform Ether (ETH) fraud detection and analyze transactional data. By examining the patterns and characteristics of the data, we aim to determine whether or not fraudulent activity has occurred. Through this analysis, we also hope to gain insights into the nature of the transactions and identify any trends or anomalies that may be present. :green[By combining statistical and machine learning techniques,] we aim to develop an effective and accurate method for detecting fraudulent transactions and improving overall security in the financial system.')

#About Dataset
elif sidebar == 'Exploratory Data Analysis':

    #List pilihan explore
    list_explore = ['About The Data', 'Summary Statistics',
                    'KDE Plot', 'Outlier', 'Correlation', 'Target Analysis', 'Feature Analysis']
    sidebar_explore = st.sidebar.selectbox('Select to Explore', list_explore)

    if sidebar_explore == 'About The Data':
        st.title('Know :green[The Data] 📚')
        st.markdown('---')
        st.subheader(':green[Data] Frame')
        st.dataframe(df)

        st.subheader(':green[Target] and :green[Features Information]')
        st.markdown('''
1. `FLAG`: Whether the transaction is fraud or not fraud           
2. `Avg min between sent tnx`: Minimum average time in minutes between transactions sent by the same wallet address
3. `Avg min between received tnx`: Maximum average time in minutes between transactions sent by the same wallet address
4. `Time Diff between first and last (Mins)`: The time difference in minutes between the first and last transaction made by a wallet address
5. `Sent tnx`: The total number of normal transactions sent by a wallet address
6. `Received Tnx`: The total number of normal transactions received by a wallet address
7. `Number of Created Contracts`: The total number of contract creation transactions created by a wallet address
8. `Unique Received From Addresses`: The total number of different wallet addresses that have made transactions and sent Ether to a wallet address
9. `Unique Sent To Addresses`: The total number of different wallet addresses that have received Ether from a particular wallet address
10. `min value received`: Minimum value of Ether ever received by a particular wallet address
11. `max value received`: Maximum value of Ether ever received by a particular wallet address
12. `avg val received`: The average value of Ether ever received by a particular wallet address
13. `min val sent`: Minimum value of Ether ever sent by a particular wallet address
14. `max val sent`: Maximum value of Ether ever sent by a particular wallet address
15. `avg val sent`: The average value of Ether ever sent by a particular wallet address
16. `min value sent to contract`: The minimum value of Ether ever sent to a contract in Ethereum by a particular wallet address
17. `max val sent to contract`: The maximum value of Ether ever sent to a contract in Ethereum by a particular wallet address
18. `avg value sent to contract`: The average value of Ether ever sent to a contract in Ethereum by a particular wallet address
19. `total transactions (including tnx to create contract)`: The total number of transactions that occurred, including transactions to create smart contracts
20. `total Ether sent`: The total amount of Ether that has been sent from an address or account on the Ethereum network
21. `total ether received`: The total amount of Ether that has been received from an address or account on the Ethereum network
22. `total ether sent contracts`: The total amount of Ether that has been sent to smart contract addresses on the Ethereum network
23. `total ether balance`: The total amount of Ether remaining in an address or account after a transaction has been made on the Ethereum network
24. `Total ERC20 tnxs`: The total number of transactions made using ERC20 tokens
25. `ERC20 total Ether received`: The total number of ERC20 token received transactions made using Ether as transaction fee payment
26. `ERC20 total ether sent`: The total number of ERC20 token sent transactions (ERC20 token sent) made using Ether as transaction fee payment
27. `ERC20 total Ether sent contract`: The total amount of Ether sent in an ERC20 token transfer transaction to another smart contract on the Ethereum network
28. `ERC20 uniq sent addr`: Number of transactions for sending unique ERC20 tokens (unique ERC20 token transactions) to unique account addresses
29. `ERC20 uniq rec addr`: Number of transactions receiving unique ERC20 tokens (unique ERC20 token transactions) from unique account addresses
30. `ERC20 uniq sent addr.1`: Number of transactions for sending unique ERC20 tokens (unique ERC20 token transactions) to unique account addresses
31. `ERC20 uniq rec contract addr`: Number of unique ERC20 token transactions received from unique smart contract addresses
32. `ERC20 min val rec`: The minimum value in Ether received from an ERC20 token transaction for an account or address
33. `ERC20 max val rec`: The maximum value in Ether received from an ERC20 token transaction for an account or address
34. `ERC20 avg val rec`: The average value in Ether received from an ERC20 token transaction for an account or address
35. `ERC20 min val sent`: The minimum value in Ether sent via an ERC20 token transaction for an account or address
36. `ERC20 max val sent`: The maximum value in Ether sent via an ERC20 token transaction for an account or address
37. `ERC20 avg val sent`: The average value in Ether sent via an ERC20 token transaction for an account or address
38. `ERC20 uniq sent token name`: The number of unique ERC20 tokens sent in a transaction to an address or account on the Ethereum network
39. `ERC20 uniq rec token name`: The number of unique ERC20 tokens received in a transaction to an address or account on the Ethereum network
       ''')

        st.subheader('First 5 Rows :green[Data Display]')
        st.dataframe(df.head())

        st.subheader(':green[Last 5 Rows] Data Display')
        st.dataframe(df.tail())

        st.subheader('Data :green[Preporcessing]')
        st.markdown('''
        * :green[Redundant Features:] The columns are unique, and the values of these variables are all 0s, as zero variance indicates constant or near-constant behavior in the variables.
        * :green[Duplicates:] The same 553 rows exist after deleting unique columns. The rows were dropped.
        * :green[Missing Values:] At 8.42% the empty rows were medianly imputed. Most of the columns with missing values are right-skew distributions.
        * :green[Multicollinearity:] Only nine features have correlated with the threshold value of 0.8. Dropping them is a good way except for one feature that has the most highly correlated with the target.
        ''')
        st.write('The classification dataset has a target with two values (:green[fraud and not fraud]) and 50 features with a total of 9,841 rows. After the process, only :green[9,288 rows] and :green[38 features] will be utilized for :green[analysis] and :green[modeling.] This data preprocessing makes features more informative and useful to improve model performance.')


    elif sidebar_explore == 'Summary Statistics':
        st.title(':green[Summary] Statistics 🔬')
        st.markdown('---')
        st.dataframe(df.describe().style.background_gradient(cmap='Greens'))
        st.write(':green[The larger] the value, the more intense the :green[green color] displayed in the cell.')
        st.markdown('''
        The provided statistical summary describes the distribution of values for multiple variables. Here is an overview of the information:
        * `count`: The number of observations for each variable is 9288.
        * `mean`: The mean (average) value of each variable is provided.
        * `std`: The standard deviation represents the dispersion or spread of the values around the mean.
        * `min`: The minimum value observed for each variable.
        * `25%`: The 25th percentile, also known as the first quartile, indicates the value below which 25% of the data falls.
        * `50%`: The 50th percentile, also known as the median, represents the value below which 50% of the data falls.
        * `75%`: The 75th percentile, also known as the third quartile, indicates the value below which 75% of the data falls.
        * `max`: The maximum value observed for each variable.
  
        The summary provides insights into the central tendency, spread, and range of the variables in the dataset.
        ''')

    elif sidebar_explore == 'KDE Plot':
        st.title('Distribution :green[Visualization] 📈')
        st.markdown('---')
        st.subheader('1. Avg min between sent tnx')
        fig = plt.figure()
        sns.kdeplot(data=df['Avg min between sent tnx'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('2. Avg min between received tnx')
        fig = plt.figure()
        sns.kdeplot(data=df['Avg min between received tnx'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)
        
        st.subheader('3. Time Diff between first and last (Mins)')
        fig = plt.figure()
        sns.kdeplot(data=df['Time Diff between first and last (Mins)'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('4. Sent tnx')
        fig = plt.figure()
        sns.kdeplot(data=df['Sent tnx'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('5. Received Tnx')
        fig = plt.figure()
        sns.kdeplot(data=df['Received Tnx'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('6. Number of Created Contracts')
        fig = plt.figure()
        sns.kdeplot(data=df['Number of Created Contracts'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('7. Unique Received From Addresses')
        fig = plt.figure()
        sns.kdeplot(data=df['Unique Received From Addresses'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('8. Unique Sent To Addresses')
        fig = plt.figure()
        sns.kdeplot(data=df['Unique Sent To Addresses'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('9. min value received')
        fig = plt.figure()
        sns.kdeplot(data=df['min value received'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('10. max value received')
        fig = plt.figure()
        sns.kdeplot(data=df['max value received'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('11. avg val received')
        fig = plt.figure()
        sns.kdeplot(data=df['avg val received'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('12. min val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['min val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('13. max val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['max val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('14. avg val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['avg val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('15. min value sent to contract')
        fig = plt.figure()
        sns.kdeplot(data=df['min value sent to contract'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('16. max val sent to contract')
        fig = plt.figure()
        sns.kdeplot(data=df['max val sent to contract'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('17. avg value sent to contract')
        fig = plt.figure()
        sns.kdeplot(data=df['avg value sent to contract'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('18. total transactions (including tnx to create contract)')
        fig = plt.figure()
        sns.kdeplot(data=df['total transactions (including tnx to create contract)'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('19. total Ether sent')
        fig = plt.figure()
        sns.kdeplot(data=df['total Ether sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('20. total ether received')
        fig = plt.figure()
        sns.kdeplot(data=df['total ether received'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('21. total ether sent contracts')
        fig = plt.figure()
        sns.kdeplot(data=df['total ether sent contracts'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('22. total ether balance')
        fig = plt.figure()
        sns.kdeplot(data=df['total ether balance'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('23. Total ERC20 tnxs')
        fig = plt.figure()
        sns.kdeplot(data=df['Total ERC20 tnxs'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('24. ERC20 total Ether received')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 total Ether received'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('25. ERC20 total ether sent')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 total ether sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('26. ERC20 total Ether sent contract')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 total Ether sent contract'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('27. ERC20 uniq sent addr')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq sent addr'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('28. ERC20 uniq rec addr')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq rec addr'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('29. ERC20 uniq sent addr.1')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq sent addr.1'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('30. ERC20 uniq rec contract addr')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq rec contract addr'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('31. ERC20 min val rec')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 min val rec'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('32. ERC20 max val rec')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 max val rec'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('33. ERC20 avg val rec')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 avg val rec'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('34. ERC20 min val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 min val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('35. ERC20 max val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 max val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('36. ERC20 avg val sent')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 avg val sent'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('37. ERC20 uniq sent token name')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq sent token name'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.subheader('38. ERC20 uniq rec token name')
        fig = plt.figure()
        sns.kdeplot(data=df['ERC20 uniq rec token name'], color='green', alpha=0.7)
        plt.grid(True)
        # Show the plot
        st.pyplot(fig)

        st.write('The data has a skewness that is skewed to the right. It indicates a longer tail of the distribution to the right of the graph. In the context of data analysis, a positive skewness indicates a larger spread of data to the right of the median value of the distribution.')

    elif sidebar_explore == 'Outlier':
        st.header('See :green[The Anomaly] 🔎')
        st.markdown('---')
        st.subheader(':green[Box Plots] for All Variables')
        fig_out = plt.figure(figsize = (30,45))
        ax =sns.boxplot(data = df, orient='h')
        ax.set(xscale='log')
        # Customize the plot
        st.pyplot(fig_out)
        st.write(':green[The features all lie in different ranges.] The outlier values in our dataset are genuinely representative of the data we are analyzing, and removing them would not introduce any biases or distort the nature of the problem we are trying to solve, so there may be :green[no need to remove the outliers.]')

    elif sidebar_explore == 'Correlation':
        corr_matrix = df.corr(method='pearson')
        st.title('Correlation 🔗')
        st.markdown('---')
        # Sets the gradient background style on the correlation matrix
        cor_matrix_style = corr_matrix.style.background_gradient(cmap='Greens')

        # Displaying the correlation matrix with gradient background style using st.dataframe()
        st.subheader('Matrix With :green[Pearson]')
        st.dataframe(cor_matrix_style)

        st.markdown('''
        :green[After observing,] for modelling later, here are the features that correlate with :green[the threshold] value of 0.8.

        1. `avg value sent to contract`
        2. `total transactions (including tnx to create contract)`
        3. `total ether sent contracts`
        4. `ERC20 max val rec`
        5. `ERC20 avg val rec`
        6. `ERC20 min val sent`
        7. `ERC20 max val sent`
        8. `ERC20 avg val sent`
        9. `ERC20 uniq rec token name`

        We need to drop these columns, except `total transactions (including tnx to create contract)` because has the most highly correlated with `FLAG`.
        ''')

        st.subheader('Most Important Features or :green[Highly Correlated Features] with Target in Absolute Value')
        correlation_matrix = df.corr()
        fig_corr = plt.figure(figsize=(10, 25))
        ax = correlation_matrix['FLAG'][1:].abs().sort_values().plot(kind='barh', color='olivedrab')
        # Displaying correlation images using st.pyplot()
        st.pyplot(fig_corr)

        # Show the high value correlated with FLAG in table
        st.subheader('The Value :green[Correlated Features] with Target')
        c = df.corr()['FLAG'][1:].sort_values()
        st.table(c)

        st.subheader(':green[Negative Correlations] with Our Target')
        col1, col2 = st.columns(2)
        colors = ['#008000', '#ff9124']
        with col1:
            st.write('Time Difference (Mins)')
            fig1, ax1 = plt.subplots(figsize=(7,7))
            sns.boxplot(x='FLAG', y='Time Diff between first and last (Mins)', data=df, palette=colors)
            st.pyplot(fig1)

            st.write('Avg Min Between Received Transaction')
            fig2, ax2 = plt.subplots(figsize=(7,7))
            sns.boxplot(x='FLAG', y='Avg min between received tnx', data=df, palette=colors)
            st.pyplot(fig2)
        with col2:
            st.write('Total Transactions')
            fig3, ax3 = plt.subplots(figsize=(7,7.35))
            sns.boxplot(x='FLAG', y='total transactions (including tnx to create contract)', data=df, palette=colors)
            st.pyplot(fig3)

            st.write('Sent Transaction')
            fig4, ax4 = plt.subplots(figsize=(7,7))
            sns.boxplot(x='FLAG', y='Sent tnx', data=df, palette=colors)
            st.pyplot(fig4)
        st.write('The lower our feature value the more likely it will be a fraud transaction')

        st.subheader(':green[Positive Correlations] with Our Target')
        col1, col2 = st.columns(2)
        colors = ['#008000', '#ff9124']
        with col1:
            st.write('Minimal Sent')
            fig1, ax1 = plt.subplots(figsize=(7,7))
            sns.boxplot(x='FLAG', y='min val sent', data=df, palette=colors)
            st.pyplot(fig1)

            st.write('ERC20 Total Ether Sent Contract')
            fig2, ax2 = plt.subplots(figsize=(7,7))
            sns.boxplot(x='FLAG', y='ERC20 total Ether sent contract', data=df, palette=colors)
            st.pyplot(fig2)
        with col2:
            st.write('ERC20 Minimal Sent')
            fig3, ax3 = plt.subplots(figsize=(7,6.5))
            sns.boxplot(x='FLAG', y='ERC20 min val sent', data=df, palette=colors)
            st.pyplot(fig3)

            st.write('ERC20 Total Ether Sent')
            fig4, ax4 = plt.subplots(figsize=(7,6.5))
            sns.boxplot(x='FLAG', y='ERC20 total ether sent', data=df, palette=colors)
            st.pyplot(fig4)

    elif sidebar_explore == 'Target Analysis':
        st.title('Target 🎯')
        st.markdown('---')
        st.subheader('Percentage of Fraud >< No Fraud Transaction')
        labels = 'No Fraud', 'Fraud'
        target = df['FLAG'].value_counts()
        color = ['olivedrab', 'darkgrey']

        fig1, ax1 = plt.subplots(facecolor='none', nrows=1, ncols=1, figsize=(15, 7.5))
        wedges, texts, autotexts = ax1.pie(target, labels=labels, colors=color,
                                           autopct='%1.2f%%', pctdistance=0.75,
                                           explode=(0.1, 0), startangle=90,
                                           shadow=True, counterclock=False,
                                           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})

        plt.setp(autotexts, size=14, weight='bold')

        # Show the amount of data FLAG
        total_data = target.sum()
        for i, v in enumerate(target):
            percentage = '{:.1%}'.format(v / total_data)
            texts[i].set_text(f'{labels[i]}\n({v:,})')
            texts[i].set_color(color[i])
            texts[i].set_fontsize(14)
        plt.show()
        st.pyplot(fig1)

        st.write('The targets in this dataset are very unbalanced, so it is necessary to handle imbalanced data for :green[machine learning] modeling.')

    elif sidebar_explore == 'Feature Analysis':
        df_EDA = copy.deepcopy(df)
        list_eda = ['Table', 'Bar Chart', 'Scatterplot']
        sidebar_eda = st.sidebar.selectbox('Select Visualization', list_eda) 

        if sidebar_eda == 'Table':
            st.title('Table 🗃')
            st.markdown('---')
            st.subheader('What is :green[the comparison] between the average value of Ether sent by an address and the average value of Ether received :green[by the particular address in a specific period of time?]')
            def time_diff(row):
                if row['Time Diff between first and last (Mins)'] == 0:
                  return 0 #0 min
                elif row['Time Diff between first and last (Mins)'] > 0 and row['Time Diff between first and last (Mins)'] <= 259200:
                  return 1 #Within 6 months
                elif row['Time Diff between first and last (Mins)'] > 259200 and row['Time Diff between first and last (Mins)'] <= 525600:
                  return 2 #Within 1 year
                else:
                  return 3 #More than a year
            df_EDA['Time Diff between first and last (Mins)_func'] = df_EDA.apply(time_diff, axis=1)
            comparison_avg_transaction = (df_EDA
                        .groupby(['Time Diff between first and last (Mins)_func', 'FLAG']) 
                        .agg(avg_eth_sent=('avg val sent','sum'),
                             avg_eth_received=('avg val received','sum'))
                        .reset_index()
                        .sort_values('FLAG',ascending=False)
                        )
            st.table(comparison_avg_transaction)
            st.markdown('''
            Refers to the difference in time (in minutes) between the first and last transactions in a given dataset. We category it as below:
            * [0] 0 minutes
            * [1] Within 6 months
            * [2] Within a year
            * [3] More than a year
            ''')

            st.subheader('What is :green[the difference] between the wallet addresses that have made ETH transactions sent and received on total transactions based on :green[fraud classification?]')
            def total_transaction(row):
                if row['total transactions (including tnx to create contract)'] == 0:
                  return 0 #No transaction
                elif row['total transactions (including tnx to create contract)'] > 0 and row['total transactions (including tnx to create contract)'] <= 10:
                  return 1 #Small amount
                elif row['total transactions (including tnx to create contract)'] > 10 and row['total transactions (including tnx to create contract)'] <= 63:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['total transactions_func'] = df_EDA.apply(total_transaction, axis=1)
            total_transaction = (df_EDA
                        .groupby(['total transactions_func', 'FLAG']) 
                        .agg(uniq_rec_add=('Unique Received From Addresses','sum'),
                             uniq_sent_add=('Unique Sent To Addresses','sum'))
                        .reset_index()
                        .sort_values('FLAG',ascending=False)
                        )
            st.table(total_transaction)
            st.markdown('''
            Total number of transactions. We category it as below:
            * [0] No transaction
            * [1] Small amount ETH transactions
            * [2] Medium amount ETH transactions
            * [3] Large amount ETH transactions
            ''')

            st.subheader('What is the average median :green[(minimum, maximum, average)] value sent based on the min value sent to contract?')
            top_min_contract_value_sent = (df_EDA
                        .groupby('min value sent to contract') 
                        .agg(med_MinValSent=('min val sent','median'),
                             med_MaxValSent=('max val sent','median'),
                             med_ValSents=('avg val sent','median'))
                        .reset_index()
                        .sort_values('med_ValSents',ascending=False)
                        )
            st.table(top_min_contract_value_sent)
            st.markdown('''
            Minimum value of Ether sent to a contract. This is a feature in the dataset that indicates the minimum amount of Ether that has been sent to a contract in a particular transaction. In Ethereum, contracts can receive and manage funds, and this feature measures the smallest amount of funds that have been sent to a contract in a transaction. We category it as below:
            * [0] No minimal value sent
            * [0.01] Small amount minimal value sent
            * [0.02] Medium amount minimal value sent
            
            It seems that the minimum value of Ether sent to a contract in the dataset is only either 0, 0.01, or 0.02. It could be that transactions that involve sending Ether to a contract with a value less than 0.01 are not recorded in the dataset, or that such transactions are simply not common in the Ethereum network.
            ''')

            st.subheader('What is the average median :green[(minimum, maximum, average)] value sent based on :green[each target?]')
            top_flag_value_sent = (df_EDA
                        .groupby('FLAG') 
                        .agg(med_MinValSent=('min val sent','median'),
                             med_MaxValSent=('max val sent','median'),
                             med_ValSents=('avg val sent','median'))
                        .reset_index()
                        .sort_values('med_ValSents',ascending=False)
                        )
            st.table(top_flag_value_sent)


        elif sidebar_eda == 'Bar Chart':
            df_EDA = copy.deepcopy(df)

            st.title(':green[Bar] Chart 📊')
            st.markdown('---')
            
            st.subheader('Comparison of :green[Target] based on Avg min between sent tnx')
            def avg_time_sent(row):
                if row['Avg min between sent tnx'] == 0:
                  return 0 #0 mins
                else:
                  if row['Avg min between sent tnx'] > 0 and row['Avg min between sent tnx'] <= 720:
                    return 1 #Within 12 hours
                  else:
                    if row['Avg min between sent tnx'] > 720 and row['Avg min between sent tnx'] <= 1440:
                      return 2 #Within a day
                    else:
                      return 3 #More than a day
            df_EDA['Avg min between sent tnx_func'] = df_EDA.apply(lambda row: avg_time_sent(row), axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Avg min between sent tnx_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            The minimum average time between sent transactions for an account in minutes is a measure in the blockchain that calculates the average minimum time between two transactions sent by a particular sending address. The smaller the Avg min between sent tnx value, the more often the user commits transactions and the quicker the user makes a new transaction after the previous one. We category it as below:
            * [0] 0 minutes
            * [1] Within 12 hours
            * [2] Within a day
            * [3] More than a day
            ''')

            st.subheader('Comparison of :green[Target] based on Avg min between received tnx')
            def avg_time_received(row):
                if row['Avg min between received tnx'] == 0:
                  return 0 #0 mins
                elif row['Avg min between received tnx'] > 0 and row['Avg min between received tnx'] <= 720:
                  return 1 #Within 12 hours
                elif row['Avg min between received tnx'] > 720 and row['Avg min between received tnx'] <= 1440:
                    return 2 #Within a day
                else:
                  return 3 #More than a day
            df_EDA['Avg min between received tnx_func'] = df_EDA.apply(avg_time_received, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Avg min between received tnx_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Refers to the average time (in minutes) between each received transaction in a given dataset. We category it as below:
            * [0] 0 minutes
            * [1] Within 12 hours
            * [2] Within a day
            * [3] More than a day
            ''')

            st.subheader('Comparison of :green[Target] based on Time Diff between first and last (Mins)')
            def time_diff(row):
                if row['Time Diff between first and last (Mins)'] == 0:
                  return 0 #0 min
                elif row['Time Diff between first and last (Mins)'] > 0 and row['Time Diff between first and last (Mins)'] <= 259200:
                  return 1 #Within 6 months
                elif row['Time Diff between first and last (Mins)'] > 259200 and row['Time Diff between first and last (Mins)'] <= 525600:
                  return 2 #Within 1 year
                else:
                  return 3 #More than a year
            df_EDA['Time Diff between first and last (Mins)_func'] = df_EDA.apply(time_diff, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Time Diff between first and last (Mins)_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Refers to the difference in time (in minutes) between the first and last transactions in a given dataset. We category it as below:
            * [0] 0 minutes
            * [1] Within 6 months
            * [2] Within a year
            * [3] More than a year
            ''')

            st.subheader('Comparison of :green[Target] based on Sent tnx')
            def sent_transaction(row):
                if row['Sent tnx'] == 0:
                  return 0 #No ETH transaction
                elif row['Sent tnx'] > 0 and row['Sent tnx'] <= 3:
                  return 1 #Small amount
                elif row['Sent tnx'] > 3 and row['Sent tnx'] <= 12:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['Sent tnx_func'] = df_EDA.apply(sent_transaction, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Sent tnx_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total number of sent normal transactions. We category it as below:
            * [0] No transaction
            * [1] Small amount ETH transaction
            * [2] Medium amount ETH transaction
            * [3] Large amount ETH transaction
            ''')

            st.subheader('Comparison of :green[Target] based on Received tnx')
            def received_transaction(row):
                if row['Received Tnx'] == 0:
                  return 0 #No ETH transaction
                elif row['Received Tnx'] > 0 and row['Received Tnx'] <= 5:
                  return 1 #Small amount
                elif row['Received Tnx'] > 5 and row['Received Tnx'] <= 29:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['Received Tnx_func'] = df_EDA.apply(received_transaction, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Received Tnx_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total number of received normal transactions. We category it as below:
            * [0] No transaction
            * [1] Small amount ETH transaction
            * [2] Medium amount ETH transaction
            * [3] Large amount ETH transaction
            ''')

            st.subheader('Comparison of :green[Target] based on Unique Received From Addresses')
            def unique_received(row):
                if row['Unique Received From Addresses'] == 0:
                  return 0 #No unique address
                elif row['Unique Received From Addresses'] > 0 and row['Unique Received From Addresses'] <= 2:
                  return 1 #Few unique addresses
                elif row['Unique Received From Addresses'] > 2 and row['Unique Received From Addresses'] <= 5:
                  return 2 #A considerable number of unique addresses
                else:
                  return 3 #A large number of unique addresses
            df_EDA['Unique Received From Addresses_func'] = df_EDA.apply(unique_received, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Unique Received From Addresses_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Unique addresses from which account received transactions.  We category it as below:
            * [0] No unique address
            * [1] Few unique addresses
            * [2] A considerable number of unique addresses
            * [3] A large number of unique addresses
            ''')

            st.subheader('Comparison of :green[Target] based on Unique Sent To Addresses')
            def unique_sent(row):
                if row['Unique Sent To Addresses'] == 0:
                  return 0 #No unique address
                elif row['Unique Sent To Addresses'] > 0 and row['Unique Sent To Addresses'] <= 2:
                  return 1 #Few unique address
                elif row['Unique Sent To Addresses'] > 2 and row['Unique Sent To Addresses'] <= 3:
                  return 2 #A considerable number of unique address
                else:
                  return 3 #A large number of unique addresses
            df_EDA['Unique Sent To Addresses_func'] = df_EDA.apply(unique_sent, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Unique Sent To Addresses_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Unique addresses from which account sent transactions. We category it as below:
            * [0] No unique address
            * [1] Few unique addresses
            * [2] A considerable number of unique addresses
            * [3] A large number of unique addresses
            ''')

            st.subheader('Comparison of :green[Target] based on min value received')
            def min_val_rec(row):
                if row['min value received'] == 0:
                  return 0 #No ETH received
                elif row['min value received'] > 0 and row['min value received'] <= 2:
                   return 1 #Several amounts ETH received
                else:
                  return 2 #Many amounts ETH received
            df_EDA['min value received_func'] = df_EDA.apply(min_val_rec, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='min value received_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Minimum value in Ether ever received. We category it as below:
            * [0] No ETH received
            * [1] Several amounts ETH received
            * [2] Many amounts ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on max value received')
            def max_val_rec(row):
                if row['max value received'] == 0:
                  return 0 #No ETH received
                elif row['max value received'] > 0 and row['max value received'] <= 7.8283:
                  return 1 #Small amount
                elif row['max value received'] > 7.8283 and row['max value received'] <= 75:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['max value received_func'] = df_EDA.apply(max_val_rec, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='max value received_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Maximum value in Ether ever received. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Medium amount ETH received
            * [3] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on avg val received')
            def avg_val_rec(row):
                if row['avg val received'] == 0:
                  return 0 #No ETH received
                elif row['avg val received'] > 0 and row['avg val received'] <= 2.078:
                  return 1 #Small amount
                elif row['avg val received'] > 2.078 and row['avg val received'] <= 28.7:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['avg val received_func'] = df_EDA.apply(avg_val_rec, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='avg val received_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Average value in Ether ever received. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Medium amount ETH received
            * [3] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on min val sent')
            def min_val_sent(row):
                if row['min val sent'] == 0:
                  return 0 #No ETH sent
                elif row['min val sent'] > 0 and row['min val sent'] <= 1:
                  return 1 #Small amount
                else:
                  return 2 #Large amount
            df_EDA['min val sent_func'] = df_EDA.apply(min_val_sent, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='min val sent_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Minimum value of Ether ever sent. We category it as below:
            * [0] No ETH sent
            * [1] Small amount ETH sent
            * [2] Large amount ETH sent
            ''')

            st.subheader('Comparison of :green[Target] based on max val sent')
            def max_val_sent(row):
                if row['max val sent'] == 0:
                  return 0 #No ETH sent
                elif row['max val sent'] > 0 and row['max val sent'] <= 6.203:
                  return 1 #Small amount
                elif row['max val sent'] > 6.203 and row['max val sent'] <= 68.505:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['max val sent_func'] = df_EDA.apply(max_val_sent, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='max val sent_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Maximum value of Ether ever sent. We category it as below:
            * [0] No ETH sent
            * [1] Small amount ETH sent
            * [2] Medium amount ETH sent
            * [3] Large amount ETH sent
            ''')

            st.subheader('Comparison of :green[Target] based on avg val sent')
            def avg_val_sent(row):
                if row['avg val sent'] == 0:
                  return 0 #No ETH sent
                elif row['avg val sent'] > 0 and row['avg val sent'] <= 2:
                  return 1 #Small amount
                elif row['avg val sent'] > 2 and row['avg val sent'] <= 25.24:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['avg val sent_func'] = df_EDA.apply(avg_val_sent, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='avg val sent_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Average value of Ether ever sent. We category it as below:
            * [0] No ETH sent
            * [1] Small amount ETH sent
            * [2] Medium amount ETH sent
            * [3] Large amount ETH sent
            ''')

            st.subheader('Comparison of :green[Target] based on min value sent to contract')
            plt.subplots(figsize=(14, 7))
            ax = sns.countplot(data=df_EDA, x='min value sent to contract', hue='FLAG', palette='Greens')
            ax.set_yscale('log')
            st.pyplot()
            st.markdown('''
            Minimum value of Ether sent to a contract. This is a feature in the dataset that indicates the minimum amount of Ether that has been sent to a contract in a particular transaction. In Ethereum, contracts can receive and manage funds, and this feature measures the smallest amount of funds that have been sent to a contract in a transaction. We category it as below:
            * [0] No minimal value sent
            * [0.01] Small amount minimal value sent
            * [0.02] Medium amount minimal value sent
            ''')
            st.write('It seems that the minimum value of Ether sent to a contract in the dataset is only either 0, 0.01, or 0.02. It could be that transactions that involve sending Ether to a contract with a value less than 0.01 are not recorded in the dataset, or that such transactions are simply not common in the Ethereum network.')

            st.subheader('Comparison of :green[Target] based on max val sent to contract')
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='max val sent to contract', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Maximum value of Ether sent to a contract. This feature tracks the highest amount of Ether transferred to a particular contract address during the observation period. It can be useful in analyzing the behavior of users who are sending larger amounts of Ether to contracts, which may indicate different types of activities or transactions taking place. We category it as below:
            * [0] No maximum value sent
            * [0.01] Small amount maximum value sent
            * [0.02] Medium amount maximum value sent
            * [0.046029] Large amount maximum value sent

            The maximum value of Ether sent to a contract is either 0, 0.01, 0.02, or 0.046029, and there are no other values in the dataset for this feature. It could be possible that the contracts in the dataset have only received these specific values of Ether, and no other values.
            ''')

            st.subheader('Comparison of :green[Target] based on avg value sent to contract')
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='avg value sent to contract', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            We category it as below:
            * [0] No average value sent
            * [0.01] Small amount average value sent
            * [0.02] Medium amount average value sent
            * [0.023014] Large amount average value sent
            ''')

            st.subheader('Comparison of :green[Target] based on total transactions (including tnx to create contract)')
            def total_transaction(row):
                if row['total transactions (including tnx to create contract)'] == 0:
                  return 0 #No transaction
                elif row['total transactions (including tnx to create contract)'] > 0 and row['total transactions (including tnx to create contract)'] <= 10:
                  return 1 #Small amount
                elif row['total transactions (including tnx to create contract)'] > 10 and row['total transactions (including tnx to create contract)'] <= 63:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['total transactions_func'] = df_EDA.apply(total_transaction, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='total transactions_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total number of transactions. We category it as below:
            * [0] No transaction
            * [1] Small amount ETH transactions
            * [2] Medium amount ETH transactions
            * [3] Large amount ETH transactions
            ''')

            st.subheader('Comparison of :green[Target] based on total Ether sent')
            def ether_sent(row):
                if row['total Ether sent'] == 0:
                  return 0 #No ETH sent
                elif row['total Ether sent'] > 0 and row['total Ether sent'] <= 0.0174:
                    return 1 #Small amount
                elif row['total Ether sent'] > 0.0174 and row['total Ether sent'] <= 100.99:
                    return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['total Ether sent_func'] = df_EDA.apply(ether_sent, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='total Ether sent_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Ether sent for account address. We category it as below:
            * [0] No ETH sent
            * [1] Small amount ETH sent
            * [2] Medium amount ETH sent
            * [3] Large amount ETH sent
            ''')

            st.subheader('Comparison of :green[Target] based on total ether received')
            def ether_received(row):
                if row['total ether received'] == 0:
                  return 0 #No ETH received
                elif row['total ether received'] > 0 and row['total ether received'] <= 0.372:
                    return 1 #Small amount
                elif row['total ether received'] > 0.372 and row['total ether received'] <= 101:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['total ether received_func'] = df_EDA.apply(ether_received, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='total ether received_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Ether received for account address. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Medium amount ETH received
            * [3] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on total ether sent contracts')
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='total ether sent contracts', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Ether sent to Contract addresses. We category it as below:
            * [0] No ETH sent
            * [0.01] Small amount ETH sent
            * [0.2] Medium amount ETH sent
            * [0.046028713] Large amount ETH sent
            ''')

            st.subheader('Comparison of :green[Target] based on total ether balance')
            def ether_balance(row):
                if row['total ether balance'] <= 0:
                  return 0 #Minus or zero balance
                elif row['total ether balance'] > 0 and row['total ether balance'] <= 0.00000201:
                  return 1 #Small amount
                elif row['total ether balance'] > 0.00000201 and row['total ether balance'] <= 0.00063:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['total ether balance_func'] = df_EDA.apply(ether_balance, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='total ether balance_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total Ether Balance following enacted transactions. We category it as below:
            * [0] Minus or no ETH balance
            * [1] Small amount ETH balance
            * [2] Medium amount ETH balance
            * [3] Large amount ETH balance
            ''')

            st.subheader('Comparison of :green[Target] based on Total ERC20 tnxs')
            def total_ERC20(row):
                if row['Total ERC20 tnxs'] == 0:
                  return 0 #No ERC20 token transaction
                elif row['Total ERC20 tnxs'] > 0 and row['Total ERC20 tnxs'] <= 1:
                  return 1 #Small amount
                elif row['Total ERC20 tnxs'] > 1 and row['Total ERC20 tnxs'] <= 3:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['Total ERC20 tnxs_func'] = df_EDA.apply(total_ERC20, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='Total ERC20 tnxs_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total number of ERC20 token transfer transactions. We category it as below:
            * [0] No ERC20 token transaction
            * [1] Small amount ERC20 token transactions
            * [2] Medium amount ERC20 token transactions
            * [3] Large amount ERC20 token transactions
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 total Ether received')
            def ERC20_received(row):
                if row['ERC20 total Ether received'] == 0:
                  return 0 #No ETH received
                elif row['ERC20 total Ether received'] > 0 and row['ERC20 total Ether received'] <= 0.000000000001:
                  return 1 #Small amount
                elif row['ERC20 total Ether received'] > 0.000000000001 and row['ERC20 total Ether received'] <= 0.886:
                  return 2 #Medium amount
                else:
                  return 3 #Large amount
            df_EDA['ERC20 total Ether received_func'] = df_EDA.apply(ERC20_received, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 total Ether received_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Total ERC20 token received transactions in Ether. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Medium amount ETH received
            * [3] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 uniq rec addr')
            def ERC20_rec_uniq(row):
                if row['ERC20 uniq rec addr'] == 0:
                  return 0 #No ERC20 token received
                elif row['ERC20 uniq rec addr'] > 0 and row['ERC20 uniq rec addr'] <= 1:
                  return 1 #Small amount ERC20 token
                elif row['ERC20 uniq rec addr'] > 1 and row['ERC20 uniq rec addr'] <= 2:
                  return 2 #Medium amount ERC20 token
                else:
                  return 3 #Large amount ERC20 token
            df_EDA['ERC20 uniq rec addr_func'] = df_EDA.apply(ERC20_rec_uniq, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 uniq rec addr_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Number of ERC20 token transactions received from Unique addresses. We category it as below:
            * [0] No ERC20 token received
            * [1] Small amount ERC20 token received
            * [2] Medium amount ERC20 token received
            * [3] Large amount ERC20 token received
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 uniq rec contract addr')
            def ERC20_rec_uniq_con(row):
                if row['ERC20 uniq rec contract addr'] == 0:
                  return 0 #No ERC20 token received
                elif row['ERC20 uniq rec contract addr'] > 0 and row['ERC20 uniq rec contract addr'] <= 1:
                  return 1 #Small amount ERC20 token
                elif row['ERC20 uniq rec contract addr'] > 1 and row['ERC20 uniq rec contract addr'] <= 2:
                  return 2 #Medium amount ERC20 token
                else:
                  return 3 #Large amount ERC20 token
            df_EDA['ERC20 uniq rec contract addr_func'] = df_EDA.apply(ERC20_rec_uniq_con, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 uniq rec contract addr_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Number of ERC20token transactions received from Unique contract addresses. We category it as below:
            * [0] No ERC20 token received
            * [1] Small amount ERC20 token received
            * [2] Medium amount ERC20 token received
            * [3] Large amount ERC20 token received
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 max val rec')
            def ERC20_rec_max_val(row):
                if row['ERC20 max val rec'] == 0:
                  return 0 #No ETH received
                elif row['ERC20 max val rec'] > 0 and row['ERC20 max val rec'] <= 0.755:
                  return 1 #Small amount
                else:
                  return 2 #Large amount
            df_EDA['ERC20 max val rec_func'] = df_EDA.apply(ERC20_rec_max_val, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 max val rec_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Maximum value in Ether received from ERC20 token transactions for account. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 avg val rec')
            def ERC20_rec_avg_val(row):
                if row['ERC20 avg val rec'] == 0:
                  return 0 #No ETH received
                elif row['ERC20 avg val rec'] > 0 and row['ERC20 avg val rec'] <= 0.2027:
                  return 1 #Small amount
                else:
                  return 2 #Large amount
            df_EDA['ERC20 avg val rec_func'] = df_EDA.apply(ERC20_rec_avg_val, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 avg val rec_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Average value in Ether received from ERC20 token transactions for account. We category it as below:
            * [0] No ETH received
            * [1] Small amount ETH received
            * [2] Large amount ETH received
            ''')

            st.subheader('Comparison of :green[Target] based on ERC20 uniq rec token name')
            def ERC20_rec_token_name(row):
                if row['ERC20 uniq rec token name'] == 0:
                  return 0 #No ERC20 token received
                elif row['ERC20 uniq rec token name'] > 0 and row['ERC20 uniq rec token name'] <= 1:
                  return 1 #Small amount ERC20 token
                elif row['ERC20 uniq rec token name'] > 1 and row['ERC20 uniq rec token name'] <= 2:
                  return 2 #Medium amount ERC20 token
                else:
                  return 3 #Large amount ERC20 token
            df_EDA['ERC20 uniq rec token name_func'] = df_EDA.apply(ERC20_rec_token_name, axis=1)
            plt.subplots(figsize=(14, 7))
            sns.countplot(data=df_EDA, x='ERC20 uniq rec token name_func', hue='FLAG', palette='Greens')
            st.pyplot()
            st.markdown('''
            Number of Unique ERC20 tokens received. We category it as below:
            * [0] No ERC20 token received
            * [1] Small amount ERC20 token received
            * [2] Medium amount ERC20 token received
            * [3] Large amount ERC20 token received
            ''')

            st.subheader('Number of Created Contracts vs. :green[Target] visualization')
            fig, ax = plt.subplots(1,1, figsize=(15, 5))
            sns.countplot(data=df_EDA, x='Number of Created Contracts', hue='FLAG', palette='Greens')
            ax.set_yscale('log')
            plt.tight_layout()
            st.pyplot(fig)

        elif sidebar_eda == 'Scatterplot':
           st.title('Scatterplot 🪡')
           st.markdown('---')

           st.subheader('Time Difference >< :green[Average Min Between Received Transactions]')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='Time Diff between first and last (Mins)', y='Avg min between received tnx', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer transactions and the smallest of different transaction times further strengthen the suspicion of a fraudulent transaction.')

           st.subheader(':green[Total Transactions] >< Received Transactions')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='total transactions (including tnx to create contract)', y='Received Tnx', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer total transactions and received transactions further strengthen the suspicion of fraudulent activity.')

           st.subheader('Unique Received From Addresses >< :green[Received Transactions]')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='Unique Received From Addresses', y='Received Tnx', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer transactions and the smallest number of different wallet addresses further strengthen the suspicion of a fraudulent transaction.')

           st.subheader(':green[Unique Sent To Addresses] >< Sent Transactions')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='Unique Sent To Addresses', y='Sent tnx', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer transactions and the smallest number of different wallet addresses further strengthen the suspicion of a fraudulent transaction.')

           st.subheader('ERC20 Unique Received Address >< :green[Total ERC20 Transactions]')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='ERC20 uniq rec addr', y='Total ERC20 tnxs', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer ERC20 transactions on assets sent and received by different address accounts tend to reveal fraud.')

           st.subheader(':green[ERC20 Unique Sent Address] >< Total ERC20 Transactions')
           plt.subplots(figsize=(14, 7))
           sns.scatterplot(data=df, x='ERC20 uniq sent addr', y='Total ERC20 tnxs', hue='FLAG', palette=['olivedrab', 'orange'])
           st.pyplot()
           st.write('The fewer ERC20 transactions on assets sent and received by different address accounts tend to reveal fraud.')


#Modelling
elif sidebar == 'Machine Learning Experiment':
    list_balancing = ['Random Under Sampling', 'SMOTE Over Sampling']
    #Side bar
    sidebar_balance = st.sidebar.selectbox("Imbalanced Data Handling", list_balancing)
    df_ml = copy.deepcopy(df)

    # Dropping correlated data above threshold 0.8
    cor_matrix = df_ml.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    dropped_cols = set()
    for feature in upper_tri.columns:
        if any(upper_tri[feature] > 0.8) and feature != 'total transactions (including tnx to create contract)':
              dropped_cols.add(feature)
    df_ml = df_ml.drop(dropped_cols,axis=1)

    # Cross Validation
    # Set the number of folds for StratifiedKFold
    n_splits = 10
    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    if sidebar_balance == 'Random Under Sampling':
        st.title(':green[Random Under Sampling] Data Modelling')
        st.markdown('---')
        X = df_ml.drop(['FLAG'],axis=1)
        y = df_ml['FLAG']

        # Splitting dataset baseline
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_ml['FLAG'])
        # X_train2, y_train2 = can be fitted to ML as baseline

        # Undersampling with random under sampling
        X_random_under, y_random_under = under_sampling.RandomUnderSampler().fit_resample(X_train2, y_train2)

        # Turn the values into an array for feeding the classification algorithms (dataset without normalization)
        X_train = X_train2.values
        X_test = X_test2.values
        y_train = y_train2.values
        y_test = y_test2.values

        X_train_random_under = X_random_under.values
        y_train_random_under = y_random_under.values

        def train_model_dtc(max_depth,max_features):
            dtc = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,random_state=42)
            dtc.fit(X_train_random_under,y_train_random_under)
            return dtc
        
        def train_model_rf(n_estimators,max_depth,max_features,bootstrap):
           rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,bootstrap=bootstrap,random_state=42)
           rf.fit(X_train_random_under,y_train_random_under)
           return rf
        
        def train_model_gbt(n_estimators,max_depth,max_features):
            gbt = GradientBoostingClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42)
            gbt.fit(X_train_random_under,y_train_random_under)
            return gbt
    
        def train_model_xgbt(n_estimators,max_depth,max_features):
            xgbt = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42, verbose=False)
            xgbt.fit(X_train_random_under,y_train_random_under)
            return xgbt
        
        def train_model_lgbm(n_estimators,max_depth,max_features):
            lgbm = LGBMClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42)
            lgbm.fit(X_train_random_under,y_train_random_under)
            return lgbm
        
        def train_model_cat(n_estimators,max_depth):
            cat = CatBoostClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42, verbose=False)
            cat.fit(X_train_random_under,y_train_random_under)
            return cat

        with st.form('Train Model'):
            options = st.selectbox('Model: ', options=['Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier',
                                             'XGBoost Classifier', 'Light GBM Classifier', 'Cat Boost Classifier'])

            if options == 'Decision Tree Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])

                submitted = st.form_submit_button('Train')

                if submitted:
                    dtc_class = train_model_dtc(max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = dtc_class.predict(X_test)
                    y_train_preds = dtc_class.predict(X_train_random_under)
                    y_train_proba = dtc_class.predict_proba(X_train_random_under)
                    y_test_proba = dtc_class.predict_proba(X_test)

                  # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')

                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))

                        st.markdown("### Confusion :green[Matrix]")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)

                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)

                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')

                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)

                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)

                    # Feature Importance
                    st.markdown("### Feature :green[Importance]")
                    feature_importance = dtc_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    st.pyplot(ax_imp.figure, use_container_width=True)

            elif options == 'Random Forest Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])
                    bootstrap = st.checkbox("Bootstrap")
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    rf_class = train_model_rf(n_estimators,max_depth,max_features,bootstrap)
                    # Make prediciton and evaluation
                    y_test_preds = rf_class.predict(X_test)
                    y_train_preds = rf_class.predict(X_train_random_under)
                    y_train_proba = rf_class.predict_proba(X_train_random_under)
                    y_test_proba = rf_class.predict_proba(X_test)
  
                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')
  
                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                        st.markdown("### Confusion :green[Matrix]")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)
  
                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')
  
                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)
  
                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)
  
                    # Feature Importance
                    st.markdown("### Feature :green[Importance]")
                    feature_importance = rf_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
  
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif options == 'Gradient Boosting Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    gbt_class = train_model_gbt(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = gbt_class.predict(X_test)
                    y_train_preds = gbt_class.predict(X_train_random_under)
                    y_train_proba = gbt_class.predict_proba(X_train_random_under)
                    y_test_proba = gbt_class.predict_proba(X_test)
  
                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')
  
                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                        st.markdown("### :green[Confusion] Matrix")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)
  
                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')
  
                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)
  
                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)
  
                    # Feature Importance
                    st.markdown("### :green[Feature] Importance")
                    feature_importance = gbt_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
  
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif options == 'XGBoost Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    xgbt_class = train_model_xgbt(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = xgbt_class.predict(X_test)
                    y_train_preds = xgbt_class.predict(X_train_random_under)
                    y_train_proba = xgbt_class.predict_proba(X_train_random_under)
                    y_test_proba = xgbt_class.predict_proba(X_test)
  
                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')
  
                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                        st.markdown("### Confusion :green[Matrix]")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)
  
                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')
  
                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)
 
                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)
    
                    # Feature Importance
                    st.markdown("### Feature :green[Importance]")
                    feature_importance = xgbt_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
    
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif options == 'Light GBM Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])

                submitted = st.form_submit_button('Train')
  
                if submitted:
                    lgbm_class = train_model_lgbm(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = lgbm_class.predict(X_test)
                    y_train_preds = lgbm_class.predict(X_train_random_under)
                    y_train_proba = lgbm_class.predict_proba(X_train_random_under)
                    y_test_proba = lgbm_class.predict_proba(X_test)
  
                # Evaluation
                with col2:
                    col21,col22 = st.columns(2,gap='medium')

                    with col21:
                        st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                    with col22:
                        st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
                    
                    st.markdown("### :green[Confusion] Matrix")
                    figure = plt.figure(figsize=(6,6))
                    ax = figure.add_subplot(111)
                    skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                    st.pyplot(figure,use_container_width=True)

                st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                col31,col32 = st.columns(2,gap='medium')
   
                with col31:
                    figure_roc = plt.figure(figsize=(8,6))
                    ax_roc = figure_roc.add_subplot(111)
                    skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                    st.pyplot(figure_roc,use_container_width=True)

                with col32:
                    figure_rc = plt.figure(figsize=(8,6))
                    ax_rc = figure_rc.add_subplot(111)
                    skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                    st.pyplot(figure_rc,use_container_width=True)
  
                # Feature Importance
                st.markdown("### :green[Feature] Importance")
                feature_importance = lgbm_class.feature_importances_
                importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
                plt.figure(figsize=(10, 20))
                ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
            
                st.pyplot(ax_imp.figure, use_container_width=True)


            elif options == 'Cat Boost Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    cat_class = train_model_cat(n_estimators,max_depth)
                    # Make prediciton and evaluation
                    y_test_preds = cat_class.predict(X_test)
                    y_train_preds = cat_class.predict(X_train_random_under)
                    y_train_proba = cat_class.predict_proba(X_train_random_under)
                    y_test_proba = cat_class.predict_proba(X_test)
  
                # Evaluation
                with col2:
                    col21,col22 = st.columns(2,gap='medium')
  
                    with col21:
                        st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_random_under,y_train_preds))))
                    with col22:
                        st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                    st.markdown("### :green[Confusion] Matrix")
                    figure = plt.figure(figsize=(6,6))
                    ax = figure.add_subplot(111)
                    skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                    st.pyplot(figure,use_container_width=True)
  
                st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                col31,col32 = st.columns(2,gap='medium')
  
                with col31:
                    figure_roc = plt.figure(figsize=(8,6))
                    ax_roc = figure_roc.add_subplot(111)
                    skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                    st.pyplot(figure_roc,use_container_width=True)
  
                with col32:
                    figure_rc = plt.figure(figsize=(8,6))
                    ax_rc = figure_rc.add_subplot(111)
                    skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                    st.pyplot(figure_rc,use_container_width=True)
  
                # Feature Importance
                st.markdown("### Feature :green[Importance]")
                feature_importance = cat_class.feature_importances_
                importance_df = pd.DataFrame({'Feature': X_random_under.columns, 'Importance': feature_importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
   
                plt.figure(figsize=(10, 20))
                ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
  
                st.pyplot(ax_imp.figure, use_container_width=True)

    elif sidebar_balance == 'SMOTE Over Sampling':
        st.title(':green[SMOTE Over Sampling] Data Modelling')
        st.markdown('---')
        X = df_ml.drop(['FLAG'],axis=1)
        y = df_ml['FLAG']     

        # Splitting dataset baseline
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_ml['FLAG'])
        # X_train2, y_train2 = can be fitted to ML as baseline

        # Overampling with smote
        X_train_over, y_train_over = over_sampling.SMOTE().fit_resample(X_train2, y_train2)

        # Turn the values into an array for feeding the classification algorithms (dataset without normalization)
        X_train = X_train2.values
        X_test = X_test2.values
        y_train = y_train2.values
        y_test = y_test2.values

        X_train_over_smote = X_train_over.values
        y_train_over_smote = y_train_over.values

        def train_model_dtc(max_depth,max_features):
            dtc = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,random_state=42)
            dtc.fit(X_train_over_smote,y_train_over_smote)
            return dtc
        
        def train_model_rf(n_estimators,max_depth,max_features,bootstrap):
           rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,bootstrap=bootstrap,random_state=42)
           rf.fit(X_train_over_smote,y_train_over_smote)
           return rf
        
        def train_model_gbt(n_estimators,max_depth,max_features):
            gbt = GradientBoostingClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42)
            gbt.fit(X_train_over_smote,y_train_over_smote)
            return gbt
    
        def train_model_xgbt(n_estimators,max_depth,max_features):
            xgbt = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42, verbose=False)
            xgbt.fit(X_train_over_smote,y_train_over_smote)
            return xgbt
        
        def train_model_lgbm(n_estimators,max_depth,max_features):
            lgbm = LGBMClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,random_state=42)
            lgbm.fit(X_train_over_smote,y_train_over_smote)
            return lgbm
        
        def train_model_cat(n_estimators,max_depth):
            cat = CatBoostClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42, verbose=False)
            cat.fit(X_train_over_smote,y_train_over_smote)
            return cat

        with st.form('Train Model'):
            #List Choice
            select_box = ['Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier',
                          'XGBoost Classifier', 'Light GBM Classifier', 'Cat Boost Classifier']
            #Box To Select
            box = st.selectbox(
                "Model", select_box
            )

            if box == 'Decision Tree Classifier':
                    col1,col2 = st.columns(2,gap='medium')
                    with col1:
                        max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                        max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])

                    submitted = st.form_submit_button('Train')

                    if submitted:
                        dtc_class = train_model_dtc(max_depth,max_features)
                        # Make prediciton and evaluation
                        y_test_preds = dtc_class.predict(X_test)
                        y_train_preds = dtc_class.predict(X_train_over_smote)
                        y_train_proba = dtc_class.predict_proba(X_train_over_smote)
                        y_test_proba = dtc_class.predict_proba(X_test)

                      # Evaluation
                        with col2:
                            col21,col22 = st.columns(2,gap='medium')

                            with col21:
                                st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                            with col22:
                                st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))

                            st.markdown("### Confusion :green[Matrix]")
                            figure = plt.figure(figsize=(6,6))
                            ax = figure.add_subplot(111)

                            skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                            st.pyplot(figure,use_container_width=True)

                        st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                        col31,col32 = st.columns(2,gap='medium')

                        with col31:
                            figure_roc = plt.figure(figsize=(8,6))
                            ax_roc = figure_roc.add_subplot(111)
                            skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                            st.pyplot(figure_roc,use_container_width=True)

                        with col32:
                            figure_rc = plt.figure(figsize=(8,6))
                            ax_rc = figure_rc.add_subplot(111)
                            skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                            st.pyplot(figure_rc,use_container_width=True)

                        # Feature Importance
                        st.markdown("### Feature :green[Importance]")
                        feature_importance = dtc_class.feature_importances_
                        importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                        plt.figure(figsize=(10, 20))
                        ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                        plt.xlabel('Importance')
                        plt.ylabel('Feature')
                        st.pyplot(ax_imp.figure, use_container_width=True)

            elif box == 'Random Forest Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])
                    bootstrap = st.checkbox("Bootstrap")
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    rf_class = train_model_rf(n_estimators,max_depth,max_features,bootstrap)
                    # Make prediciton and evaluation
                    y_test_preds = rf_class.predict(X_test)
                    y_train_preds = rf_class.predict(X_train_over_smote)
                    y_train_proba = rf_class.predict_proba(X_train_over_smote)
                    y_test_proba = rf_class.predict_proba(X_test)
  
                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')
  
                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                        st.markdown("### :green[Confusion] Matrix")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)
  
                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')
  
                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)
  
                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)
  
                    # Feature Importance
                    st.markdown("### Feature :green[Importance]")
                    feature_importance = rf_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
  
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif box == 'Gradient Boosting Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])

                submitted = st.form_submit_button('Train')

                if submitted:
                    gbt_class = train_model_gbt(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = gbt_class.predict(X_test)
                    y_train_preds = gbt_class.predict(X_train_over_smote)
                    y_train_proba = gbt_class.predict_proba(X_train_over_smote)
                    y_test_proba = gbt_class.predict_proba(X_test)

                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')

                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))

                        st.markdown("### Confusion :green[Matrix]")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)

                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')

                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)

                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)

                    # Feature Importance
                    st.markdown("### :green[Feature] Importance")
                    feature_importance = gbt_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif box == 'XGBoost Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    xgbt_class = train_model_xgbt(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = xgbt_class.predict(X_test)
                    y_train_preds = xgbt_class.predict(X_train_over_smote)
                    y_train_proba = xgbt_class.predict_proba(X_train_over_smote)
                    y_test_proba = xgbt_class.predict_proba(X_test)
  
                    # Evaluation
                    with col2:
                        col21,col22 = st.columns(2,gap='medium')
  
                        with col21:
                            st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                        with col22:
                            st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                        st.markdown("### :green[Confusion] Matrix")
                        figure = plt.figure(figsize=(6,6))
                        ax = figure.add_subplot(111)
                        skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                        st.pyplot(figure,use_container_width=True)
  
                    st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                    col31,col32 = st.columns(2,gap='medium')
  
                    with col31:
                        figure_roc = plt.figure(figsize=(8,6))
                        ax_roc = figure_roc.add_subplot(111)
                        skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                        st.pyplot(figure_roc,use_container_width=True)
 
                    with col32:
                        figure_rc = plt.figure(figsize=(8,6))
                        ax_rc = figure_rc.add_subplot(111)
                        skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                        st.pyplot(figure_rc,use_container_width=True)
    
                    # Feature Importance
                    st.markdown("### Feature :green[Importance]")
                    feature_importance = xgbt_class.feature_importances_
                    importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 20))
                    ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
    
                    st.pyplot(ax_imp.figure, use_container_width=True)
  
            elif box == 'Light GBM Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
                    max_features = st.selectbox('Max Features: ', options=['sqrt', 'log2', None])

                submitted = st.form_submit_button('Train')
  
                if submitted:
                    lgbm_class = train_model_lgbm(n_estimators,max_depth,max_features)
                    # Make prediciton and evaluation
                    y_test_preds = lgbm_class.predict(X_test)
                    y_train_preds = lgbm_class.predict(X_train_over_smote)
                    y_train_proba = lgbm_class.predict_proba(X_train_over_smote)
                    y_test_proba = lgbm_class.predict_proba(X_test)
  
                # Evaluation
                with col2:
                    col21,col22 = st.columns(2,gap='medium')

                    with col21:
                        st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                    with col22:
                        st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
                    
                    st.markdown("### Confusion :green[Matrix]")
                    figure = plt.figure(figsize=(6,6))
                    ax = figure.add_subplot(111)
                    skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                    st.pyplot(figure,use_container_width=True)

                st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                col31,col32 = st.columns(2,gap='medium')
   
                with col31:
                    figure_roc = plt.figure(figsize=(8,6))
                    ax_roc = figure_roc.add_subplot(111)
                    skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                    st.pyplot(figure_roc,use_container_width=True)

                with col32:
                    figure_rc = plt.figure(figsize=(8,6))
                    ax_rc = figure_rc.add_subplot(111)
                    skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                    st.pyplot(figure_rc,use_container_width=True)
  
                # Feature Importance
                st.markdown("### :green[Feature] Importance")
                feature_importance = lgbm_class.feature_importances_
                importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
                plt.figure(figsize=(10, 20))
                ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
            
                st.pyplot(ax_imp.figure, use_container_width=True)


            elif box == 'Cat Boost Classifier':
                col1,col2 = st.columns(2,gap='medium')
                with col1:
                    n_estimators = st.slider('n_estimators: ', min_value=100, max_value=1000)
                    max_depth = st.slider('Max Depth: ', min_value=2, max_value=20)
  
                submitted = st.form_submit_button('Train')
  
                if submitted:
                    cat_class = train_model_cat(n_estimators,max_depth)
                    # Make prediciton and evaluation
                    y_test_preds = cat_class.predict(X_test)
                    y_train_preds = cat_class.predict(X_train_over_smote)
                    y_train_proba = cat_class.predict_proba(X_train_over_smote)
                    y_test_proba = cat_class.predict_proba(X_test)
  
                # Evaluation
                with col2:
                    col21,col22 = st.columns(2,gap='medium')
  
                    with col21:
                        st.metric('Train F1 Score', value="{:.2f} %".format(100*(f1_score(y_train_over_smote,y_train_preds))))
                    with col22:
                        st.metric('Test F1 Score', value="{:.2f} %".format(100*(f1_score(y_test,y_test_preds))))
  
                    st.markdown("### Confusion :green[Matrix]")
                    figure = plt.figure(figsize=(6,6))
                    ax = figure.add_subplot(111)
                    skplt.metrics.plot_confusion_matrix(y_test,y_test_preds,ax=ax,cmap='Greens')
                    st.pyplot(figure,use_container_width=True)
  
                st.markdown("### :green[ROC] and :green[Precission-Recall] Curves")
                col31,col32 = st.columns(2,gap='medium')
  
                with col31:
                    figure_roc = plt.figure(figsize=(8,6))
                    ax_roc = figure_roc.add_subplot(111)
                    skplt.metrics.plot_roc(y_test,y_test_proba,ax=ax_roc)
                    st.pyplot(figure_roc,use_container_width=True)
  
                with col32:
                    figure_rc = plt.figure(figsize=(8,6))
                    ax_rc = figure_rc.add_subplot(111)
                    skplt.metrics.plot_precision_recall(y_test,y_test_proba,ax=ax_rc)
                    st.pyplot(figure_rc,use_container_width=True)
  
                # Feature Importance
                st.markdown("### Feature :green[Importance]")
                feature_importance = cat_class.feature_importances_
                importance_df = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': feature_importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
   
                plt.figure(figsize=(10, 20))
                ax_imp = sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
  
                st.pyplot(ax_imp.figure, use_container_width=True)
