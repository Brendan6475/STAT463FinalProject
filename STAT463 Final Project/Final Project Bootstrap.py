import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats
import random


# The get_arr function returns an aggregated array binned to the specified number of bins.
def get_arr(data_list, bin_size_num, rounding=0):
    min_val = min(data_list)
    max_val = max(data_list)
    binsize = abs((max_val - min_val) / bin_size_num)
    # Call helper function to create the bins.
    bins = get_bins(min_val, min_val + binsize, binsize, bin_size_num, rounding=rounding)
    map_value_arr = []
    # Loop through the values in the provided list.
    for element in data_list:
        # Loop through each of the bins.
        for which_bin in bins:
            # If the data point falls in that bin, then append that data point to that bin.
            if which_bin[0] < element <= which_bin[1]:
                map_value_arr.append(which_bin[2])
            if which_bin == bins[0] and element == which_bin[0]:
                map_value_arr.append(which_bin[2])
    return map_value_arr


# Helper function that iteratively makes the bins.
def get_bins(min_value, max_value, bin_size, bin_size_num, rounding=0):
    # map value for each bin = (bin size / 2) + bin_min
    bins = []
    # Loop through the specified number of bins
    for num in range(bin_size_num):
        # Make the bins, taking into account the specified number of decimals to round to from the rounding variable.
        bins.append((round(min_value, rounding),
                     round(max_value, rounding),
                     round((bin_size / 2) + min_value, rounding)))
        min_value += bin_size
        max_value += bin_size
    return bins


# Read in the data and put the column of data we're using into a list.
df = pd.read_csv("../STAT463 Final Project/Student Exam Scores Datasets/exams.csv")
df = df["reading score"]
data = list(np.asarray(df))

# Steps to plot a histogram of the data with labels and a plot title.
plt.hist(data, bins=11)
plt.xlabel("Reading Test Scores")
plt.ylabel("Count")
plt.title("Reading Test Scores for 1,000 Students")
plt.show()

# Steps to make a table of aggregated scores with the number of occurrences, labeled "Count."
aggregated = get_arr(data, 11, rounding=0)
# print(aggregated)
count_dict = Counter(aggregated)
table = []
# Loop through the Counter object, using its dictionary properties to
# append the data to a table (which is actually lists of [x, y] within one main list).
for key in count_dict.keys():
    table.append([key, count_dict[key]])
# Sort the table from smallest x value to largest x value using the built-in sorted function.
table = sorted(table, key=lambda x: x[0])
# Convert the 2d list to a pandas DataFrame so that it is easier to view when printed.
table_df = pd.DataFrame(table, columns=["Score (%)", "Count"])
print(table_df)

# Find the skewness of the aggregated data using the built-in scipy.stats function skew.
skewness = stats.skew(np.asarray(aggregated))
print("The skewness of the dataset, taking the frequencies of each score into account, is:", skewness)
# Find the kurtosis of the aggregated data using the built-in scipy.stats function kurtosis
# NOTE: This kurtosis is calculated under Fisher's definition, so a normal distribution would have a kurtosis of 0.
kurtosis = stats.kurtosis(np.asarray(aggregated))
print("The kurtosis of the dataset, under Fisher's definition, is", kurtosis)


# Bootstrap method done here: for setup purposes, num_iterations can be changed to the desired number of resamples.
num_iterations = 100
# Instantiate the lists that the resample skewnesses and kurtoses will be stored in.
skewnesses = []
kurtoses = []
# Loop through the desired number of resamples.
for i in range(num_iterations):
    # Instantiate the resample list and loop through the list of aggregated data points.
    resample = []
    for j in range(len(aggregated)):
        # Get a random index to sample the data with.
        index = random.randint(0, len(aggregated)-1)
        # Grab the data point at the randomly generated index.
        resample.append(aggregated[index])
    # Outside the inner for loop, find the skewness and kurtosis of that resample.
    skewness_i = stats.skew(np.asarray(resample))
    kurtosis_i = stats.kurtosis(np.asarray(resample))
    # Append these resampled statistics to their respective lists.
    skewnesses.append(skewness_i)
    kurtoses.append(kurtosis_i)

# Steps to make a table of skewnesses and counts. This is very similar code to the code to make the previous DataFrame.
aggregated = get_arr(skewnesses, 11, rounding=4)
count_dict = Counter(aggregated)
table = []
for key in count_dict.keys():
    table.append([key, count_dict[key]])
table = sorted(table, key=lambda x: x[0])
table_df = pd.DataFrame(table, columns=["Skewness", "Count"])
print(table_df)

# Steps to make a table of kurtoses and counts (which uses the same code as above, just with kurtoses as the list).
aggregated = get_arr(kurtoses, 11, rounding=4)
# print(aggregated)
count_dict = Counter(aggregated)
table = []
for key in count_dict.keys():
    table.append([key, count_dict[key]])
table = sorted(table, key=lambda x: x[0])
table_df = pd.DataFrame(table, columns=["Kurtosis", "Count"])
print(table_df)

# Find the mean of each of the lists of resampled statistics using the built-in numpy function.
skewness_mean = np.mean(skewnesses)
kurtosis_mean = np.mean(kurtoses)
# Print the results
print("The mean of the skewnesses from the bootstrap distribution is", skewness_mean)
print("The mean of the kurtoses from the bootstrap distribution is", kurtosis_mean)

# Plot the skewnesses on a histogram.
plt.hist(skewnesses, bins=11)
plt.xlabel("Skewness value")
plt.ylabel("Count")
plt.title("Skewness Bootstrap Distribution")
plt.show()

# Plot the kurtoses on a histogram.
plt.hist(kurtoses, bins=11)
plt.xlabel("Kurtosis value")
plt.ylabel("Count")
plt.title("Kurtosis Bootstrap Distribution")
plt.show()
