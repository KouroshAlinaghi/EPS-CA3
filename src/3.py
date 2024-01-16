import pandas as pd
import numpy as np, random
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import scipy.stats as stats

green = '#57cc99'
red = '#e56b6f'
blue = '#22577a'
yellow = '#ffca3a'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

df = pd.read_csv('FIFA2020.csv', encoding="ISO-8859-1")

df['dribbling'].fillna(df['dribbling'].mean(), inplace=True)
df['pace'].fillna(df['pace'].mean(), inplace=True)

age = df['age']

q0 = np.min(age)
q1 = np.percentile(age, 25)
median = np.median(age)
mean = np.mean(age)
q3 = np.percentile(age, 75)
q4 = np.max(age)

bplot = plt.boxplot(age, vert=False, showfliers=True, whis=[0, 100], patch_artist=True)
bplot['boxes'][0].set_facecolor(red)

plt.grid(True)
plt.show()

bplot = plt.boxplot(age, vert=False, patch_artist=True)
bplot['boxes'][0].set_facecolor(blue)

plt.grid(True)
plt.show()

def compare(n):
    weight_samples = np.random.choice(df['weight'], n, replace=False)

    mean = np.mean(weight_samples)
    var = np.var(weight_samples)
    sigma = var ** 0.5

    sample_normal_data = np.random.normal(mean, sigma, n)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    stats.probplot(sample_normal_data, dist="norm", plot=ax1)
    stats.probplot(weight_samples, dist="norm", plot=ax2)

    ax1.set_title("Sample normal data, n = {}".format(n))
    ax2.set_title("Sample of weights, n = {}".format(n))
    ax1.grid(True)
    ax2.grid(True)
    plt.show()

    statistic, p_value = stats.shapiro(sample_normal_data)
    statistic, p_value = stats.shapiro(weight_samples)

compare(100)
compare(500)
compare(2000)

def poisson_comp(lam, n):
    pois_data = poisson.rvs(lam, size=n)
    norm_data = np.random.normal(lam, lam ** -0.5, n)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    stats.probplot(pois_data, dist="norm", plot=ax1)
    stats.probplot(norm_data, dist="norm", plot=ax2)

    ax1.set_title('X ~ Pois({}), n = {}'.format(lam, n))
    ax2.set_title('Y ~ N({}, {:.3g}), n = {}'.format(lam, lam ** -0.5, n))
    ax1.grid(True)
    ax2.grid(True)
    plt.show()

    statistic, p_value = stats.shapiro(norm_data)
    statistic, p_value = stats.shapiro(pois_data)

poisson_comp(3, 5000)
poisson_comp(3, 5)
poisson_comp(3, 50)
poisson_comp(3, 500)
