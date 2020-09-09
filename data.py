import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def exponential(x, a, b, c):
    return a*np.exp(-b*x) + c

data = pd.read_csv("roundData.csv")

max_rounds = data["Round"].max()
games = data["Game"].max()

fill_data = {"Game": [], "Round": [], "Deadwood": []}

for i in range(1,games+1):
    current_rounds = data.loc[data["Game"] == i,"Round"].max()
    min_dwood = data.loc[(data["Game"] == i) & (data["Round"] == current_rounds),"Deadwood"].to_numpy()[0]
    if  min_dwood <= 10 and current_rounds < max_rounds:
        for j in range(current_rounds +1,max_rounds+1):
            fill_data["Game"].append(i)
            fill_data["Round"].append(j)
            fill_data["Deadwood"].append(min_dwood)
FD = pd.DataFrame(fill_data)
data = data.append(FD,ignore_index=True).sort_values(by=["Game","Round"])

rounds = data["Round"].to_numpy()
unique_rounds = np.unique(rounds)
Deadwood = data["Deadwood"].to_numpy()

plt.scatter(rounds,Deadwood, color="BLUE")
plt.ylabel("Deadwood")
plt.xlabel("Round #")
plt.title("Deadwood per Round")

avg = data.groupby("Round")["Deadwood"].mean()
averages = data.groupby("Round")["Deadwood"].mean().to_numpy()


slope, intercept, rvalue, pvalue, stderr = stats.linregress(unique_rounds,averages)
plt.plot(unique_rounds,slope*unique_rounds + intercept, color="RED", linewidth=5, linestyle='--')
popt, pcov = curve_fit(exponential,unique_rounds,averages)
plt.plot(unique_rounds,exponential(unique_rounds,*popt), color="PURPLE", linewidth=5)
plt.plot(unique_rounds,10*np.ones(len(unique_rounds)),color="BLACK", linewidth=5, linestyle=':')

plt.scatter(unique_rounds,averages,color="ORANGE")
plt.show()

print("Num Sub-games played: %d" %(np.count_nonzero(rounds == 1)))
print("LINEAR REGRESSION correlation: %f pvalue: %f stderr: %f" %(rvalue,pvalue,stderr))
print("Exponential Fit covariance: ",pcov)
print("Exponential data:",popt)