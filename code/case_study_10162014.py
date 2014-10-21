from __future__ import print_function, division

from matplotlib import pyplot as plt
import numpy as np
import thinkbayes2
import thinkplot

class Text(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood"""
        lam1 = hypo[0]
        lam2 = hypo[1]
        tau = hypo[2]
        
        like = thinkbayes2.EvalPoissonPmf(len(data),lam1)
        return like

def main():
    # Texting Data
    data = [12, 24, 8, 24, 6, 35, 12, 11, 13, 11, 22, 22, 11, 56, 11, 19, 29, 
    5, 19, 12, 22, 12, 18, 72, 32, 8, 7, 13, 19, 23, 27, 20, 5, 13, 18, 23, 27, 20, 
    6, 18, 13, 10, 14, 6, 16, 15, 8, 2, 15, 15, 19, 70, 59, 7, 53, 22, 21, 31, 19, 
    11, 17, 20, 12, 35, 16, 23, 16, 3, 2, 31, 30, 13, 27, 38, 37, 3, 14, 13, 22]
    data_size = len(data) #79
    
    alpha = 1.0 / (sum(data)/float(len(data)))
    lam1 = thinkbayes2.MakeExponentialPmf(alpha,max(data), data_size)
    lam2 = thinkbayes2.MakeExponentialPmf(alpha,max(data), data_size)
    tau = thinkbayes2.MakeUniformPmf(0, max(data), data_size)

    #hypos = np.linspace(0, max(data), 79)
    hypos = [lam1,lam2,tau] 
    suite = Text(hypos)
    suite.Update(data)
    print('posterior mean', suite.Mean())

    thinkplot.Pmf(suite)
    thinkplot.Show(xlabel='FILLERTEXT',ylabel='PMF')

    """
    # Plot of Texting Data for Visualization (not 100% the same as the example, but close enough)
    plt.bar(np.arange(data_size), data, color="#348ABD")
    plt.xlabel("Time (days)")
    plt.ylabel("count of text-msgs received")
    plt.title("Did the user's texting habits change over time?")
    plt.xlim(0, data_size);
    plt.show()
    """
    
if __name__ == '__main__':
    main()
