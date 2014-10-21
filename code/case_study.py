from __future__ import print_function, division

from matplotlib import pyplot as plt
import numpy as np
import thinkbayes2
import thinkplot

class Text(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood"""
        x = data
        lam = hypo / 90
        like = thinkbayes2.EvalExponentialPdf(x, lam)
        return like

def main():
    # Texting Data
    count_data = [12, 24, 8, 24, 6, 35, 12, 11, 13, 11, 22, 22, 11, 56, 11, 19, 29, 
    5, 19, 12, 22, 12, 18, 72, 32, 8, 7, 13, 19, 23, 27, 20, 5, 13, 18, 23, 27, 20, 
    6, 18, 13, 10, 14, 6, 16, 15, 8, 2, 15, 15, 19, 70, 59, 7, 53, 22, 21, 31, 19, 
    11, 17, 20, 12, 35, 16, 23, 16, 3, 2, 31, 30, 13, 27, 38, 37, 3, 14, 13, 22]
    n_count_data = len(count_data) #79
    
    # Plot of Texting Data for Visualization (not 100% the same as the example, but close enough)
    plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
    plt.xlabel("Time (days)")
    plt.ylabel("count of text-msgs received")
    plt.title("Did the user's texting habits change over time?")
    plt.xlim(0, n_count_data);
    plt.show()
    
    rand = np.random.rand
    
    def lambda_(t, l_1, l_2, n_count_data):
        out = np.zeros(n_count_data)
        out[:t] = l_1  # lambda before tau is lambda1
        out[t:] = l_2  # lambda after (and including) tau is lambda2
        return out

    
    alpha = 1.0 / count_data.mean()  # Recall count_data is the variable that holds our txt counts
    lambda_1 = thinkbayes2.MakeExponentialPmf(alpha,max(count_data), 79)
    lambda_2 = thinkbayes2.MakeExponentialPmf(alpha,max(count_data), 79)
    tau = thinkbayes2.MakeUniformPmf(0, n_count_data)
    observation = thinkbayes2.MakePoissonPmf(lambda_(tau,lambda_1,lambda_2,n_count_data), max(count_data))
   
    hypos = np.linspace(0, max(count_data), 79) 
    suite = Txt(hypos)


if __name__ == '__main__':
    main()
