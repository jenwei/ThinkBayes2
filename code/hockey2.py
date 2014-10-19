"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Hockey2(thinkbayes2.Suite):
    """Represents hypotheses about."""
    
    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: goal rate (goals per game)
        data: goals scored in a game
        """
        goals = data
        lam = hypo
        like = thinkbayes2.EvalPoissonPmf(goals, lam)
        return like
        
    def PredictiveDist(self, label='pred'):
       """Computes the distribution of goals scored in a game.
       returns: new Pmf (mixture of Poissons)
       """
       metapmf = thinkbayes2.Pmf()
       for lam, prob in self.Items():
           pred = thinkbayes2.MakePoissonPmf(lam, 20) #20 is the upperbound (though it's not going to actually be reached)
           metapmf[pred] = prob
           
       mix = thinkbayes2.MakeMixture(metapmf, label=label)
       return mix

def main():
    hypos = numpy.linspace(0, 16, 201) #16 is the most goals ever scored in a game according to Google
    #2.875 is the mean of the average goals/game per team
    # start with a prior based on a pseudo observation chosen to yield the right prior mean
    suite1 = Hockey2(hypos, label='Blackhawks')
    suite1.Update(3.1) #3.1 is the average goals/game for the Blackhawks
    suite2 = Hockey2(hypos, label='Bruins')
    suite2.Update(2.65) #2.65 is the average goals/game for the Bruins
    
    # Update with the results the Stanley Cup
    #Game1
    suite1.Update(3)
    suite2.Update(3)
    #Game2
    suite1.Update(1)
    suite2.Update(1)
    #Game3
    suite1.Update(2)
    suite2.Update(0)
    #Game4
    suite1.Update(5)
    suite2.Update(5)
    #Game5
    suite1.Update(1)
    suite2.Update(3)
    #Game6
    suite1.Update(2)
    suite2.Update(3)

    print('Posterior mean of Blackhawks:', suite1.Mean())
    print('Posterior mean of Bruins:', suite2.Mean())

    # plot the posteriors
    thinkplot.PrePlot(2)
    thinkplot.Pdfs([suite1, suite2])
    thinkplot.Show(xlabel='Number of Goals',ylabel='Probability')
    

    # compute posterior prob that the Blackhawks are better than the Bruins
    post_prob = suite1 > suite2
    print('Posterior prob Blackhawks > Bruins:', post_prob)

    prior_odds = 1
    post_odds = post_prob / (1 - post_prob)
    k = post_odds / prior_odds
    print('Bayes factor', k)    
    

if __name__ == '__main__':
    main()
