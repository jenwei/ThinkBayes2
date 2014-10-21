"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Electorate(thinkbayes2.Suite):
    """Represents hypotheses about the state of the electorate."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: 
        data: 
        """
        
        a_hypo = hypo
        mean, std, measurement = data #mean = mean prior error
        e_hypo = measurement - a_hypo #error = e_hypo
        like = thinkbayes2.EvalNormalPdf(e_hypo, mean, std) #likelihood of error in the distribution of error
        return like
    
    def Lose(self):
        """Computes the probability of losing"""
        total = 0 
        for value, prob in self.Items():
            if value < 50:
                total += prob
        return total


def main():
    hypos = numpy.linspace(0, 100, 101)
    suite = Electorate(hypos)

    thinkplot.Pdf(suite, label='prior')

    data = 1.1, 3.7, 53
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior')
    thinkplot.Show()
    
    print(suite.Mean())
    print(suite.Std())
    print(suite.Lose())
    print(suite.probLess(50))


if __name__ == '__main__':
    main()
