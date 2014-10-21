"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Homes(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: sell rate in homes per day
        data: interarrival time in days
        """
        k = data
        lam = hypo
        like = thinkbayes2.EvalExponentialPdf(k, lam)
        return like

    def MakePoissonPmf(lam,high):
        pmf = thinkbayes2.Pmf()
        for k in xrange(0,high+1):
            p = thinkbayes2.EvalPoissonPmf(k,lam)
            pmf.Set(k,p)
        pmf.Normalize()
        return pmf
        
    def MakeHomesPmf(suite):
        metapmf = thinkbayes2.Pmf()
        
        for lam,prob in suite.Items():
            pmf=thinkbayes2.MakePoissonPmf(lam,10)
            metapmf.Set(pmf,prob)
        
        mix=thinkbayes2.MakeMixture(metapmf)
        return mix


def main():
    hypos = numpy.linspace(0, 2, 201)
    suite = Homes(hypos)
    # the mean number of homes sold in a day was 2 per day
    mean_rate = 2

    # start with a prior based on the mean interarrival time
    suite.Update(2)
    thinkplot.Pdf(suite, label='prior')
    print('prior mean', suite.Mean())

    suite.Update(3)
    thinkplot.Pdf(suite, label='posterior 1')
    print('three homes', suite.Mean())


if __name__ == '__main__':
    main()
