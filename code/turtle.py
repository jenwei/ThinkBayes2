from __future__ import print_function, division
import thinkbayes2

class Turtle(thinkbayes2.Suite):
    """A map from hypotheetical Turtle to probability."""
    def Likelihood(self,data,hypo):
        """The likelihood of the data under the hypothesis.
        data: string 'green' or 'blue'
        hypo:integer Turtle number
        """
        p=hypo/3
        like = p if data=='green' else 1-p
        return like
        
def main():
    pmf=Turtle(range(0,4))
    pmf.Update('green')
    for hypo,prob in pmf.Items():
        print(hypo,prob)
        
if __name__ =='__main__':
    main()
                
