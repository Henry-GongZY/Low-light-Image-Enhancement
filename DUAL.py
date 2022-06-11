from LIME import LIME

class DUAL:
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        self.limecore = LIME(iterations,alpha,rho,gamma,limestrategy)

    def load(self):
        pass

    def run(self):
        print('Using LIME for forward illumination!')

        print('Using LIME for reverse illumination!')

        print('Use multi-exposure image fusion to generate the result!')
