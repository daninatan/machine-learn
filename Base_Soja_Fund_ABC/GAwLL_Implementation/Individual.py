class Individual:
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = []
        else:
            self.chromosome = chromosome.copy()
