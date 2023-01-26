class Timestamp:
    """ A class representing a timestamp as found in a .eimu file, i.e. a single point in time. """
    def __init__(self, time : int, ttype : str):
        self.time = time
        self.type = ttype