class Timestamp:
    """ A class representing a timestamp as found in a .eimu file, i.e. a single point in time. """
    def __init__(self, time : int, ttype : str) -> None:
        self.time = time
        self.type = ttype
        
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, Timestamp):
            return False
        return self.time == other.time and self.type == other.type
    