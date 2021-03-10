class Kline:

    def __init__(self, get):
        self.openTime = get[0]
        self.open = float(get[1])
        self.high = float(get[2])
        self.low = float(get[3])
        self.close = float(get[4])
        self.volume = float(get[5])
        self.closeTime = get[6]
        self.avg = self.open + self.high + self.low + self.close
        self.avg /= 4


    def to_dict(self):
        return {
            'openTime': self.openTime,
            'open': self.openTime,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'closeTime': self.closeTime,
        }