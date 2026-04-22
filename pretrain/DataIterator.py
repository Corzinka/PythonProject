class DataIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = (len(data) + batch_size - 1) // batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            yield self.data[i * self.batch_size : (i + 1) * self.batch_size]