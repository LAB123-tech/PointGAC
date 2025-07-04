class AverageMeter(object):
    def __init__(self, items):
        self.items = items
        self.n_items = len(items)
        self.val = [0] * self.n_items
        self.sum = [0] * self.n_items
        self.count = [0] * self.n_items

    def update(self, values, idx=None):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self.val[idx] = v
                self.sum[idx] += v
                self.count[idx] += 1
        else:
            self.val[idx] = values
            self.sum[idx] += values
            self.count[idx] += 1

    def val(self, idx=None):
        if idx is None:
            return [self.val[i] for i in range(self.n_items)]
        else:
            return self.val[idx]

    def count(self, idx=None):
        if idx is None:
            return [self.count[i] for i in range(self.n_items)]
        else:
            return self.count[idx]

    def avg(self, idx=None):
        if idx is None:
            return [self.sum[i] / self.count[i] for i in range(self.n_items)]
        else:
            return self.sum[idx] / self.count[idx]


class Acc_Metric:
    def __init__(self, acc=0.):
        self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def less_than(self, other):
        if self.acc <= other.acc:
            return True
        else:
            return False

    def state_dict(self):
        acc_dict = dict()
        acc_dict['acc'] = self.acc
        return acc_dict
