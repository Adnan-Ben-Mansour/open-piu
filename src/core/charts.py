import numpy as np


class Chart:
    def __init__(self, title="Title", author="Author", level=-1, tempo=120, beat=1, single=True, length=1):
        self.title = title
        self.author = author
        self.level = level

        self.tempo = tempo
        self.beat = beat

        self.single = single
        self.length = length
        self.data = np.zeros((self.length, 20), dtype=np.int64)

    def read_notes(self):
        # yield (type, timestep, note) 
        tapnotes = np.argwhere(self.data[:, :10]==1)
        tapnotes = np.concatenate([np.ones((tapnotes.shape[0], 1), dtype=np.int64), tapnotes], axis=-1)
        
        begin_holds = np.argwhere(self.data[:, 10:] == 1)
        begin_holds = np.concatenate([2*np.ones((begin_holds.shape[0], 1), dtype=np.int64), begin_holds], axis=-1)

        end_holds = np.argwhere(self.data[:, 10:] == 0)
        end_holds = np.concatenate([3*np.ones((end_holds.shape[0], 1), dtype=np.int64), end_holds], axis=-1)
        
        open_holds = {}

        for note in sorted(np.concatenate([tapnotes, begin_holds, end_holds], axis=0), key=lambda n: n[1]+0.2*n[0]):
            if note[0] == 2:
                if note[1] not in open_holds:
                    open_holds[note[1]] = note
            if note[0] == 3:
                if note[1] in open_holds:
                    yield open_holds[note[1]]
                    del open_holds[note[1]]
                    yield note
            if note[0] == 1:
                yield note
        
        yield None
