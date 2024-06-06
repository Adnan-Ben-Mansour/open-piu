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
        # yield (timestep, note, type)
        print(f"[ERROR]: Chart.read_notes() is not implemented yet.")
        exit(1) 
        tapnotes = np.argwhere(self.data==1)
        tapnotes = np.concatenate([np.ones((tapnotes.shape[0], 1), dtype=np.int64), tapnotes], axis=-1)
        
        begin_holds = np.argwhere(self.data == 2)
        begin_holds = np.concatenate([2*np.ones((begin_holds.shape[0], 1), dtype=np.int64), begin_holds], axis=-1)

        end_holds = np.argwhere(self.data == 3)
        end_holds = np.concatenate([3*np.ones((end_holds.shape[0], 1), dtype=np.int64), end_holds], axis=-1)

        for note in sorted(np.concatenate([tapnotes, begin_holds, end_holds], axis=0), key=lambda n: n[2]):
            yield note
        
        yield None
