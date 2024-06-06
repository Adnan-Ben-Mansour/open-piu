import numpy as np
import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData

from src.core.charts import Chart


def nb_charts(path):
    return len(simfile.open(path).charts)


def read_ssc(path:str, idx=0):
    sfile = simfile.open(path) # load simfile
    chart = sfile.charts[idx]  # get the indexed chart
    
    max_denominator = 0
    max_numerator = 1

    single = True # single or double

    for note in NoteData(chart):
        max_denominator = max(max_denominator, note.beat.denominator)
        max_num = max(max_numerator, note.beat.numerator/note.beat.denominator)
        
        if note.column > 4:
            single = False

    length = int(max_num*max_denominator)+1

    split_timing = TimingData(sfile, chart)
    new_chart = Chart(title=sfile.title,
                      author=sfile.artist,
                      level=chart.meter,
                      tempo=split_timing.bpms,
                      beat=max_denominator,
                      single=single,
                      length=length)

    tmp_data = np.zeros((length, 40), dtype=np.int64)

    for note in sorted(NoteData(chart), key=lambda n: n.beat.numerator*(max_denominator//n.beat.denominator) + int(n.note_type == NoteType.TAIL)):
        position = note.beat.numerator * (max_denominator // note.beat.denominator)

        if note.note_type == NoteType.TAP:
            tmp_data[position, note.column] = 1
        
        if note.note_type == NoteType.HOLD_HEAD:
            tmp_data[position, 10+note.column] = 1

        if note.note_type == NoteType.TAIL:
            tmp_data[position, 20+note.column] = 1
            
            p = position
            while (p>=0) and (tmp_data[p, 10+note.column]==0):
                tmp_data[p, 30+note.column] = 1
                p -= 1
            if tmp_data[p, 10+note.column] == 1:
                tmp_data[p, 30+note.column] = 1
            if p == -1:
                raise Exception("[Warning]: HOLD_TAIL without HOLD_HEAD.")
    
    new_chart.data = tmp_data[:, np.array([(i<10 or i>=30) for i in range(40)])]
    if np.any(new_chart.data.sum(axis=1) > 4):
        raise Exception("[Warning]: Too much notes at the same time.")

    return new_chart
