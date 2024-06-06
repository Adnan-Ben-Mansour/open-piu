import numpy as np
import PIL.Image as Image


def data_to_png(data, filename="path.png"):
    # data: (N, 20)
    print(f"[ERROR]: data_to_png() is not implemented yet.")
    exit(1) 
    

    data = data.T[:10]
    colors_blue = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)
    colors_red = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    colors_yellow = np.array([[0, 0, 0], [255, 255, 0]], dtype=np.uint8)
    print(data.dtype, data.shape)
    to_combine = []
    for i, c in enumerate('bryrbbryrb'):
        if c=='b':
            to_combine.append(colors_blue[data[i, :]])
        elif c=='r':
            to_combine.append(colors_red[data[i, :]])
        elif c=='y':
            to_combine.append(colors_yellow[data[i, :]])

    arr = np.stack(to_combine, 0).swapaxes(0, 1)
    im = Image.fromarray(arr)
    im.save(filename)
    

def chart_to_png(chart, filename="path.png"):
    return data_to_png(chart.data, filename)