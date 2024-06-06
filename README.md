# OpenPIU

Open-source tools for the rythm game Pump It Up. 


## Content
- displayer.py -> chart rendering from simfiles/custom charts in real time.
- 


## To-Do List
- [x] Read and display charts with single tempo.
- [ ] Generate k notes from n previous notes with a good accuracy.
- [ ] Detect level from n notes (or the whole song). 
- [ ] Take into account BPM variation. 
- [ ] Sync display with music.
- [ ] Condition chart generation by the style of the artist. 
- [ ] Compute metrics that explain the level of a song.
- [ ] Combine this work with piucenter.com work.


## Generator
- [ ] Beat generator (number of steps at each beat/holds).
- [ ] Probability for each step to come after multiple steps. (tempo/arrow decomposition)
- [ ] Fusion.


## Architecture
- `main.py` -> make_dataset, train_model, generate_chart
- `README.md` -> this
- `src/` -> all the code
  - `src/core/` -> main code
  - `src/models/` -> pytorch models (definitions)
  - `src/weights/` -> store pytorch weights
  - `src/display/`-> tools to display charts
- `notes/` -> tex files explaining the project
  - `notes/main/` -> project report
- `data/` -> all the data
  - `data/skins/` -> skins for arrows 
  - `data/dataset/` -> loaded charts ready for use
  - `data/PIU-Simfiles-main/` -> raw dataset with SSC files from the game
- `outputs/` -> generated data
  - `outputs/charts/` -> loadable charts ready to use