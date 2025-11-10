<h1>Instructions</h1>

<h2>How to run the scripts</h2>
Example:

```
python -m src.momentum_graph.crop_scoreboard_tracked path/to/input/video path/to/output/folder
```

<h2>Pipeline</h2>

1. Run `crop_scoreboard_tracked`, `crop_score_lights_tracked` to get the areas of interest for further processing
- Example:
```bash
python -m src.momentum_graph.crop_scoreboard outputs/foil_1_full --demo
```
```bash
python -m src.momentum_graph.crop_score_lights outputs/foil_1_full --demo
```

2. Run `perform_ocr` and `detect_score_lights` on the above outputs respectively to get the raw outputs
- Example
```bash
python -m src.momentum_graph.perform_ocr outputs/foil_1_full --demo
```
```bash
python -m src.momentum_graph.detect_score_lights outputs/foil_1_full --demo
```

- Run `process_scores` and `process_score_lights` on the folder to get the processed outputs
```bash
python -m src.momentum_graph.process_scores outputs/foil_1_full --demo
```
```bash
python -m src.momentum_graph.process_score_lights outputs/foil_1_full --demo
```

- Run `plot_momentum` on the folder to get the momentum_graph
```bash
python -m src.momentum_graph.plot_momentum outputs/foil_1_full
```