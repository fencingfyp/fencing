<h1>Instructions</h1>

<h2>How to run the scripts</h2>
Example:

```
python -m src.momentum_graph.crop_scoreboard_tracked path/to/input/video path/to/output/folder
```

<h2>Pipeline</h2>
1. Crop areas of interest

- Run `crop_scoreboard_tracked`, `crop_score_lights_tracked` to get the areas of interest for further processing

- Run `perform_ocr` and `perform_score_light_detection` on the above outputs respectively to get the raw outputs

- Run `process_scores` and `process_score_lights` on the folder to get the processed outputs

- Run `run_analysis` on the folder to get the predictions