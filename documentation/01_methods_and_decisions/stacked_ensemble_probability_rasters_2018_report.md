# 2018 Spatial-CV Stacked Ensemble Probability Rasters

These are continuous probability rasters generated from the saved local Spatial-CV Stacked Ensemble models.

| Feature set | Size GB | Min | Max | Mean | Std | Non-finite pixels |
|---|---:|---:|---:|---:|---:|---:|
| Conventional | 0.118 | 0.015895 | 0.998742 | 0.229195 | 0.156678 | 0 |
| AlphaEarth Embeddings | 0.064 | 0.031939 | 0.981962 | 0.415649 | 0.248830 | 0 |
| Conventional + AlphaEarth Embeddings | 0.103 | 0.020377 | 0.995051 | 0.219272 | 0.158347 | 0 |

## Output Folder

`D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble`

## QA Tables

- `D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble_qa\stacked_ensemble_probability_rasters_2018_qa.csv`
- `D:\DING PROJECT\04_maps\rasters_2018_probability_stacked_ensemble_qa\stacked_ensemble_probability_rasters_2018_qa.json`

## Notes

- No susceptibility classes are created.
- Rasters are 250 m probability outputs from stacked models.
- A 30 m cartographic display resample can be produced from these rasters, but it should be described as display resampling rather than true 30 m prediction.
