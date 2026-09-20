// Export continuous 2018 CPEC landslide susceptibility probability maps.
// No very-low/low/moderate/high/very-high classes are created.

var studyArea = ee.FeatureCollection(
  'projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area'
);
var samples = ee.FeatureCollection(
  'projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v2'
);
var factors2018 = ee.Image(
  'projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m'
);

var conventionalReduced = [
  'elevation_m', 'slope_deg', 'aspect_deg',
  'rain_monsoon_total', 'rain_max_1day',
  'ndvi_median', 'ndvi_amplitude',
  'lst_day_mean_c', 'modis_lc_type1'
];
var alpha = ee.List.sequence(0, 63).map(function(i) {
  return ee.String('A').cat(ee.Number(i).format('%02d'));
}).getInfo();
var fusedReduced = conventionalReduced.concat(alpha);

function probabilityImage(name, bands) {
  var classifier = ee.Classifier.smileGradientTreeBoost({
    numberOfTrees: 500,
    shrinkage: 0.03,
    samplingRate: 0.85,
    maxNodes: 32,
    loss: 'Logistic',
    seed: 141300
  }).setOutputMode('PROBABILITY').train({
    features: samples,
    classProperty: 'label',
    inputProperties: bands
  });
  return factors2018.select(bands).classify(classifier)
    .rename('prob_' + name)
    .toFloat()
    .clip(studyArea)
    .set({
      year: 2018,
      feature_set: name,
      output_type: 'continuous_probability_no_classes'
    });
}

var probConventional = probabilityImage('conventional_reduced', conventionalReduced);
var probAlpha = probabilityImage('alphaearth_only', alpha);
var probFused = probabilityImage('fused_reduced', fusedReduced);
var diffFusedConventional = probFused.subtract(probConventional)
  .rename('diff_fused_minus_conventional_reduced')
  .toFloat();
var diffFusedAlpha = probFused.subtract(probAlpha)
  .rename('diff_fused_minus_alphaearth_only')
  .toFloat();

Map.centerObject(studyArea, 5);
Map.addLayer(probFused, {min: 0, max: 1, palette: ['#16365c', '#7fbf7b', '#ffffbf', '#fdae61', '#a50026']}, 'Fused probability');
Map.addLayer(diffFusedConventional, {min: -0.5, max: 0.5, palette: ['#313695', '#ffffbf', '#a50026']}, 'Fused - conventional');

// Exports are handled by the Python script for reproducibility:
// D:\DING PROJECT\06_scripts\python\export_2018_probability_maps_gee.py
