// Build the first clean 2018 sampling table in Earth Engine.
//
// Run after this asset exists:
// projects/ee-mohibullah141300/assets/cpec_lsm_clean/inventory/cpec_inventory_model_candidates_v1

var studyArea = ee.FeatureCollection(
  'projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area'
);

var positives = ee.FeatureCollection(
  'projects/ee-mohibullah141300/assets/cpec_lsm_clean/inventory/cpec_inventory_model_candidates_v1'
).filter(ee.Filter.eq('use_role', 'train_candidate'));

var factors2018 = ee.Image(
  'projects/ee-mohibullah141300/assets/cpec_lsm_clean/public/cpec_public_factors_2018_alphaearth_250m'
);

var seed = 141300;
var scale = 250;
var positiveBufferMeters = 500;

// Valid terrain mask. This keeps negative samples inside the factor domain.
var validMask = factors2018.select('slope_deg').mask();

var positiveMask = ee.Image(0).byte()
  .paint(positives.map(function(f) {
    return f.buffer(positiveBufferMeters);
  }), 1)
  .selfMask();

var negativeMask = validMask.updateMask(positiveMask.unmask(0).not()).clip(studyArea);

// Terrain stratification for negatives: slope and elevation bins.
var slope = factors2018.select('slope_deg');
var elevation = factors2018.select('elevation_m');
var slopeClass = slope.expression(
  "b('slope_deg') < 5 ? 1" +
  ": b('slope_deg') < 15 ? 2" +
  ": b('slope_deg') < 30 ? 3" +
  ": b('slope_deg') < 45 ? 4" +
  ": 5"
).rename('slope_class');
var elevClass = elevation.expression(
  "b('elevation_m') < 500 ? 1" +
  ": b('elevation_m') < 1500 ? 2" +
  ": b('elevation_m') < 3000 ? 3" +
  ": b('elevation_m') < 4500 ? 4" +
  ": 5"
).rename('elev_class');
var strata = slopeClass.multiply(10).add(elevClass).rename('strata').updateMask(negativeMask);

var positiveCount = positives.size();
print('Positive train candidates', positiveCount);

var classValues = ee.List.sequence(11, 55);
var negativePoints = strata.stratifiedSample({
  numPoints: positiveCount.divide(classValues.length()).ceil(),
  classBand: 'strata',
  region: studyArea.geometry(),
  scale: scale,
  seed: seed,
  geometries: true,
  dropNulls: true,
  tileScale: 4
}).map(function(f) {
  return f.set({
    label: 0,
    use_role: 'negative_train_candidate',
    source: 'GEE_stratified_negative_sampling',
    seed: seed,
    positive_buffer_m: positiveBufferMeters
  });
});

var positiveSamples = positives.map(function(f) {
  return f.set('label', 1);
});

var samples = positiveSamples.merge(negativePoints);

// Spatial block fold ID: 1 degree blocks, deterministic modulo 5.
samples = samples.map(function(f) {
  var xy = f.geometry().coordinates();
  var lonBlock = ee.Number(xy.get(0)).floor();
  var latBlock = ee.Number(xy.get(1)).floor();
  var blockId = lonBlock.format('%d').cat('_').cat(latBlock.format('%d'));
  var fold = lonBlock.multiply(31).add(latBlock.multiply(17)).abs().mod(5).add(1);
  return f.set({
    spatial_block_1deg: blockId,
    spatial_fold_5: fold
  });
});

var sampled = factors2018.sampleRegions({
  collection: samples,
  properties: [
    'label',
    'inventory_id',
    'source',
    'hazard_type',
    'event_year',
    'confidence',
    'use_role',
    'spatial_block_1deg',
    'spatial_fold_5'
  ],
  scale: scale,
  geometries: true,
  tileScale: 4
});

print('Negative points', negativePoints.size());
print('All labelled samples before factor extraction', samples.size());
print('Sampled table preview', sampled.limit(5));

// Enable after checking fold balance in the Console.
// Export.table.toAsset({
//   collection: sampled,
//   description: 'cpec_2018_lsm_samples_v1',
//   assetId: 'projects/ee-mohibullah141300/assets/cpec_lsm_clean/samples/cpec_2018_lsm_samples_v1'
// });

// Export.table.toDrive({
//   collection: sampled,
//   description: 'cpec_2018_lsm_samples_v1',
//   folder: 'GEE_CPEC_LSM',
//   fileNamePrefix: 'cpec_2018_lsm_samples_v1',
//   fileFormat: 'CSV'
// });
