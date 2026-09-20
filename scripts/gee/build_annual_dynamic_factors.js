// Clean CPEC annual dynamic factor builder.
//
// Master boundary:
//   projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area
//
// This script is a foundation script. It builds year-specific factor images
// without relying on previous model outputs. It is designed to run inside the
// Earth Engine Code Editor or be ported to the Python API.

var studyArea = ee.FeatureCollection(
  'projects/ee-mohibullah141300/assets/cpec_boundary_official_study_area'
);

// CPEC-wide maps should start at a moderate resolution. For KKH/infrastructure
// corridor sub-areas, use a separate high-resolution script.
var EXPORT_SCALE = 250;
var EXPORT_CRS = 'EPSG:4326';

function renameOne(image, oldName, newName) {
  return image.select([oldName]).rename([newName]);
}

function annualRainfallFactors(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(start, end)
    .filterBounds(studyArea)
    .select('precipitation');

  var annualTotal = chirps.sum().rename('rain_annual_total');
  var monsoonTotal = chirps
    .filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 10, 1))
    .sum()
    .rename('rain_monsoon_total');
  var max1Day = chirps.max().rename('rain_max_1day');

  var days = ee.List.sequence(0, end.difference(start, 'day').subtract(3));
  var rolling3 = ee.ImageCollection.fromImages(days.map(function(dayOffset) {
    var d = start.advance(ee.Number(dayOffset), 'day');
    return chirps.filterDate(d, d.advance(3, 'day')).sum()
      .set('system:time_start', d.millis());
  }));
  var max3Day = rolling3.max().rename('rain_max_3day');

  var days7 = ee.List.sequence(0, end.difference(start, 'day').subtract(7));
  var rolling7 = ee.ImageCollection.fromImages(days7.map(function(dayOffset) {
    var d = start.advance(ee.Number(dayOffset), 'day');
    return chirps.filterDate(d, d.advance(7, 'day')).sum()
      .set('system:time_start', d.millis());
  }));
  var max7Day = rolling7.max().rename('rain_max_7day');

  return ee.Image.cat([annualTotal, monsoonTotal, max1Day, max3Day, max7Day]);
}

function annualVegetationFactors(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
    .filterDate(start, end)
    .filterBounds(studyArea);

  var ndvi = modis.select('NDVI').map(function(img) {
    return img.multiply(0.0001).copyProperties(img, ['system:time_start']);
  });
  var evi = modis.select('EVI').map(function(img) {
    return img.multiply(0.0001).copyProperties(img, ['system:time_start']);
  });

  return ee.Image.cat([
    ndvi.median().rename('ndvi_median'),
    ndvi.max().rename('ndvi_max'),
    ndvi.max().subtract(ndvi.min()).rename('ndvi_amplitude'),
    evi.median().rename('evi_median')
  ]);
}

function annualThermalFactors(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var lst = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterDate(start, end)
    .filterBounds(studyArea)
    .select('LST_Day_1km')
    .map(function(img) {
      // MOD11A2 scale factor is 0.02 K. Convert to Celsius.
      return img.multiply(0.02).subtract(273.15)
        .copyProperties(img, ['system:time_start']);
    });

  return ee.Image.cat([
    lst.mean().rename('lst_day_mean_c'),
    lst.max().rename('lst_day_max_c')
  ]);
}

function annualLandCoverFactors(year) {
  var lc = ee.ImageCollection('MODIS/061/MCD12Q1')
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .first();
  return ee.Image(lc).select('LC_Type1').rename('modis_lc_type1');
}

function terrainFactors() {
  var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
    .filterBounds(studyArea)
    .select('DEM')
    .mosaic()
    .clip(studyArea);
  var terrain = ee.Terrain.products(dem);
  return ee.Image.cat([
    dem.rename('elevation_m'),
    terrain.select('slope').rename('slope_deg'),
    terrain.select('aspect').rename('aspect_deg')
  ]);
}

function alphaEarthFactors(year) {
  // Satellite Embedding/AlphaEarth is available from 2017 onward.
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  var embedding = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(start, end)
    .filterBounds(studyArea)
    .mosaic()
    .select('A.*');
  return embedding;
}

function buildAnnualFactorStack(year, includeAlphaEarth) {
  year = ee.Number(year);
  var y = year.getInfo();

  var stack = terrainFactors()
    .addBands(annualRainfallFactors(y))
    .addBands(annualVegetationFactors(y))
    .addBands(annualThermalFactors(y))
    .addBands(annualLandCoverFactors(y));

  if (includeAlphaEarth && y >= 2017) {
    stack = stack.addBands(alphaEarthFactors(y));
  }

  return stack
    .clip(studyArea)
    .toFloat()
    .set('year', y)
    .set('includes_alphaearth', includeAlphaEarth && y >= 2017);
}

// Change this year for inspection/export.
var year = 2018;
var includeAlphaEarth = true;
var annualStack = buildAnnualFactorStack(year, includeAlphaEarth);

print('Study area', studyArea);
print('Annual factor stack year', year);
print('Band names', annualStack.bandNames());
print('Band count', annualStack.bandNames().size());

Map.centerObject(studyArea, 5);
Map.addLayer(studyArea, {}, 'Study area');
Map.addLayer(annualStack.select('rain_monsoon_total'), {min: 0, max: 1200}, 'Monsoon rainfall');
Map.addLayer(annualStack.select('ndvi_median'), {min: 0, max: 0.8}, 'NDVI median');
Map.addLayer(annualStack.select('slope_deg'), {min: 0, max: 60}, 'Slope');

// Example managed export. Enable only when needed.
// Export.image.toDrive({
//   image: annualStack,
//   description: 'cpec_dynamic_factors_' + year + (includeAlphaEarth ? '_alphaearth' : '_conventional'),
//   folder: 'GEE_CPEC_LSM',
//   fileNamePrefix: 'cpec_dynamic_factors_' + year + (includeAlphaEarth ? '_alphaearth' : '_conventional'),
//   region: studyArea.geometry(),
//   scale: EXPORT_SCALE,
//   crs: EXPORT_CRS,
//   maxPixels: 1e13
// });
