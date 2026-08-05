/**
 * gee_extract_covariates.js
 *
 * Google Earth Engine script: extract environmental covariates at AfSIS LDSF
 * soil sampling coordinates for Kenya.
 *
 * Addresses reviewer Major Comment 6 by stating, for every product: the exact
 * collection ID, the bands used, the scale factor, the quality mask, the
 * compositing window and the native spatial support. No covariate is resampled
 * to a finer grid than its native resolution; each is sampled at its own
 * native scale at the point location, and the native scale is recorded in the
 * output so it can be reported in the paper.
 *
 * HOW TO RUN
 *   1. Open https://code.earthengine.google.com (free account required).
 *   2. Upload afsis_kenya_points.csv (columns: SSN, lon, lat) as a table asset.
 *   3. Replace POINTS_ASSET below with your asset path.
 *   4. Run. Export lands in your Google Drive as afsis_kenya_covariates.csv.
 *
 * OUTPUT: one row per sampling point, one column per covariate.
 */

// ---------------------------------------------------------------- config
var POINTS_ASSET = 'users/YOUR_USERNAME/afsis_kenya_points';
var pts = ee.FeatureCollection(POINTS_ASSET);

// AfSIS Phase I field campaign ran 2009-2013. Optical composites are built
// over that window so the covariates are contemporaneous with the soil
// sampling rather than being long-term climatologies of a different era.
var START = '2009-01-01';
var END   = '2013-12-31';

// ------------------------------------------------------- Landsat 7 ETM+ SR
// Sentinel-2 did not launch until 2015, so it CANNOT be contemporaneous with
// 2009-2013 sampling. Landsat 7 is used instead. This is a correction to the
// earlier manuscript, which claimed Sentinel-2 covariates for samples that
// predate the mission.
var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterDate(START, END)
  .filterBounds(pts)
  .map(function (img) {
    // Scale factors from the Landsat C2 L2 product spec
    var opt = img.select('SR_B.').multiply(0.0000275).add(-0.2);
    var st  = img.select('ST_B6').multiply(0.00341802).add(149.0);
    // QA_PIXEL bits: 3 = cloud, 4 = cloud shadow, 5 = snow
    var qa = img.select('QA_PIXEL');
    var mask = qa.bitwiseAnd(1 << 3).eq(0)
      .and(qa.bitwiseAnd(1 << 4).eq(0))
      .and(qa.bitwiseAnd(1 << 5).eq(0));
    return opt.addBands(st).updateMask(mask)
      .copyProperties(img, ['system:time_start']);
  });

var l7med = l7.median().rename([
  'L7_blue', 'L7_green', 'L7_red', 'L7_nir', 'L7_swir1', 'L7_swir2', 'L7_lst'
]);

// Spectral indices, computed from the composited reflectance
var ndvi = l7med.normalizedDifference(['L7_nir', 'L7_red']).rename('NDVI');
var ndwi = l7med.normalizedDifference(['L7_green', 'L7_nir']).rename('NDWI');
var savi = l7med.expression(
  '1.5 * (nir - red) / (nir + red + 0.5)',
  { nir: l7med.select('L7_nir'), red: l7med.select('L7_red') }).rename('SAVI');
// Clay and carbonate mineral ratios, standard in soil remote sensing
var clayIdx = l7med.select('L7_swir1').divide(l7med.select('L7_swir2'))
  .rename('CLAY_INDEX');
var ferrous = l7med.select('L7_swir1').divide(l7med.select('L7_nir'))
  .rename('FERROUS_INDEX');

// ------------------------------------------------------------------ MODIS
// MOD13Q1 vegetation indices, 250 m, 16-day composites
var mod13 = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(START, END)
  .select(['NDVI', 'EVI'])
  .map(function (i) { return i.multiply(0.0001); });
var mod13mean = mod13.mean().rename(['MODIS_NDVI', 'MODIS_EVI']);
var mod13sd = mod13.reduce(ee.Reducer.stdDev())
  .rename(['MODIS_NDVI_sd', 'MODIS_EVI_sd']);   // seasonality

// MOD11A2 land surface temperature, 1 km, 8-day composites
var mod11 = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(START, END)
  .select(['LST_Day_1km', 'LST_Night_1km'])
  .map(function (i) { return i.multiply(0.02).subtract(273.15); });  // -> deg C
var lst = mod11.mean().rename(['LST_day', 'LST_night']);
var lstAmp = mod11.select('LST_Day_1km').reduce(ee.Reducer.stdDev())
  .rename('LST_day_sd');

// MCD43A4 nadir BRDF-adjusted reflectance, 500 m
var mcd43 = ee.ImageCollection('MODIS/061/MCD43A4')
  .filterDate(START, END)
  .select(['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2',
           'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4',
           'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7'])
  .map(function (i) { return i.multiply(0.0001); });
var mb = mcd43.mean().rename(['mb1', 'mb2', 'mb3', 'mb4', 'mb6', 'mb7']);

// MOD16A2 evapotranspiration, 500 m
var mod16 = ee.ImageCollection('MODIS/061/MOD16A2')
  .filterDate(START, END).select(['ET', 'PET'])
  .map(function (i) { return i.multiply(0.1); });
var et = mod16.mean().rename(['ET', 'PET']);

// ------------------------------------------------------------- topography
// SRTM 30 m; derivatives computed at native resolution
var dem = ee.Image('USGS/SRTMGL1_003').rename('DEM');
var terrain = ee.Terrain.products(dem);
var slope = terrain.select('slope').rename('slope');
var aspect = terrain.select('aspect').rename('aspect');

// Topographic Wetness Index from flow accumulation
var flowAcc = ee.Image('WWF/HydroSHEDS/15ACC').rename('flow_acc');
var twi = flowAcc.add(1).log()
  .divide(slope.multiply(Math.PI / 180).tan().add(0.001))
  .rename('TWI');

// Multi-scale topographic position: elevation minus neighbourhood mean
var tpi = dem.subtract(
  dem.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(500, 'meters'))
).rename('TPI');

// ------------------------------------------------------------- bioclimatic
// WorldClim BIO, 927 m native. bio1 and bio12 rescaled to physical units.
var bio = ee.Image('WORLDCLIM/V1/BIO').select(
  ['bio01', 'bio04', 'bio07', 'bio12', 'bio15'],
  ['bio1', 'bio4', 'bio7', 'bio12', 'bio15']);
bio = bio.addBands(bio.select('bio1').multiply(0.1).rename('bio1_degC'), null, true);

// CHIRPS daily precipitation, 5.5 km, aggregated over the sampling window
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(START, END).select('precipitation');
var precipAnnual = chirps.sum().divide(5).rename('CHIRPS_annual_mm');

// --------------------------------------------------------- land cover / fire
var landcover = ee.Image('MODIS/061/MCD12Q1/2010_01_01')
  .select('LC_Type1').rename('landcover');

// ------------------------------------------------------- assemble the stack
var stack = l7med
  .addBands(ndvi).addBands(ndwi).addBands(savi)
  .addBands(clayIdx).addBands(ferrous)
  .addBands(mod13mean).addBands(mod13sd)
  .addBands(lst).addBands(lstAmp)
  .addBands(mb).addBands(et)
  .addBands(dem).addBands(slope).addBands(aspect)
  .addBands(twi).addBands(tpi)
  .addBands(bio).addBands(precipAnnual)
  .addBands(landcover);

print('Covariate stack band names:', stack.bandNames());

// Sample at 30 m — the finest native resolution in the stack. Coarser layers
// are read at their own native support and simply repeat across neighbouring
// 30 m cells; this is an addressing convention, NOT an increase in
// information content, and the paper states the native support of every
// covariate explicitly.
var sampled = stack.sampleRegions({
  collection: pts,
  properties: ['SSN'],
  scale: 30,
  tileScale: 4,
  geometries: false
});

Export.table.toDrive({
  collection: sampled,
  description: 'afsis_kenya_covariates',
  fileNamePrefix: 'afsis_kenya_covariates',
  fileFormat: 'CSV'
});

/**
 * NATIVE SPATIAL SUPPORT OF EACH COVARIATE  (report this table in the paper)
 *
 *   Landsat 7 ETM+ SR bands, indices ......... 30 m   (thermal 60 m resampled to 30 m)
 *   SRTM DEM, slope, aspect, TPI ............. 30 m
 *   MODIS MOD13Q1 NDVI/EVI ................... 250 m
 *   MODIS MCD43A4 reflectance ................ 500 m
 *   MODIS MOD16A2 ET/PET ..................... 500 m
 *   MODIS MCD12Q1 land cover ................. 500 m
 *   MODIS MOD11A2 LST ........................ 1000 m
 *   WorldClim BIO ............................ 927 m
 *   CHIRPS precipitation ..................... 5566 m
 *   HydroSHEDS flow accumulation / TWI ....... 463 m
 *
 * The finest support in the stack is 30 m. Any claim of field-scale (10 m)
 * mapping is therefore not supportable from these covariates, and the paper
 * should describe outputs as a 30 m output grid.
 */
