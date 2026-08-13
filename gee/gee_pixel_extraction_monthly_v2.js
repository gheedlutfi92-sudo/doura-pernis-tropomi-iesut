// ============================================================
// GEE Monthly Pixel Extraction v2 — RF-LUR Monthly Analysis
// Doura Refinery (Baghdad) & Shell Pernis (Rotterdam)
//
// v2 additions over v1:
//   - TROPOMI SO2 (monthly composite)
//   - Landsat 8+9 LST (static summer mean 2018-2024)
//   - SO2/NO2 ratio computed in Python after export (safer)
//
// Exports one row per pixel per month (~20,000 rows per site)
// New CSV columns: so2, lst_celsius
// Run in: code.earthengine.google.com
// ============================================================

var doura  = ee.Geometry.Point([44.40, 33.25]);
var pernis = ee.Geometry.Point([4.30,  51.88]);
var doura_buf  = doura.buffer(50000);
var pernis_buf = pernis.buffer(50000);

// ── TROPOMI cloud mask ────────────────────────────────────────
function applyCloudMask(img) {
  var cf = img.select('cloud_fraction');
  return img.updateMask(cf.lte(0.5));
}

// ── Landsat QA_PIXEL cloud+shadow mask ───────────────────────
function applyLandsatMask(img) {
  var qa    = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 3).eq(0)
               .and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(clear);
}

// ── Static predictors ────────────────────────────────────────
var worldpop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('year', 2020)).mosaic().rename('population');

var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface').rename('built_up');

var elevation = ee.Image('USGS/SRTMGL1_003')
  .select('elevation').rename('elevation');

// Static Landsat LST — summer mean 2018-2024, Landsat 8+9, June-August
var lst_static = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterDate('2018-01-01', '2025-01-01')
  .filter(ee.Filter.calendarRange(6, 8, 'month'))
  .map(applyLandsatMask)
  .select('ST_B10')
  .mean()
  .multiply(0.00341802).add(149.0).subtract(273.15)
  .rename('lst_celsius');

function makeDist(ref_lon, ref_lat) {
  return ee.Image.pixelLonLat()
    .expression(
      'sqrt(pow((lon-ref_lon)*111320*cos(lat*3.14159/180),2)' +
      '    +pow((lat-ref_lat)*110540,2))',
      { lon: ee.Image.pixelLonLat().select('longitude'),
        lat: ee.Image.pixelLonLat().select('latitude'),
        ref_lon: ref_lon, ref_lat: ref_lat }
    ).rename('dist_from_refinery');
}
var dist_doura  = makeDist(44.40, 33.25);
var dist_pernis = makeDist(4.30,  51.88);

// ── Month list (pandemic excluded) ───────────────────────────
function makeMonthMillis(startStr, endStr) {
  var start   = ee.Date(startStr);
  var nMonths = ee.Date(endStr).difference(start, 'month').round();
  return ee.List.sequence(0, nMonths.subtract(1)).map(function(i) {
    return start.advance(ee.Number(i), 'month').millis();
  });
}
var period1   = makeMonthMillis('2018-05-01', '2020-01-01');
var period2   = makeMonthMillis('2022-01-01', '2025-12-01');
var allMillis = period1.cat(period2);
print('Total months:', allMillis.size());

// ── sampleMonth ───────────────────────────────────────────────
// Masked fallback pattern: merge a fully-masked constant image into every
// collection before .mean() so empty months never return a 0-band image.
// A fully-masked constant contributes nothing to the mean but ensures
// the result always has the correct number of bands.

function sampleMonth(millis, region, dist_ref, site_label) {
  var start = ee.Date(millis);
  var end   = start.advance(1, 'month');
  var year  = start.get('year');
  var month = start.get('month');

  // NO2 ─────────────────────────────────────────────────────
  var no2_fb = ee.Image.constant(0)
    .rename('tropospheric_NO2_column_number_density')
    .updateMask(ee.Image(0));
  var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
    .select(['tropospheric_NO2_column_number_density', 'cloud_fraction'])
    .filterBounds(region).filterDate(start, end)
    .map(applyCloudMask)
    .select('tropospheric_NO2_column_number_density')
    .merge(ee.ImageCollection([no2_fb]))
    .mean().rename('no2').unmask(0)
    .reproject({crs: 'EPSG:4326', scale: 5500});

  // SO2 ─────────────────────────────────────────────────────
  var so2_fb = ee.Image.constant(0)
    .rename('SO2_column_number_density')
    .updateMask(ee.Image(0));
  var so2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2')
    .select(['SO2_column_number_density', 'cloud_fraction'])
    .filterBounds(region).filterDate(start, end)
    .map(applyCloudMask)
    .select('SO2_column_number_density')
    .merge(ee.ImageCollection([so2_fb]))
    .mean().rename('so2').unmask(0)
    .reproject({crs: 'EPSG:4326', scale: 5500});

  // NOTE: SO2/NO2 ratio is computed in Python after export:
  //   df['so2_no2_ratio'] = df['so2'] / df['no2'].abs().clip(lower=1e-10)

  // ERA5 ────────────────────────────────────────────────────
  var era5_fb = ee.Image.constant([0, 0, 273.15])
    .rename(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'temperature_2m'])
    .updateMask(ee.Image(0));
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterDate(start, end)
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'temperature_2m'])
    .merge(ee.ImageCollection([era5_fb]))
    .mean();

  var wind_speed  = era5.select('u_component_of_wind_10m').pow(2)
    .add(era5.select('v_component_of_wind_10m').pow(2))
    .sqrt().rename('wind_speed');
  var temperature = era5.select('temperature_2m')
    .subtract(273.15).rename('temperature_c');

  // VIIRS ───────────────────────────────────────────────────
  var viirs_fb = ee.Image.constant(0).rename('avg_rad').updateMask(ee.Image(0));
  // VCMCFG = includes gas flare radiance (not fire-masked)
  // VCMSLCFG = fire-masked (removes flare signal — wrong for this study)
  var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
    .filterDate(start, end).select('avg_rad')
    .merge(ee.ImageCollection([viirs_fb]))
    .mean().rename('viirs_ntl');

  // Stack ───────────────────────────────────────────────────
  var stack = no2
    .addBands(so2)
    .addBands(wind_speed)
    .addBands(temperature)
    .addBands(viirs)
    .addBands(worldpop)
    .addBands(ghsl)
    .addBands(elevation)
    .addBands(lst_static)
    .addBands(dist_ref)
    .clip(region);

  var samples = stack.sample({
    region: region, scale: 5500,
    seed: 42, geometries: true, dropNulls: true
  });

  return samples.map(function(f) {
    return f.set('site',  site_label)
             .set('year',  year)
             .set('month', month)
             .set('lon', f.geometry().coordinates().get(0))
             .set('lat', f.geometry().coordinates().get(1));
  });
}

// ── Map over all months ───────────────────────────────────────
var doura_all = ee.FeatureCollection(
  allMillis.map(function(m) {
    return sampleMonth(m, doura_buf, dist_doura, 'doura');
  })
).flatten();

var pernis_all = ee.FeatureCollection(
  allMillis.map(function(m) {
    return sampleMonth(m, pernis_buf, dist_pernis, 'pernis');
  })
).flatten();

// ── Simple diagnostics (avoid triggering full evaluation) ─────
print('Script loaded. Go to Tasks panel and run both exports.');
print('Expected: ~67 months x ~250 pixels = ~16,000-20,000 rows per site.');

// ── Exports ───────────────────────────────────────────────────
Export.table.toDrive({
  collection: doura_all,
  description: 'doura_monthly_samples_v2',
  folder: 'GEE_Exports',
  fileNamePrefix: 'doura_monthly_samples_v2',
  fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'site', 'year', 'month',
              'no2', 'so2',
              'viirs_ntl', 'population', 'built_up', 'elevation', 'lst_celsius',
              'wind_speed', 'temperature_c', 'dist_from_refinery']
});

Export.table.toDrive({
  collection: pernis_all,
  description: 'pernis_monthly_samples_v2',
  folder: 'GEE_Exports',
  fileNamePrefix: 'pernis_monthly_samples_v2',
  fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'site', 'year', 'month',
              'no2', 'so2',
              'viirs_ntl', 'population', 'built_up', 'elevation', 'lst_celsius',
              'wind_speed', 'temperature_c', 'dist_from_refinery']
});
