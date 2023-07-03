import Map from 'ol/Map';
import View from 'ol/View';
import Group from 'ol/layer/Group.js';
import TileLayer from 'ol/layer/Tile';
import WMTS from 'ol/source/WMTS.js';
import WMTSTileGrid from 'ol/tilegrid/WMTS.js';
import GeoJSON from 'ol/format/GeoJSON';
import {get as getProjection, transform} from 'ol/proj.js';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import {getTopLeft, getWidth} from 'ol/extent.js';
import {Style, Text, Fill, Stroke, Icon} from 'ol/style.js';
import {Attribution, defaults as defaultControls} from 'ol/control.js';
import 'ol-layerswitcher/dist/ol-layerswitcher.css';
import LayerSwitcher from 'ol-layerswitcher';

window.show_story = function() {
  document.getElementById('story').classList.add("active");
}

window.hide_story = function() {
  document.getElementById('story').classList.remove("active");
}


const projection = getProjection('EPSG:3857');
console.log(projection.getExtent())
const size = getWidth(projection.getExtent()) / 256;
const resolutions = new Array(19);
const matrixIds = new Array(19);
for (let z = 0; z < 19; ++z) {
  // generate resolutions and matrixIds arrays for this WMTS
  resolutions[z] = size / Math.pow(2, z);
  matrixIds[z] = z;
}

const s2_source = new WMTS({
  attributions: `Base Layer: <a class="a-light" xmlns:dct="http://purl.org/dc/terms/" href="https://s2maps.eu" property="dct:title">Sentinel-2 cloudless - https://s2maps.eu</a> by <a class="a-light" xmlns:cc="http://creativecommons.org/ns#" href="https://eox.at" property="cc:attributionName" rel="cc:attributionURL">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2021)`,
  url: 'https://s2maps-tiles.eu/wmts/',
  maxZoom: 18,
  layer: 's2cloudless-2020_3857',
  matrixSet: 'GoogleMapsCompatible',
  projection: projection,
  format: 'image/jpeg',
  style: 'default',
  wrapX: true,
  tileGrid: new WMTSTileGrid({
    origin: getTopLeft(projection.getExtent()),
    resolutions: resolutions,
    matrixIds: matrixIds,
  })
});

const extract_date = function(feature) {
  let src = feature.get('src');
  let match1 = /((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})/
  let match2 = /((?:19|20)\d{2})(\d{2})(\d{2})/
  let matches = (src.match(match1) || []).concat(src.match(match2) || []);
  console.log(matches);
  return matches[1] + '-' + matches[2];
}

// Create the basemap layer
const s2_layer = new TileLayer({
  source: s2_source,
});

const calfin_style = feature => new Style({
  stroke: new Stroke({color: 'red', width: 3}),
  text: new Text({
    // textAlign: align == '' ? undefined : align,
    textBaseline: 'middle',
    font: 'bold 11pt sans-serif',
    text: extract_date(feature),
    fill: new Fill({color: 'red'}),
    stroke: new Stroke({color: 'white', width: 4}),
    placement: 'line',
  })
});
const calfin_layer = new VectorLayer({
  source: new VectorSource({
    url: 'geojson/CALFIN.geo.json',
    format: new GeoJSON({}),
  }),
  style: calfin_style,
  title: '<span style="color:red;font-weight:bold;">—</span> Calfin Predictions'
});

const tud_style = feature => new Style({
  stroke: new Stroke({color: 'blue', width: 3}),
  text: new Text({
    // textAlign: align == '' ? undefined : align,
    textBaseline: 'middle',
    font: 'bold 11pt sans-serif',
    text: extract_date(feature),
    fill: new Fill({color: 'blue'}),
    stroke: new Stroke({color: 'white', width: 4}),
    placement: 'line',
  })
});
const tud_layer = new VectorLayer({
  source: new VectorSource({
    url: 'geojson/TUD.geo.json',
    format: new GeoJSON({}),
  }),
  style: tud_style,
  title: '<span style="color:blue;font-weight:bold;">—</span> TUD Predictions'
});

const baumhoer_style = feature => new Style({
  stroke: new Stroke({color: 'green', width: 3}),
  text: new Text({
    // textAlign: align == '' ? undefined : align,
    textBaseline: 'middle',
    font: 'bold 11pt sans-serif',
    text: extract_date(feature),
    fill: new Fill({color: 'green'}),
    stroke: new Stroke({color: 'white', width: 4}),
    placement: 'line',
  })
});
const baumhoer_layer = new VectorLayer({
  source: new VectorSource({
    url: 'geojson/Baumhoer.geo.json',
    format: new GeoJSON({}),
  }),
  style: baumhoer_style,
  title: '<span style="color:green;font-weight:bold;">&mdash;</span> Baumhoer Predictions'
});

const attribution = new Attribution({collapsible: false});
const layer_switcher = new LayerSwitcher({
  startActive: true,
  reverse: false,
  activationMode: 'click',
})

// Create the map
const map = new Map({
  target: 'map',
  layers: [
    s2_layer,
    calfin_layer,
    tud_layer,
    baumhoer_layer
  ],
  controls: defaultControls({attribution: false}).extend([attribution, layer_switcher]),
  view: new View({
    center: transform([-45,75], 'EPSG:4326', projection),
    zoom: 5
  })
});
