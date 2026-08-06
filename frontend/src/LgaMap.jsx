// Catchment map: the three council areas (LGA polygons) plus the modelled
// station. The LGA that contains Murray Bridge is tinted with the current risk
// band; the others stay neutral, because only Murray Bridge has a trained model.

import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { BAND_META, MODELLED } from "./domain";

// Frame the map on the council areas once they load, so the three LGAs fill the
// card instead of sitting in a wide, mostly empty view of the state.
function FitToLga({ data }) {
  const map = useMap();
  const done = useRef(false);
  useEffect(() => {
    if (!data || done.current) return;
    const bounds = L.geoJSON(data).getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18] });
    done.current = true;
  }, [data, map]);
  return null;
}

// Murray Bridge sits in the Rural City of Murray Bridge LGA; that is the only
// council area we colour by prediction.
const MODELLED_LGA = "Murray Bridge";

export default function LgaMap({ band }) {
  const [lga, setLga] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/lga_boundaries.geojson`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setLga)
      .catch(() => setFailed(true));
  }, []);

  const bandHex = BAND_META[band].hex;

  const styleFor = (feature) => {
    const isModelled = feature?.properties?.name === MODELLED_LGA;
    return {
      color: isModelled ? bandHex : "#8fa3bf",
      weight: isModelled ? 2.5 : 1.2,
      fillColor: isModelled ? bandHex : "#c9d6e6",
      fillOpacity: isModelled ? 0.28 : 0.12,
    };
  };

  const onEachFeature = (feature, layer) => {
    const name = feature?.properties?.name ?? "LGA";
    layer.bindTooltip(
      name === MODELLED_LGA
        ? `${name} — modelled (${band} risk)`
        : `${name} — context only, no trained model`,
      { sticky: true },
    );
  };

  return (
    <div className="map-wrap">
      <MapContainer center={[-34.9, 139.4]} zoom={8} scrollWheelZoom={false} className="map">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        {lga && (
          <>
            <GeoJSON
              key={band /* restyle when the band changes */}
              data={lga}
              style={styleFor}
              onEachFeature={onEachFeature}
            />
            <FitToLga data={lga} />
          </>
        )}
        <CircleMarker
          center={[MODELLED.lat, MODELLED.lon]}
          radius={11}
          pathOptions={{ color: "#ffffff", weight: 2, fillColor: bandHex, fillOpacity: 0.95 }}
        >
          <Tooltip>{`${MODELLED.name} (modelled): ${band} risk`}</Tooltip>
        </CircleMarker>
      </MapContainer>

      <div className="map-foot">
        {failed
          ? "LGA boundaries unavailable"
          : "3 council areas · modelled station: Murray Bridge · sources: BoM, DEW, Water Data SA"}
      </div>
    </div>
  );
}
