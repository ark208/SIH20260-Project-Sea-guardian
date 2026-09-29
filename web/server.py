# Project Sea Guardian - Web Dashboard & Multi-Region API Server
import http.server
import json
import urllib.parse
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.pipeline import SeaGuardianPipeline
from core.attribution.llm_intelligence import GroqMaritimeIntelligence

pipeline = SeaGuardianPipeline()
ai_intel = GroqMaritimeIntelligence()
latest_pipeline_cache = {}

class SeaGuardianRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global latest_pipeline_cache
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
            with open(template_path, "rb") as f:
                self.wfile.write(f.read())
            return
            
        elif parsed.path.startswith("/static/"):
            file_rel = parsed.path.replace("/static/", "", 1)
            file_path = os.path.join(os.path.dirname(__file__), "static", file_rel)
            if os.path.exists(file_path):
                self.send_response(200)
                if file_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                elif file_path.endswith(".css"):
                    self.send_header("Content-Type", "text/css")
                elif file_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript")
                self.end_headers()
                with open(file_path, "rb") as sf:
                    self.wfile.write(sf.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        elif parsed.path == "/api/incidents":
            incidents = pipeline.list_incidents()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"incidents": incidents}).encode("utf-8"))
            return

        elif parsed.path == "/api/run-pipeline":
            incident_key = params.get("incident_key", ["INC-ARABIAN-MUMBAI"])[0]
            wind = float(params.get("wind", [5.5])[0]) if "wind" in params else None
            volume = float(params.get("volume", [30.0])[0]) if "volume" in params else None
            stokes_str = params.get("stokes", ["true"])[0].lower()
            enable_stokes = (stokes_str == "true")
            
            result = pipeline.run_end_to_end(
                incident_key=incident_key,
                wind_speed_ms=wind,
                enable_stokes=enable_stokes,
                estimated_volume_m3=volume
            )
            # Frontend compatibility layer
            slick = result.get("sar_scene", {}).get("detected_slick", {})
            fay = result.get("fay_spreading", {})
            env = result.get("origin_envelope", {})
            suspects = result.get("ranked_suspects", [])
            fleet = result.get("fleet", [])
            inc_info = result.get("incident_info", {})
            
            vessels_adapted = []
            for v in fleet:
                curr = v.get("track", [{}])[-1] if v.get("track") else {}
                vessels_adapted.append({
                    "name": v.get("name"),
                    "mmsi": v.get("mmsi"),
                    "flag": v.get("flag_state"),
                    "type": v.get("vessel_type"),
                    "current_lat": curr.get("lat", 0),
                    "current_lon": curr.get("lon", 0),
                    "speed_knots": curr.get("speed_knots", 0),
                    "course": curr.get("heading_deg", 0),
                    "ais_dark_hours": v.get("dark_duration_hours", 0),
                    "track_history": [{"lat": t["lat"], "lon": t["lon"]} for t in v.get("track", [])]
                })

            suspects_adapted = []
            for s in suspects:
                dark_hrs = 0
                if isinstance(s.get("dark_details"), dict):
                    dark_hrs = s.get("dark_details", {}).get("duration_hours", 0)
                elif s.get("has_dark_gap"):
                    dark_hrs = 2.5
                suspects_adapted.append({
                    "name": s.get("name"),
                    "mmsi": str(s.get("mmsi")),
                    "flag": s.get("flag_state"),
                    "type": s.get("vessel_type"),
                    "composite_score": s.get("dampened_composite", s.get("raw_composite", 0)),
                    "ais_dark_hours": dark_hrs,
                    "factors": {
                        "spatial_proximity": s.get("s_spatial", 0),
                        "ais_dark_gap": s.get("s_dark", 0),
                        "kinematic_anomaly": s.get("s_behavior", 0),
                        "vessel_risk": s.get("s_risk", 0)
                    },
                    "evidence_summary": f"Vessel ranked #{s.get('rank', 1)} with {s.get('culprit_percentage', 0)}% culprit probability."
                })

            poly_coords = [[p[1], p[0]] for p in slick.get("polygon_geo", [])]

            result["incident_meta"] = {
                "name": inc_info.get("name", "Active Hotspot"),
                "region": inc_info.get("region", "Indian EEZ"),
                "lat": inc_info.get("center_coords", [19.25, 71.40])[0],
                "lon": inc_info.get("center_coords", [19.25, 71.40])[1],
            }
            result["sar_detection"] = {
                "center_lat": slick.get("centroid_lat", 19.25),
                "center_lon": slick.get("centroid_lon", 71.40),
                "slick_area_sqkm": slick.get("area_km2", 0),
                "polygon_coords": poly_coords
            }
            result["hydrodynamics"] = {
                "estimated_volume_bbl": fay.get("estimated_volume_m3", 0) * 6.2898,
                "spill_age_hours": fay.get("spill_age_hours", 0),
                "reverse_trajectory": [{"lat": snap["mean_lat"], "lon": snap["mean_lon"]} for snap in env.get("snapshots", [])]
            }
            result["vessels"] = vessels_adapted
            result["suspect_rankings"] = suspects_adapted
            result["primary_suspect"] = suspects_adapted[0] if suspects_adapted else None

            latest_pipeline_cache[incident_key] = result
            latest_pipeline_cache["current"] = result
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            def default_serializer(obj):
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                elif hasattr(obj, "item"):
                    return obj.item()
                raise TypeError(f"Type {type(obj)} not serializable")
                
            json_bytes = json.dumps(result, default=default_serializer).encode("utf-8")
            self.wfile.write(json_bytes)
            return
            
        elif parsed.path == "/api/ai-intelligence-brief":
            incident_key = params.get("incident_key", ["current"])[0]
            curr_res = latest_pipeline_cache.get(incident_key, latest_pipeline_cache.get("current"))
            if not curr_res:
                curr_res = pipeline.run_end_to_end()
                latest_pipeline_cache["current"] = curr_res
                
            dossier = curr_res.get("ntro_dossier", {})
            brief = ai_intel.generate_executive_intelligence_brief(dossier)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"brief": brief, "briefing": brief}).encode("utf-8"))
            return

        elif parsed.path == "/api/incident-report":
            incident_key = params.get("incident_key", ["current"])[0]
            curr_res = latest_pipeline_cache.get(incident_key, latest_pipeline_cache.get("current"))
            if not curr_res:
                curr_res = pipeline.run_end_to_end()
                latest_pipeline_cache["current"] = curr_res
                
            dossier = curr_res.get("ntro_dossier", {})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(dossier, indent=2).encode("utf-8"))
            return
            
        else:
            super().do_GET()

    def do_POST(self):
        global latest_pipeline_cache
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/chat-copilot":
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length).decode('utf-8')
            try:
                body_data = json.loads(post_body)
                query = body_data.get("query") or body_data.get("message") or ""
                incident_key = body_data.get("incident_key", "current")
            except:
                query = post_body
                incident_key = "current"
                
            curr_res = latest_pipeline_cache.get(incident_key, latest_pipeline_cache.get("current"))
            if not curr_res:
                curr_res = pipeline.run_end_to_end()
                latest_pipeline_cache["current"] = curr_res
                
            answer = ai_intel.answer_analyst_query(query, curr_res)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"answer": answer, "response": answer}).encode("utf-8"))
            return
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = http.server.ThreadingHTTPServer(server_address, SeaGuardianRequestHandler)
    print(f"Sea Guardian Multi-Region Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
