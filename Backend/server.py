from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from pathlib import Path

# Import our benchmarking logic
import run_from_configs

# Directories
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"  # renamed folder

app = Flask(__name__)

# Config file stored in Backend
CONFIG_FILE = BACKEND_DIR / "configs.jsonl"  # each line is one JSON object


@app.route("/", methods=["GET"])
def index():
    """
    Serve the UI from the frontend folder so the page and API share the same origin.
    """
    return send_from_directory(FRONTEND_DIR, "ui.html")


@app.route("/save-config", methods=["POST"])
def save_config():
    """
    Receive JSON from the frontend, append it to configs.jsonl with a timestamp,
    then automatically run the benchmarks and generate plots.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": data,
    }

    with CONFIG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Run benchmarks based on the latest config
    try:
        df = run_from_configs.run_benchmarks(
            run_from_configs.load_latest_config(CONFIG_FILE)
        )
        if not df.empty:
            out_dir = run_from_configs.BASE_DIR
            csv_path = out_dir / "results_from_config.csv"
            df.to_csv(csv_path, index=False)
            # Optional: still generate standalone HTML plots on disk
            run_from_configs.make_plots(df)
            # Prepare data for frontend charts
            results = df.to_dict(orient="records")
            return jsonify(
                {
                    "success": True,
                    "message": "Configuration saved and benchmarks completed.",
                    "results_csv": str(csv_path),
                    "time_plot": str(out_dir / "time_vs_input_from_config.html"),
                    "memory_plot": str(out_dir / "memory_vs_input_from_config.html"),
                    "results": results,
                }
            )
        else:
            return jsonify(
                {
                    "success": True,
                    "message": "Configuration saved, but no benchmark data was produced.",
                }
            )
    except Exception as e:
        # If benchmarking fails, still report that config was saved
        return jsonify(
            {
                "success": False,
                "error": f"Configuration saved, but benchmark failed: {e}",
            }
        ), 500


if __name__ == "__main__":
    # install Flask once: pip install flask
    app.run(host="127.0.0.1", port=5000, debug=True)