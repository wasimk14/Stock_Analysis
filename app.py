from flask import Flask, render_template, request, send_from_directory
from flask import Response
import os
from stock_analyzer_core import run_analysis

app = Flask(__name__)

# Folder to store CSVs
REPORT_FOLDER = os.path.join(app.root_path, "static", "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    ticker = request.form.get("ticker", "").strip()

    # Auto-append ".NS" if user enters a plain symbol
    if ticker and "." not in ticker:
        ticker = ticker.upper() + ".NS"

    # Cleanup accidental double dots or spaces
    ticker = ticker.replace(" ", "").replace("..", ".")

    if not ticker:
        return render_template("index.html", error="Please enter a stock symbol.")

    try:
        df, summary, csv_path = run_analysis(ticker)

        # Move CSV into static/reports/
        final_name = os.path.basename(csv_path)
        final_path = os.path.join(REPORT_FOLDER, final_name)
        os.replace(csv_path, final_path)

        return render_template(
            "result.html",
            summary=summary,
            csv_file=final_name,
            price_plot=summary["price_plot"],
            volume_plot=summary["volume_plot"]
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/download/<filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

@app.route("/robots.txt")
def robots_txt():
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Sitemap: https://stock-oracle-468628049094.asia-south1.run.app/sitemap.xml"
    )
    return Response(content, mimetype="text/plain")

@app.route("/sitemap.xml")
def sitemap_xml():
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">

    <url>
        <loc>https://stock-oracle-468628049094.asia-south1.run.app/</loc>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>

</urlset>
"""
    return Response(content, mimetype="application/xml")

from flask import redirect

@app.errorhandler(405)
def method_not_allowed(e):
    return redirect("/")

@app.errorhandler(404)
def page_not_found(e):
    return redirect("/")



if __name__ == "__main__":
    app.run(debug=True)
