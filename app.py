from flask import Flask, render_template, request, flash, jsonify

app = Flask(__name__)

@app.route('/', methods=["GET"])
def index():
    if request.method == "GET":
        return render_template('index.html')

@app.route("/about", methods=["GET"])
def about():
    if request.method == "GET":
        return render_template('about.html')

@app.route("/projects", methods=["GET"])
def projects():
    if request.method == "GET":
        return render_template('projects.html')

projects = {
    "1": {
        "title": "--",
        "description": "__",
        "ghlink": "--"
    },
    "2": {
            "title": "--",
            "description": "__",
            "ghlink": "--"
    }
}

@app.route("/project/<id>", methods=["GET"])
def project(id):
    if request.method == "GET":
        if id in projects:
            return render_template("project.html", project=projects[id])
        else:
            return "Project not found", 404

if __name__ == '__main__':
    app.run(debug=True)

