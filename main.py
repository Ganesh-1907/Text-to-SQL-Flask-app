from flask import Flask, request, render_template, jsonify
from flask_cors import CORS 
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)
model.eval()

# Predefined schema
schema = """
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    description TEXT,
    domain TEXT,
    team_lead TEXT,
    team_members TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT
);

CREATE TABLE labs (
    lab_id TEXT PRIMARY KEY,
    lab_name TEXT,
    department_id TEXT,
    incharge TEXT
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    lab_id TEXT,
    experiment_name TEXT,
    description TEXT
);

CREATE TABLE faculty (
    faculty_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    department_id TEXT,
    email TEXT,
    phone TEXT,
    research_area TEXT
);

CREATE TABLE patents (
    patent_id TEXT PRIMARY KEY,
    patent_name TEXT,
    inventor TEXT,
    filed_date TEXT,
    status TEXT
);

CREATE TABLE publications (
    publication_id TEXT PRIMARY KEY,
    title TEXT,
    author TEXT,
    journal TEXT,
    publication_date TEXT
);

CREATE TABLE internships (
    internship_id TEXT PRIMARY KEY,
    student_id TEXT,
    company TEXT,
    role TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT
);

CREATE TABLE workshops (
    workshop_id TEXT PRIMARY KEY,
    workshop_name TEXT,
    speaker TEXT,
    date TEXT,
    venue TEXT,
    department_id TEXT
);
"""

def generate_sql(query):
    input_prompt = f"tables: {schema} query for: {query}"
    print(f"Processing input: {input_prompt}")  # Debugging
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL: {generated_text}")  
    return generated_text



@app.route("/generate-sql", methods=["POST"])
def generate_sql_route():
    data = request.json  # Read JSON data from frontend
    query_input = data.get("query_input", "")

    if query_input:
        generated_sql = generate_sql(query_input)
        return jsonify({"generated_sql": generated_sql})

    return jsonify({"error": "Invalid input"}), 400

##for frontend within flask

@app.route("/", methods=["GET", "POST"])
def index():
    generated_sql = ""
    if request.method == "POST":
        query_input = request.form.get("query_input")

        if query_input:
            generated_sql = generate_sql(query_input)
    
    return render_template("index.html", output_text=generated_sql)


if __name__ == "__main__":
    app.run(debug=True)
