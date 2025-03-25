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
CREATE TABLE employees (
    emp_id TEXT PRIMARY KEY,
    name TEXT,
    age INTEGER,
    department_id TEXT,
    salary DECIMAL
);

CREATE TABLE departments (
    department_id TEXT PRIMARY KEY,
    department_name TEXT
);

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    department_id TEXT
);

CREATE TABLE employee_projects (
    emp_id TEXT,
    project_id TEXT,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
"""

# Function to generate SQL query
def generate_sql(query):
    system_prompt = f"""
    You are an advanced SQL query generator for a database with the following schema:

    {schema}

    ### Instructions:
    1. Generate **correct and optimized** SQL queries based on natural language questions.
    2. **Use proper JOINs** when necessary, especially for foreign key relationships.
    3. Select **only the relevant columns** instead of using `SELECT *`.
    4. Use **correct data types** (e.g., text values should be in single quotes: `'value'`).
    5. Ensure **column names match** the schema exactly.
    6. For queries involving comparisons (e.g., salary > 50000), ensure numeric fields are used correctly.
    7. Format SQL queries properly with indentation.
    8. use joins only when necessary. fi not require no need to use
    

    ### Example Queries and Answers:

    - **Question:** "Find the names of employees working in the 'IT' department."
      **SQL:**
      ```sql
      SELECT e.name 
      FROM employees e
      JOIN departments d ON e.department_id = d.department_id
      WHERE d.department_name = 'IT';
      ```

    - **Question:** "Retrieve the project names and their corresponding departments."
      **SQL:**
      ```sql
      SELECT p.project_name, d.department_name 
      FROM projects p
      JOIN departments d ON p.department_id = d.department_id;
      ```

    - **Question:** "List employees who earn more than 60,000."
      **SQL:**
      ```sql
      SELECT name, salary 
      FROM employees 
      WHERE salary > 60000;
      ```

    - **Question:** "Find employees who are working on projects."
      **SQL:**
      ```sql
      SELECT e.name, p.project_name 
      FROM employees e
      JOIN employee_projects ep ON e.emp_id = ep.emp_id
      JOIN projects p ON ep.project_id = p.project_id;
      ```

    - **Question:** "Count the number of employees in each department."
      **SQL:**
      ```sql
      SELECT d.department_name, COUNT(e.emp_id) AS employee_count 
      FROM employees e
      JOIN departments d ON e.department_id = d.department_id
      GROUP BY d.department_name;
      ```

    **Only generate SQL queries for English-language questions. Do not generate queries for unsupported languages or ambiguous queries.**
    """

    input_prompt = f"{system_prompt}\nUser Query: {query}"
    print(f"Processing input: {input_prompt}")  # Debugging
    
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL: {generated_text}")  # Debugging
    return generated_text

# Flask route for the frontend (HTML-based UI)
@app.route("/", methods=["GET", "POST"])
def index():
    generated_sql = ""
    if request.method == "POST":
        query_input = request.form.get("query_input")

        if query_input:
            generated_sql = generate_sql(query_input)
    
    return render_template("index.html", output_text=generated_sql)

# API endpoint for React frontend
@app.route("/generate-sql", methods=["POST"])
def generate_sql_route():
    data = request.json  # Read JSON data from frontend
    query_input = data.get("query_input", "")

    if query_input:
        generated_sql = generate_sql(query_input)
        return jsonify({"generated_sql": generated_sql})

    return jsonify({"error": "Invalid input"}), 400

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
