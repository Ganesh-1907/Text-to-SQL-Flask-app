from flask import Flask, request, render_template,jsonify
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

def generate_sql(schema, query):
    input_prompt = f"tables: {schema} query for: {query}"
    print(f"Processing input: {input_prompt}")  # Debugging
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL: {generated_text}")  # Debugging
    return generated_text



@app.route("/", methods=["GET", "POST"])
def index():
    generated_sql = ""
    if request.method == "POST":
        schema_input = request.form.get("schema_input")
        query_input = request.form.get("query_input")

        if schema_input and query_input:
            generated_sql = generate_sql(schema_input, query_input)
    
    return render_template("index.html", output_text=generated_sql)

@app.route("/generate-sql", methods=["POST"])
def generate_sql_route():
    data = request.json  # Read JSON data from frontend
    schema_input = data.get("schema_input", "")
    query_input = data.get("query_input", "")

    if schema_input and query_input:
        generated_sql = generate_sql(schema_input, query_input)
        return jsonify({"generated_sql": generated_sql})

    return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(debug=True)
