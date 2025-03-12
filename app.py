from flask import Flask, request, render_template
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize Flask App
app = Flask(__name__)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)
model.eval()

def generate_sql(input_prompt):
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/main", methods=["GET", "POST"])
def home():
    generated_sql = ""
    if request.method == "POST":
        user_input = request.form.get("text_input")
        if user_input:
            generated_sql = generate_sql(user_input)
    
    return render_template("main.html", output_text=generated_sql)

@app.route("/", methods=["GET", "POST"])  
def index():
    generated_sql = ""
    if request.method == "POST":
        schema_input = request.form.get("schema_input")
        query_input = request.form.get("query_input")

        if schema_input and query_input:
            combined_input = f"Schema: {schema_input} | Query: {query_input}"
            generated_sql = generate_sql(combined_input)  # Pass combined input
    
    return render_template("index.html", output_text=generated_sql)

if __name__ == "__main__":
    app.run(debug=True)
