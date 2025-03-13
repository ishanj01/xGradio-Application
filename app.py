import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
import io

# Function to answer user queries using Ollama
def ask_llm(question, df):
    """Generate an LLM response based on CSV content."""
    prompt = f"Analyze this CSV:\n\n{df.head(5).to_string()}\n\nQuestion: {question}"

    try:
        response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])

        # Debugging: Print raw response
        print("Raw LLM Response:", response)

        # Check if response contains 'error' key
        if 'error' in response:
            return f"LLM Error: {response['error']}"

        # Ensure response contains 'message' key and is correctly structured
        if isinstance(response, dict) and 'message' in response:
            return response['message']['content']
        
        return "Error: Unexpected response format from LLM."

    except Exception as e:
        return f"LLM API error: {e}"

# Function to plot graphs from the CSV file
def generate_plot(df, x_column, y_column):
    if x_column not in df.columns or y_column not in df.columns:
        return None, f"Error: '{x_column}' or '{y_column}' column not found in CSV."

    plt.figure(figsize=(8, 4))
    plt.plot(df[x_column], df[y_column], marker='o', linestyle='-')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} vs {x_column}")

    # Save plot in memory
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    return plot_path, None

# Function to process CSV and generate response
def process_csv(file, question, column_x, column_y):
    try:
        # Read CSV from uploaded file
        df = pd.read_csv(io.BytesIO(file))  # Read file directly as bytes
        
        # Answer the question using the CSV data
        llm_answer = ask_llm(question, df)
        
        # Generate graph
        plot_path, plot_error = generate_plot(df, column_x, column_y)

        return llm_answer, plot_path if plot_path else plot_error

    except Exception as e:
        return f"Error processing file: {str(e)}", None

# Gradio UI
def launch_app():
    with gr.Blocks() as app:
        gr.Markdown("# 📊 CSV Question Answering & Visualization App")
        
        file_input = gr.File(label="Upload CSV File", type="binary")  # Changed to 'binary'
        question_input = gr.Textbox(label="Ask a Question About the CSV")
        
        col_x_input = gr.Textbox(label="Enter X-axis Column Name")
        col_y_input = gr.Textbox(label="Enter Y-axis Column Name")
        
        submit_button = gr.Button("Submit")

        answer_output = gr.Textbox(label="LLM Answer")
        plot_output = gr.Image(label="Generated Graph")

        submit_button.click(
            fn=process_csv,
            inputs=[file_input, question_input, col_x_input, col_y_input],
            outputs=[answer_output, plot_output]
        )

    return app

if __name__ == "__main__":
    app = launch_app()
    app.launch()
