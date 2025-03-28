
# xGradio-based CSV Question Answering and Visualization Application

The purpose of this project is to develop a Gradio-based application that allows users to upload
a CSV file, ask questions—including numerical queries—about its contents, and receive
answers generated by a local Large Language Model (LLM). The application must also support
graph plotting, with all visualizations displayed within the Gradio interface.


## Code Breakdown

1. Importing Libraries


1.gradio → Used to create an interactive web UI.

2.pandas → Used for handling and processing CSV data.

3.matplotlib.pyplot → Used for generating visualizations (graphs).

4.ollama → Used to communicate with a local LLM (Llama 3.1).

5.io → Helps handle uploaded files in memory.

```bash
import gradio as gr 
import pandas as pd  
import matplotlib.pyplot as plt
import ollama 
import io 

```
2. LLM Query Function:

1.Takes a user question and a CSV dataframe (df). 

2.Extracts the first 5 rows of the CSV and formats it into a prompt.

3.Sends the prompt to Ollama (local LLM API).

4.Checks if the response is valid and handles errors.

5.Returns the model’s answer.

 ```bash
  def ask_llm(question, df):
    """Generate an LLM response based on CSV content."""
    prompt = f"Analyze this CSV:\n\n{df.head(5).to_string()}\n\nQuestion: {question}"

    try:
        response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])

        # Debugging: Print raw response
        print("Raw LLM Response:", response)

        # Handle errors
        if 'error' in response:
            return f"LLM Error: {response['error']}"

        if isinstance(response, dict) and 'message' in response:
            return response['message']['content']
        
        return "Error: Unexpected response format from LLM."

    except Exception as e:
        return f"LLM API error: {e}"

```



3. Plot Generation Function:

1.Checks if the given columns exist in the CSV.

2.Plots a graph of y_column vs x_column using Matplotlib.

3.Saves the plot as plot.png and returns the file path.

```bash
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
    plt.close()

    return plot_path, None

```

4. CSV Processing Function:

1.Reads the uploaded CSV file.

2.Uses Ollama to answer the user’s question.

3.Generates a graph for the selected columns.

4.Returns both the answer and the graph path.

```bash
  def process_csv(file, question, column_x, column_y):
    try:
        df = pd.read_csv(io.BytesIO(file))  # Read uploaded CSV
        
        llm_answer = ask_llm(question, df)  # Get answer from LLM
        
        plot_path, plot_error = generate_plot(df, column_x, column_y)  # Generate plot

        return llm_answer, plot_path if plot_path else plot_error  # Return both results

    except Exception as e:
        return f"Error processing file: {str(e)}", None

```

5. Gradio UI (Frontend):

1.Creates a Gradio interface using Blocks().

UI includes:

1.File upload for CSV.

2.Textbox for a user question.

3.Textbox for selecting X-axis and Y-axis columns.

4.Submit button to process data.

Output fields for:

LLM response.

Generated graph.


```bash
  def launch_app():
    with gr.Blocks() as app:
        gr.Markdown("# 📊 CSV Question Answering & Visualization App")
        
        file_input = gr.File(label="Upload CSV File", type="binary")  # Upload CSV
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


```

6. Running the Web App

Runs the app when the script is executed.

Launches the Gradio interface in the browser.

```bash
 if __name__ == "__main__":
    app = launch_app()
    app.launch()


```


🔹 Summary

1.Upload a CSV file.

2.Ask a question about the CSV (processed by LLM).

3.Select columns for graph visualization.

Get:

1.LLM-generated insights.

2.A graph based on CSV data

           
## UI-:

![2025-03-16 18 55 21](https://github.com/user-attachments/assets/04846eb2-8ffd-4a53-98ee-b493fe194f2a)


## Analysis:

Insight 1: Price is inversely proportional to area
Larger properties tend to be cheaper per square foot, suggesting that buyers might be prioritizing space over luxury.

Insight 2: Most houses have 3-4 stories, but only half have hot water heating and air conditioning
Buyers seem willing to sacrifice certain luxury features in favor of affordability or space.

Insight 3: Parking is a top priority for most buyers (80%)
Convenient parking availability significantly impacts desirability.

Insight 4: Most houses are semi-furnished or furnished, but some buyers prefer unfurnished properties
There is a mix of buyer preferences, suggesting demand for both furnished and unfurnished options.

Insight 5: The preferential area ("prefarea") is often mentioned alongside parking and furnishing status
Buyers consider the location along with amenities like parking and furnishing when making purchasing decisions.



