
import openai
import pandas as pd
import csv
import pandas as pd
from io import StringIO
# Set your OpenAI API key
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key


import random

def read_csv_header(file_path):
    df = pd.read_csv(file_path)
    header = df.head()
    columns = df.columns
    description = df.describe()
    # print("Describe", description)
    generated_data = generate_model_data([header , columns , description])
    print(generated_data,type(generated_data))
    lines = generated_data.split("\n")
    generated_data = "\n".join(lines[:-1])
    print("-------------------------------------------")
    print(generated_data)
    # synthetic_data = pd.read_csv(StringIO(generated_data))
    # synthetic_data.to_csv("synthetic_data.csv")
    
    return generated_data

def generate_model_data(csv_data):
    prompt = f"""Generate 100 rows of data with columns: {', '.join(csv_data[0])}\n as a csv output refer  {', '.join(csv_data[2])}
                 strictly as csv without any missing values
    
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3000,  # Adjust as needed
        n=1,  # Number of responses to generate
        stop=None,  # You can specify a stop sequence to end the response
    )

    return response.choices[0].text.strip()







