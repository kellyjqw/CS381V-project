import os
from openai import OpenAI
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/facebookresearch/VidOSC/refs/heads/main/data_files/osc_split.csv")
tasks = list(df["osc"])
tasks = [task.replace("_", " ") for task in tasks]

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

count = 1
output_file = "../data_samples/descriptions.csv"
write_header = True
for task in tasks:
    print(f"GENERATING RESULT {count}\n")

    prompt_progress = f"""
    Suppose you are a professional video dataset annotator and have been asked to succinctly
    describe in under 77 words while still being precise and informative: the start or intermediate 
    states of {task} (including but not limited to attributes like shape, size, color, consistency, etc.). 
    What does it look like when {task} is incomplete or still in progress? For the person 
    engaging in {task}, what might they be doing? Do not mention anything about the end state. 
    """
    response_progress = client.responses.create(
        model="gpt-5-nano",
        input=prompt_progress
    )
    output_progress = response_progress.output_text
    print(output_progress, "\n")
    
    prompt_done = f"""
    Suppose you are a professional video dataset annotator and have been asked to succinctly
    describe in under 77 words while still being precise and informative: the end state of 
    {task} (including but not limited to attributes like shape, size, color, consistency, etc.). 
    What does it look like when {task} is done? For the person finished with {task}, what 
    might they be doing?
    """
    response_done = client.responses.create(
        model="gpt-5-nano",
        input=prompt_done
    )
    output_done = response_done.output_text
    print(output_done, "\n\n")

    row_df = pd.DataFrame([[task.replace(" ", "_"), output_progress, output_done]], 
                          columns=["osc", "progress_description", "finished_description"])
    row_df.to_csv(output_file, mode="a", header=write_header, index=False)

    write_header = False
    count += 1