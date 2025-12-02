import os
from openai import OpenAI
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/facebookresearch/VidOSC/refs/heads/main/data_files/osc_split.csv")
tasks = list(df["osc"])
tasks = [task.replace("_", " ") for task in tasks]

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

index = 0   # CHECK GENERATED CSV FILE BEFORE SETTING THIS
output_file = "../data_samples/multiclass_descriptions.csv"
for i in range(index, len(tasks)):
    task = tasks[i]
    print(f"GENERATING RESULT {index + 1}: {task}\n")

    prompt_prefix = """Suppose you are a professional video dataset annotator and have been asked to succinctly
    describe in under 80 words while still being precise and informative"""

    prompt_initial = f"""{prompt_prefix}: the initial state of {task} (including but not limited to attributes
    like shape, size, color, consistency, etc.). What does it look like when {task} is just starting? For the
    person beginning {task}, what might they be doing? Do not mention anything about the intermediate or end 
    states of {task}. 
    """
    response_initial = client.responses.create(
        model="gpt-5-nano",
        input=prompt_initial
    )
    output_initial = response_initial.output_text
    print(output_initial, "\n")

    prompt_transition = f"""{prompt_prefix}: the intermediate states of {task} (including but not limited to attributes 
    like shape, size, color, consistency, etc.). What does it look like after {task} has been started 
    and is currently in progress? For the person engaging in {task}, what might they be doing? 
    Do not mention anything about the start or end states of {task}. 
    """
    response_transition = client.responses.create(
        model="gpt-5-nano",
        input=prompt_transition
    )
    output_transition = response_transition.output_text
    print(output_transition, "\n")
    
    prompt_done = f"""{prompt_prefix}: the end state of {task} (including but not limited to 
    attributes like shape, size, color, consistency, etc.). What does it look like when {task} 
    is done? For the person finished with {task}, what might they be doing? Do not mention anything
    about the start or intermediate states of {task}. 
    """
    response_done = client.responses.create(
        model="gpt-5-nano",
        input=prompt_done
    )
    output_done = response_done.output_text
    print(output_done, "\n\n")

    row_df = pd.DataFrame([[task.replace(" ", "_"), output_initial, output_transition, output_done]], 
                          columns=["osc", "initial_description", "transition_description", "finished_description"])
    row_df.to_csv(output_file, mode="a", header=(index == 0), index=False)
    index += 1