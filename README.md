# Hack Prep

Repository for hack prep.


> **Warning**
> Due to token limitations, it is highly recommended to only choose relevant files.


## Desired Goal State
![GoalState.png](assets%2FGoalState.png)

### Current Document RAG State
![RAG Goal State.png](assets%2FRAG%20Goal%20State.png)

## Installation

PDFNinja has certain dependencies and was compiled in Python3.9 but shoudl have backward and forward compatibility. The other dependencies are listed in requirements.csv
```bash
$ pip install -r requirements.txt
```

## Usage
You can call the app right from the command line interface. 

### Pre-processing

[//]: # (1. Obtain an API key from OpenAI.)

[//]: # (2. Store it as a string in a .txt file)

[//]: # (3. This is parsed to the program using the --apikey_filepath argument.)

[//]: # (4. Have the folder where you want to search for pdf documents as a directory path as well. This will be parsed to the program using --documents_path argument.)

### Run from terminal

```bash
streamlit run main.py
# returns the workflow with default args in streamlit
```

### See an interactive of this below
![alt text](assets/SS_1.png)
![alt text](assets/SS_2.png)
![plot_output.png](assets%2Fplot_output.png)
![doccomparison.png](assets%2Fdoccomparison.png)

### Demo app
![alt text](assets/streamlit-main-2024-11-04-15-11-02_light.gif)

