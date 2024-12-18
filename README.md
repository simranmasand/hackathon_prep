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
<table>
  <tr>
    <td><a href="assets/SS_1.png" target="_blank"><img src="assets/SS_1.png" alt="Screenshot 1" width="200"/></td>
    <td><img src="assets/SS_2.png" alt="Screenshot 2" width="200"/></td>
  </tr>
  <tr>
    <td><img src="assets/plot_output.png" alt="Plot Output" width="200"/></td>
    <td><img src="assets/doccomparison.png" alt="Doc Comparison" width="200"/></td>
  </tr>
  <tr>
    <td><img src="assets/BoECPITable.png" alt="Screenshot 3" width="200"/></td>
    <td><img src="assets/PlotOutputCPI.png" alt="Screenshot 4" width="200"/></td>
  </tr>
</table>

### Demo app
![alt text](assets/streamlit-main-2024-11-04-15-11-02_light.gif)

