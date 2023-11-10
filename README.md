# PregnancyLoss-Prediction-in-DairyCow-Project
The code of paper: Transformer neural network to predict and interpret pregnancy loss from activity data in Holstein dairy cows
![Image text](https://github.com/lindan1128/PregnancyLoss-Prediction-Project/blob/main/Workflow.png)

## Installation
Install the required packages
    
    $ pip install --user --requirement requirements.txt
    
## Usage
    
    Attention-based LSTM: Train and evaluate the attention-based lstm model using the time-series activity and the metadata
    # Directory: LSTM
    $ python main.py --path=Data/data.csv --epoch=100
    
    Transformers: Train and evaluate the attention-based transformers model using the time-series activity and the metadata
    # Directory: Transformers
    $ python main.py --path=Data/data.csv --epoch=100
