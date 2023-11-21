## Citation
If you find this project useful in your research, please consider cite:
		
     Lin D, Kenéz Á, McArt J A A, et al. Transformer neural network to predict and interpret pregnancy loss from activity data in Holstein dairy cows[J]. Computers and Electronics in Agriculture, 2023, 205: 107638.

The paper could be found [here](https://www.dropbox.com/scl/fi/dcu46on9w03i1f6yoves7/Comparative-Genomics-Reveals-Recent-Adaptive-Evolution-in-Himalayan-Giant-Honeybee-Apis-laboriosa.pdf?rlkey=ydw1cn06e1pn16dd84lykx0dv&dl=0](https://www.dropbox.com/scl/fi/95y0rd775ms1sqvc5ey7e/Transformer-neural-network-to-predict-and-interpret-pregnancy-loss-from-activity-data-in-Holstein-dairy-cows.pdf?rlkey=52sqpr37ks59l2n0nxs0a7fvf&dl=0)


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
