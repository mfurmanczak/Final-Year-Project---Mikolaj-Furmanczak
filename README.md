# Final Year Project - Mikolaj Furmanczak

This is my final year project, focused on training a CNN on the UrbanSound8K dataset using Tensorflow. 


## Installation

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

In addition, you will need to download the UrbanSound8K dataset from the official website (http://urbansounddataset.weebly.com/urbansound8k.html) and place the extracted folder in the main directory of the repository.


## Scripts

The project includes the following scripts:

- `data_visualization.py`: This script provides visualizations of the audio data.
- `data_processing.py`: This script preprocesses the audio data.
- `main.py`: This script trains the model.
- `results_analysis.py`: This script shows the results of the training.

And the `history` folder where the history of 3 proposed models is included.


## To see the description and various visualizations of the dataset
```
python data_visualization.py
```


## To train the model

Preprocesss the data
```
python data_processing.py
```

Train the model
```
python main.py
```

## To view the results
```
python results_analysis.py
```

## License

This project is licensed under the [MIT LICENSE](LICENSE.md).
