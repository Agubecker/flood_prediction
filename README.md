# Floods prediction in Santiago Metropolitan region - Chile


## Description

A RNN deep learning model that predicts floodings in given points of rivers Mapocho and Maipo in the Metropolitan region of Santiago, Chile.

The model tells us the river discharge or flow values for the next 24 hours, indicating wether it is a level that should be considered a warning level or a flooding level.


### Training

Our model was trained with information from two different API´s. Such information consisted of different meteorological (https://api.open-meteo.com/v1/forecast) and rivers discharge data (https://flood-api.open-meteo.com/v1/flood).

By training a deep learning Recurrent Neural Network with data such as precipitation, soil moisture, pressure, etc. our model learned to predict rivers discharge levels with different thresholds to indicate normal, warning and dangerous levels.

The model´s interface is quite straightforward and only requires for the users to click a button to make a forecast.

## Getting Started

### Dependencies

This project can be run in any OS and the required dependencies can be found in our requirements.txt file (https://github.com/Agubecker/flood_prediction/blob/master/requirements.txt).

Once the repository is downloaded, this requirements can be installed through terminal:

```bash
pip install -r requirements.txt

```

### Installing

There's no need to install anything other than the required dependencies mentioned above.

### Executing program

You can run a prediction locally from your terminal by executing:

```bash

make run_pred

```
You can also access our API at https://flood-pred-intel-idtyvgmhca-vp.a.run.app/ and run a prediction by adding "/forecast" to the URL

And finally you can access our model's website at: https://flood-prediction.streamlit.app/

## Authors

Contributors names and contact info

[@Agustin Becker-Github's](https://github.com/Agubecker)

[@Agustin Becker-Linkedin's](https://www.linkedin.com/in/agust%C3%ADn-becker-queirolo-313733104/)

[@Valentin Radovich-Github's](https://github.com/Valenradovich)

[@Valentin Radovich-Linkedin's](https://www.linkedin.com/in/valentin-fernandez-radovich/)

[@Matias Duarte-Github's](https://github.com/matiasduarte86)

[@Matias Duarte-Linkedin's](https://www.linkedin.com/in/matias-martin-duarte86/)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [MIT] License
