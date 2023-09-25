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

## Demo
[video.webm](https://github.com/Agubecker/flood_prediction/assets/86897297/50129c6d-89f7-43bb-8470-3b079e874f65)

## Model performance
![model_work](https://github.com/Agubecker/flood_prediction/assets/86897297/dd313967-ac91-4f24-b07e-6e538fd5e714)
![model_zoom](https://github.com/Agubecker/flood_prediction/assets/86897297/cef1d4e2-e153-4340-803c-ecde13789b76)

## Features Engineering: Before and after

![vel_x_dir](https://github.com/Agubecker/flood_prediction/assets/86897297/7092a359-3ae8-4c04-bc65-9f67e8aecb27)
![x_y](https://github.com/Agubecker/flood_prediction/assets/86897297/5362322e-3bf0-4622-a7bc-15af9bd08a49)





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
