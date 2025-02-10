
# Setup
```
python -m venv .venv
source .venv/bin/activate
```
To get out type 
`deactivate`

```
pip install -r requirements.txt
```

```
python3
import main
model, history = main.train_month_predictor('photos')
main.predict_month(model, 'photos/PXL_20240820_155222742.jpg')
('August', np.int64(8), np.float32(0.24923027))
```

In my dataset downloaded from Google Photos most predictions were February.
Just the one above was not. Need to investigate further.


# VS Code 
The above setup gets vs code confused.
It's better to create the venv from inside VS code

Cmd+Shift+P 
Type "Python: Select Interpreter"
You will see option to create virtual env. (with reasonable default of doing a .venv sub-directory)
Then move on to selecting interpreter.

# Data
I went into google photos, selected a bunch of stuff and did "Download" from hamburger menu.
Then unzipped them into "photos" subfolder of this project.


