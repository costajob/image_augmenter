# Table of Contents

* [Scope](#scope)
* [Setup](#setup)
  * [Versions](#versions)
  * [Virtualenv](#virtualenv)
  * [Installation](#installation)
  * [Tests](#tests)
* [Dataset](#dataset)
* [Labeller](#labeller)
* [Normalizer](#normalizer)
* [Augmenter](#augmenter)
* [Persister](#persister)


## Scope
The scope of this library is to augment the dataset for an image classification ML system.

## Setup

### Versions
The library is compatible with python `3.7`.

### Virtualenv
We suggest to isolate your installation via python virtualenv:
```shell
python3 -m venv .imgaug
...
source .imgaug/bin/activate
```

### Installation
Update `pip` package manager:
```shell
pip install pip --upgrade
...
pip install -r requirements.txt
```

### Tests
The library is covered, by fast, isolated unit and doc testing (the latter to grant reliable documentation):
```shell
python -m unittest discover -s imgaug -p '*'
...................
----------------------------------------------------------------------
Ran 19 tests in 0.898s

OK
```

## Dataset
The system is aimed to work with images of different sizes, saved as PNG or JPG files (supporting RGBA conversion).

## Labeller
The target label is extracted directly by inspecting the image name and trying to extract meaningful information (customisable).

```python
lbl = Labeller(digits=10)

lbl('resources/bag.png')
'bag'

lbl('resources/109-602-3906-001-c-suit-veletta-albino.jpg')
'1096023906'
```

## Normalizer
The images are normalized by:
- resizing them to the specified max size (default to 256 pixels)
- optionally applying a squared, transparent/backgound canvas and centering the image on it, thus avoiding any deformation

```python
norm = Normalizer(size=128, canvas=True)
img = norm('resources/bag.png')
img.shape
(128, 128, 4)
```

## Augmenter
The number of images is augmented by three orders of magnitude (depending on the cutoff attribute) by applying different transformations to the original one.  
Transformations are applied by using generators, thus saving memory consumption.

```python
aug = Augmenter(cutoff=.5)

aug.count
498

aug('resources/bag.png')
<generator object Augmenter.__call__ at 0x125354480>
```

## Persister
Images are persisted upon normalization and augmentation, by passing a `BytesIO` object to the specified action function. 
The persister supports iteration by yielding the image label and the function return value (typically the saved path), allowing to generate CSV files specific to cloud platforms (i.e. [Google Vision APIs](https://cloud.google.com/vision/automl/docs/prepare)).

```python
def persist(name, data):
    filename = f'temp/{name}'
    with open(filename, 'wb') as f:
        f.write(data.getvalue())
    return filename

pers = Persister('resources/skirt.jpg', action=perist)
for label, filename in pers:
    print(label, filename)
```
