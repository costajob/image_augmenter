# Table of Contents

* [Scope](#scope)
* [Setup](#setup)
  * [Versions](#versions)
  * [Virtualenv](#virtualenv)
  * [Installation](#installation)
  * [Tests](#tests)
* [APIs](#apis)
  * [Labeller](#labeller)
  * [Normalizer](#normalizer)
  * [Augmenter](#augmenter)
  * [Persister](#persister)


## Scope
The scope of this library is to augment the dataset for an image classification ML system.

## Setup

### Versions
The library is compatible with python `3.6` on.

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
```

## APIs
The library is composed by different collaborators, each with its specific responsibility.
Each class tries to expose a minimal public APIs in the form of `__call__` or `__iter__` methods (when generators are used).  
The classes are aimed to work with one image at time, in case you need to transform and augment multiple images, avoid creating multiple instances of the classes, just change the argument of the `__call__` function (but for `Persister`, which need a new instance and/or instance attribute modification).

### Labeller
The target label is extracted directly by inspecting the image name and trying to extract meaningful information (customisable).

```python
lbl = Labeller(digits=10)

lbl('resources/bag.png')
'bag'

lbl('resources/109-602-3906-001-c-suit-veletta-albino.jpg')
'1096023906'
```

### Normalizer
The images are normalized by:
- resizing them to the specified max size (default to 256 pixels)
- optionally applying a squared, transparent/backgound canvas and centering the image on it, thus avoiding any deformation

```python
norm = Normalizer(size=128, canvas=True)
img = norm('resources/bag.png')
img.shape
(128, 128, 4)
```

### Augmenter
The number of images is augmented by three orders of magnitude (depending on the cutoff float attribute) by applying different transformations to the original one.  
Transformations are applied by using generators, thus saving memory consumption.

```python
aug = Augmenter(cutoff=1.)

aug.count
1000

aug('resources/bag.png')
<generator object Augmenter.__call__ at 0x125354480>
```

### Persister
Images are persisted upon normalization and augmentation, by specifying an action function that accepts the name of the file (original basename suffixed by an index) and a `BytesIO` object containing the image data stream.  
The persister supports both a filename path and, optionally, a stream-like object (in case the file is not yet persisted to disk).  
The persister supports iteration by yielding the image label and the function return value (typically the saved path), allowing to generate CSV files specific to cloud platforms (i.e. [Google Vision APIs](https://cloud.google.com/vision/automl/docs/prepare)).

```python
def persist(name, stream):
    filename = f'temp/{name}'
    with open(filename, 'wb') as f:
        f.write(stream.getvalue())
    return filename

pers = Persister('resources/skirt.jpg', action=perist)
for label, filename in pers:
    print(label, filename)
```
