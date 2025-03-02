# Motion in three party meetings


# Setup
```
conda update conda

conda create --name nameEnv python=3.11
conda activate nameEnv

conda install conda-forge::opencv
conda install conda-forge::dlib

conda install pytorch::pytorch
conda install pytorch::torchvision
conda install conda-forge::matplotlib
conda install conda-forge::av
conda install conda-forge::tqdm
conda install anaconda::pandas

conda install ipykernel
```


# Authors
- Andrea Cucchietti
- Kevin Karim
- Paul Gérard Bernard Leroux

# License notes
The weights used to extract the mouth and jaw positions in order to remove their optical flow were trained over the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/). 

```
C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
300 faces In-the-wild challenge: Database and results. 
Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
```

Their license exclude commercial use. You should contact a lawer or talk to Imperial College London in order to know if it is possible to use motion features in the `data` folder for your use case.