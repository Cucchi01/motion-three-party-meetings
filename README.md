# Motion in three party meetings
This repository includes an analysis of motions in the context of 3-party Zoom meetings using the MEET corpus (The MEET Corpus: Collocated, Distant and Hybrid Three-party Meetings with a Ranking Task).  
The study is based on the intuitive belief that motion provides insights on speech dialogues. In particular, it wants to address:  
- Is it possible to predict the person speaking considering only the overall body motion?  
- Is there a correlation between overall body motion and the tasks performed?  

It is used the average optical flow frame by frame as a measure of overall body motion motion.

# Project structure
The directories `data\annotations`, `data\MEETdata-group13`, `data\video`, `img`, and `weights` are not present because they contain data not publicly available.  
It is possible to see directly the results in the notebook files.  

To run the various scripts from scratch the steps are:
- follow the Setup section below
- add the Zoom videos in `data\MEETdata-group13\videos\`
- run `scripts\divide_sources.py` to separate each channels in the Zoom meeting and then manually filter the channels not needed in `data\video\channels\` 
- (in case of layout changes in the videos run `scripts\combine_channels.py` modifying the variables at the beginning)
- run `scripts\optical_flow_pytorch.py`
- extract json files from Elan and save them in `annotations/`
- run the code in `notebooks\data_analysis.ipynb`
- run the code in `notebooks\tasks.ipynb`


```cmd
.
│   .gitignore
│   LICENSE
│   README.md
│
├───data
│   │
│   ├───annotations
│   │   │   SESS00x_Focus.json // focus annotations extracted from Elan
│   │   │   SESS00x_Turns.json // turns annotations extracted from Elan
│   │   │
│   │   └───parsed
│   │           aggregated_data_speaking.csv
│   │           aggregated_Turns_RAFT.csv // data used for question 1
│   │           aggregated_Focus_RAFT.csv // data used for question 2
│   │           SESS00x_Focus.csv:
│   │           SESS00x_Turns.csv
│   │           SESS00x_Turns_bl.csv
│   │           SESS00x_Turns_br.csv
│   │           SESS00x_Turns_tl.csv
│   │
│   ├───features
│   │   └───motions // motions extracted from the channels
│   │           aggregated.csv
│   │           SESS00x_digital_\[GAME\]\_\[position\]\_\[method\].csv
│   │
│   ├───MEETdata-group13
│   │   │
│   │   └───videos # Original videos
│   │           SESS00x_digital_\[GAME\].mp4
│   │
│   └───video
│       └────channels // Channels extracted from the videos
│               SESS00x_digital_\[GAME\]_\[Position\].mp4
│
├───img
│
├───notebooks
│       data_analysis.ipynb // notebook containing a data analysis of the average optical flow and extraction of aggregated_Focus_RAFT.csv and aggregated_Turns_RAFT.csv 
│       tasks.ipynb // analysis of the two tasks
│
├───scripts
│       combine_channels.py // merges parts of two channels
│       constants.py // constants for the project
│       divide_sources.py // divides the original videos in the single channels
│       optical_flow.py // extracts the avg optical flow with Farneback's alg 
│       optical_flow_pytorch.py // extracts the avg optical flow with RAFT
│       parse_json.py // parses the json extracted from the Elan files to csv
│
└───weights
        shape_predictor_68_face_landmarks.dat # weights used for mouth detection

```


# Setup
```
conda update conda

conda create --name nameEnv python=3.11
conda activate nameEnv

conda install -c conda-forge opencv dlib matplotlib av tqdm seaborn
conda install -c pytorch pytorch torchvision
conda install -c anaconda pandas ipykernel scipy scikit-learn 
```

# License notes
The weights used to extract the mouth and jaw positions in order to remove their optical flow were trained over the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/). 

```
C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
300 faces In-the-wild challenge: Database and results. 
Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
```

Their license exclude commercial use. You should contact a lawer or talk to Imperial College London in order to know if it is possible to use motion features in the `data` folder for your use case.