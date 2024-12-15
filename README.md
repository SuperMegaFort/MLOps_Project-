# Land Cover Classification with Satellite Imagery

This project demonstrates how to build a deep learning model to classify different types of land cover (e.g., urban areas, forests, water bodies) from satellite imagery. 

## Project Structure
├── .dvc
    ├── .gitignore
    └── config
├── .dvcignore
├── .github
    └── workflows
    │   └── mlops.yaml
├── .gitignore
├── .vscode
    └── settings.json
├── README.md
├── data
    ├── .gitignore
    └── raw.dvc
├── dvc.lock
├── dvc.yaml
├── evaluation
    ├── .gitignore
    └── plots
    │   └── .gitignore
├── google_cloud_fonction
    └── function-source (1)
    │   ├── main.py
    │   └── requirements.txt
├── params.yaml
├── requirements-freeze.txt
├── requirements.txt
└── src
    ├── evaluate.py
    ├── prepare.py
    ├── sentinelle.py
    ├── serve.py
    ├── train.py
    └── utils
        ├── __init__.py
        └── seed.py
