# Land Cover Classification with Satellite Imagery

This project demonstrates how to build a deep learning model to classify different types of land cover (e.g., urban areas, forests, water bodies) from satellite imagery. 

## Project Structure
    ├── .dvc  
    │   ├── .gitignore                
    │   └── config                   
    ├── .dvcignore                    
    ├── .github  
    │   └── workflows  
    │       └── mlops.yaml           
    ├── .gitignore                   
    ├── .vscode  
    │   └── settings.json            
    ├── README.md                    
    ├── data  
    │   ├── .gitignore                
    │   └── raw.dvc                   
    ├── dvc.lock                      
    ├── dvc.yaml                     
    ├── evaluation  
    │   ├── .gitignore                
    │   └── plots  
    │       └── .gitignore            
    ├── google_cloud_fonction  
    │   └── function-source (1)       # Directory for the Google Cloud Function source code.  
    │       ├── main.py               # Entry point for the serverless function.  
    │       └── requirements.txt      # Python dependencies for the Cloud Function.  
    ├── params.yaml                   # YAML file containing hyperparameters and configurations.  
    ├── requirements-freeze.txt       # Frozen list of Python dependencies for exact replication.  
    ├── requirements.txt              # List of Python dependencies for the project.  
    └── src  
        ├── evaluate.py               # Script for evaluating the model’s performance.  
        ├── prepare.py                # Script for preparing and processing data.  
        ├── sentinelle.py             # Simulates satellite image generation and collection.  
        ├── serve.py                  # Script for serving the model via an API.  
        ├── train.py                  # Script for training the machine learning model.  
        └── utils  
            ├── __init__.py          
            └── seed.py               # Utility for setting random seed for reproducibility.  
