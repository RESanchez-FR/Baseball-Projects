This Model is dedicated to predicting Homeruns for the 2025 season but can adjusted for future seasons. 

Libraires that are needed for this analysis are Scikit-learn, Numpy, Pandas, and Pybasebaall

The Homerun features of interest for training our model are:
        Age, AB (At Bats), SLG (Slugging), FB% (Fly Ball Percentage),
        HR/FB (HomeRun/Fly Ball Ratio) , K% (Strike out Rate), wRC+ (Weighted Runs Created Plus),
        Contact% (Contact Rate), ISO (Isolated Power),
        HardHit% (Hard Hit Rate), Barrel% (Barrel Rate), LA (Launch Angle), EV (Exit Velocity),
        Rolling_HR (New Feature)

We run four different models:
        Ridge, 
        Lasso, 
        ElasticNet,  
        RandomForest
        
To account for the uncertainty of the game of baseball we run with the best model that doesn't overfit. 

A CSV file is created to save the predicted stats. 



The Next Project is Quantifying Pitching Deception. We look at four different metrics that I consider to be useful to quantifying deception.

## updated version coming soon to fix biases
