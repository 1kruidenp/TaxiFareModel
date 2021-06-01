# imports
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data, set_X_Y, holdout
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
                            ('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())
                        ])
        
        time_pipe = Pipeline([
                            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))
                        ])
        preproc_pipe = ColumnTransformer([
                                        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                                        ('time', time_pipe, ['pickup_datetime'])
                                        ], remainder="drop")
        self.pipeline= Pipeline([
                        ('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())
                        ])
        #pipeline = Pipeline([
        pass

    def run(self):
        """set and train the pipeline"""
        #pipeline.fit(self.X, self.y)
        self.pipeline.fit(self.X, self.y)
        pass

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        #y_pred = pipeline.predict(X_test)
        score = compute_rmse(y_pred,y_test)
        print(score)
        return score
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
       # self.experiment_name="[PT] [Lisbon] [1kruidenp] TaxiFareModel + 0.0"
        try:
            return self.mlflow_client.create_experiment("[PT] [Lisbon] [1kruidenp] TaxiFareModel + 0.0")
        except BaseException:
            return self.mlflow_client.get_experiment_by_name("[PT] [Lisbon] [1kruidenp] TaxiFareModel + 0.0").experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key="LinearRegression", value=0):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key='rmse', value=5.9):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10000)
    
    # clean data
    df = clean_data(df)
    
    # set X and y
    X,y = set_X_Y(df)
    
    # hold out
    X_train, X_test, y_train, y_test = holdout(X,y)
    
    # train
    trainer=Trainer(X_train,y_train)
    trainer.set_pipeline()
    print(trainer.pipeline)
   
    trainer.run()
    
    # evaluate
    trainer.evaluate(X_test,y_test)
    
    experiment_id = trainer.mlflow_experiment_id
    trainer.mlflow_run
    trainer.mlflow_log_param()
    trainer.mlflow_log_metric()
    

    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
