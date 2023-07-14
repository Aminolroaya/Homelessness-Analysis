import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

if __name__ == '__main__':
    # Reading the relative ratio of occupation over capacity for a covid shelter 
    df =pd.read_csv('Anomal.csv')
    print(df.describe())
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date').reset_index()
    # Using Isolation Forest as an ML anomaly detection approach 
    model = IsolationForest(contamination=0.1)
    model.fit(df[['Ratio']])
    df['Outliers'] = pd.Series(model.predict(df[['Ratio']])).apply(lambda x: 'yes' if (x == -1) else 'no')
    print(df.query('Outliers=="yes"'))
    fig = px.scatter(df.reset_index(), x='Date', y='Ratio', color='Outliers')
    fig.update_xaxes(
        rangeslider_visible=True,
    )
    fig.show()