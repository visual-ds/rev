import altair_viewer as alv 
import altair as alt 

import pandas as pd 

def plot_chart(tsv_name): 
    """ 
    Plots a scatter plot with the 
    data in `tsv_name`. 
    """ 
    data = pd.read_csv(tsv_name) 
    print(data.columns) 

    chart = alt.Chart(data).mark_point(filled = True).encode(
                x = alt.X("value "), 
                y = alt.Y("metric"), 
                color = alt.Color("dataset"),  
                shape = alt.Shape("method") 
            ) 

    alv.show(chart) 

if __name__ == "__main__": 
    plot_chart("metrics/metrics.tsv") 
