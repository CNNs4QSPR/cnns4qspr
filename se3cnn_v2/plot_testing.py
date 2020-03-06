import numpy as np
import pandas as pd
import plotly
import plotly.express as px

### 3D scatter plot of Iris dataset
# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')
# fig.show()

### 4D scatter plot
import plotly.express as px
df = px.data.iris()
print(df)
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='petal_length', symbol='species')
fig.show()
