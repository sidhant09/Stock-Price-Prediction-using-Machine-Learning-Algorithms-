import pandas as pd
import nvd3
from IPython.display import Image
from IPython.core.display import display, HTML
from nvd3 import lineChart
#
df = pd.read_excel(r"lg_micro.xlsx")
df = pd.DataFrame(df)
#
output_file = open('micro.html', 'w')
chart = lineChart(width = 1000,name="lineChart", x_is_date=False, x_axis_format=None)
#
xdata = df['features']
y1 = df['SVM_7']
y2 = df['SVM_31']
y3 = df['NN_7']
y4 = df['NN_31']
#
extra_serie = {"tooltip": {"y_start": "There are ", "y_end": " calls"}}
chart.add_serie(y=y1, x=xdata, name='SVM 7 years', extra=extra_serie)
extra_serie = {"tooltip": {"y_start": "", "y_end": " min"}}
chart.add_serie(y=y2, x=xdata, name='SVM 31 years', extra=extra_serie)
extra_serie = {"tooltip": {"y_start": "There are ", "y_end": " calls"}}
chart.add_serie(y=y3, x=xdata, name='Neural Network 7 years', extra=extra_serie)
extra_serie = {"tooltip": {"y_start": "There are ", "y_end": " calls"}}
chart.add_serie(y=y4, x=xdata, name='Neural Network 31 years ', extra=extra_serie)
#
chart.set_containerheader("<h2>Accuracy rate for Microsoft stock price prediction <h2>")
chart.buildhtml()
display(HTML(chart.htmlcontent))
output_file.write(chart.htmlcontent)
output_file.close()